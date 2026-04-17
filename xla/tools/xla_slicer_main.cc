#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <queue>
#include <fstream>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/container/flat_hash_set.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_verifier.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tools/hlo_extractor.h" 
#include "xla/tools/hlo_slicer.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/path.h"
#include "xla_slicer.h"

namespace {
const char* const kUsage = R"(
TODO
Usage:
  bazel run //xla/tools:xla_slicer -- \
    --input_format=[hlo|mhlo|pb|pbtxt|stablehlo] \
    path/to/module
)";

}  // namespace

namespace xla {

namespace {

absl::Status RunXlaSlicer(const XlaSlicerConfig& opts) {
  std::string format = opts.input_format;
  if (format.empty()) {
    format = std::string(tsl::io::Extension(opts.input_file));
  }
  TF_ASSIGN_OR_RETURN(auto module, LoadModuleFromFile(opts.input_file, format));

  HloVerifier verifier(
      HloVerifierOpts{}.WithLayoutSensitive(false).WithAllowMixedPrecision(
          true));
  TF_RETURN_IF_ERROR(verifier.Run(module.get()).status());
  
  // TODO: your logic goes here
	// ==========================================
  // 核心逻辑开始：基于深度和数量限制的自定义后向切片
  // ==========================================

  // 第一步：全局扫描，建立白名单子计算集合
  // 将 kReduce, kReduceWindow, kMap, kSort 的内部计算函数加入白名单
  absl::flat_hash_set<const HloComputation*> whitelisted_comps;
  for (const HloComputation* comp : module->computations()) {
    for (const HloInstruction* inst : comp->instructions()) {
      if (inst->opcode() == HloOpcode::kReduce ||
          inst->opcode() == HloOpcode::kReduceWindow ||
          inst->opcode() == HloOpcode::kMap ||
          inst->opcode() == HloOpcode::kSort) {
        for (const HloComputation* called_comp : inst->called_computations()) {
          whitelisted_comps.insert(called_comp);
        }
      }
    }
  }

  HloComputation* entry_comp = module->entry_computation();
  int slice_index = 0;

  // 第二步：遍历 main 函数中的每一条指令，作为切片的 ROOT
  for (HloInstruction* root_inst : entry_comp->instructions()) {
    
    // 初始化 BFS 队列和有效节点集合
    // 队列中存储对：{指令指针, 当前深度}
    std::queue<std::pair<const HloInstruction*, int>> bfs_queue;
    absl::flat_hash_set<const HloInstruction*> valid_set;

    bfs_queue.push({root_inst, 0});
    valid_set.insert(root_inst);

    // 第三步：执行严格管控的 BFS
    while (!bfs_queue.empty()) {
      auto [curr_inst, depth] = bfs_queue.front();
      bfs_queue.pop();

      // 如果当前指令的深度已经达到 5，停止向其 operands 扩展
      if (depth >= 5) {
        continue;
      }

      for (const HloInstruction* operand : curr_inst->operands()) {
        // 如果节点已被访问过，跳过
        if (valid_set.contains(operand)) {
          continue;
        }

        // 拦截逻辑：直接拒绝不需要的控制流或非白名单 Call
        if (operand->opcode() == HloOpcode::kWhile ||
            operand->opcode() == HloOpcode::kConditional ||
            operand->opcode() == HloOpcode::kCall) {
          continue; // 不加入 valid_set，也不继续推入队列
        }

        // 如果指令总数已满 15，停止吸收新节点
        if (valid_set.size() >= 15) {
          break; 
        }

        // 验证通过，加入集合并推入下一层队列
        valid_set.insert(operand);
        bfs_queue.push({operand, depth + 1});
      }
      
      // 再次检查，防止内部循环 break 后外层队列继续处理
      if (valid_set.size() >= 15) {
        break;
      }
    }

    // 第四步：构造自定义的 Extract 拦截器并提取 Module
    auto extract_selector = [&](const HloInstruction* inst) -> bool {
			if (inst->opcode() == HloOpcode::kConstant) {
        return true;
      }
			
      // 如果指令在主函数中，严格遵循 BFS 计算出的 valid_set
      if (inst->parent() == entry_comp) {
        return valid_set.contains(inst);
      }
      // 如果指令在白名单运算的子图内部，无条件保留
      return whitelisted_comps.contains(inst->parent());
    };

    auto replace_type_selector = [](const HloInstruction* inst) {
      // 任何未被选中的指令，直接原地替换为全 0
      return ReplaceType::kReplaceParam; 
    };

    // 调用 XLA 底层的 ExtractModule
    auto extracted_module = ExtractModule(
        /*instruction=*/root_inst, 
        /*height=*/-1, // 高度限制交由我们的 extract_selector 控制
        /*extract_selector=*/extract_selector, 
        /*replace_type_selector=*/replace_type_selector, 
        /*cross_computation=*/true);

    // 第五步：将切片序列化并输出到独立文件中
    // 文件名格式：sliced_0_opname.hlo
    std::string safe_name = root_inst->ToString();
    // 替换文件名中可能存在的非法字符 (如 % 或 .)
    for (char& c : safe_name) {
      if (c == '%' || c == '/' || c == '\\') c = '_';
    }
    
    std::string output_filename = absl::StrCat("sliced_", slice_index, "_", safe_name, ".hlo");
    std::ofstream out_file(output_filename);
    if (out_file.is_open()) {
      out_file << extracted_module->ToString();
      out_file.close();
      std::cout << "Successfully saved slice to: " << output_filename << " (Instructions: " << valid_set.size() << ")" << std::endl;
    } else {
      std::cerr << "Failed to open file for writing: " << output_filename << std::endl;
    }

    slice_index++;
  }
  
  return absl::OkStatus();
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  xla::XlaSlicerConfig opts;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("input_format", &opts.input_format,
                "The format of the input file. Valid values:\n"
                "  hlo : HLO textual format\n"
                "  mhlo : MHLO in textual or bytecode format\n"
                "  pb : xla::HloProto in binary proto format\n"
                "  pbtxt : xla::HloProto in text proto format\n"
                "  stablehlo : StableHLO in textual or bytecode format")};
  // The usage string includes the message at the top of the file and the flags
  // defined above.
  const std::string kUsageString =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[0], flag_list));

  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  if (!parse_ok) {
    // Print the usage using cerr to avoid truncation by LOG.
    std::cerr << kUsageString;
    return 1;
  }
  tsl::port::InitMain(kUsageString.c_str(), &argc, &argv);

  QCHECK(argc == 2) << "Must specify a single input file. Number of args: "
                    << argc;
  opts.input_file = argv[1];

  absl::Status status = xla::RunXlaSlicer(opts);
  if (!status.ok()) {
    std::cerr << status << std::endl;
    return 1;
  }
  return 0;
}