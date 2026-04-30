#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <queue>
#include <fstream>
#include <cstdlib> // 新增：为了使用 setenv 屏蔽底层日志

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/service/hlo_verifier.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tools/hlo_extractor.h" 
#include "xla/tools/hlo_slicer.h"
#include "xla/shape_util.h"
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

int64_t ComputeParameterElements(const HloModule* module) {
  int64_t total = 0;
  for (const HloInstruction* instr :
       module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kParameter) {
      total += ShapeUtil::ElementsInRecursive(instr->shape());
    }
  }
  return total;
}

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
  
  // ==========================================
  // 核心逻辑开始：基于深度和数量限制的自定义后向切片
  // ==========================================

  // 第一步：全局扫描，建立白名单子计算集合
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
  
  // fingerprint → (best_module, best_total_param_elements)
  absl::flat_hash_map<uint64_t,
      std::pair<std::unique_ptr<HloModule>, int64_t>> best_slices;
  std::vector<uint64_t> fingerprint_order;

  // 第二步：遍历 main 函数中的每一条指令，作为切片的 ROOT
  for (HloInstruction* root_inst : entry_comp->instructions()) {
    
    // 初始化 BFS 队列和有效节点集合
    std::queue<std::pair<const HloInstruction*, int>> bfs_queue;
    absl::flat_hash_set<const HloInstruction*> valid_set;

    bfs_queue.push({root_inst, 0});
    valid_set.insert(root_inst);

    // 第三步：执行严格管控的 BFS
    while (!bfs_queue.empty()) {
      auto [curr_inst, depth] = bfs_queue.front();
      bfs_queue.pop();

      // 如果当前指令的深度已经达到配置值，停止向其 operands 扩展
      if (depth >= opts.depth) {
        continue;
      }

      for (const HloInstruction* operand : curr_inst->operands()) {
        if (valid_set.contains(operand)) {
          continue;
        }

        // 拦截逻辑：直接拒绝不需要的控制流或非白名单 Call
        if (operand->opcode() == HloOpcode::kWhile ||
            operand->opcode() == HloOpcode::kConditional ||
            operand->opcode() == HloOpcode::kCall) {
          continue; 
        }

        // 如果指令总数已满，停止吸收新节点
        if (valid_set.size() >= opts.max_inst_count) {
          break; 
        }

        valid_set.insert(operand);
        bfs_queue.push({operand, depth + 1});
      }
      
      if (valid_set.size() >= opts.max_inst_count) {
        break;
      }
    }

    // 第四步：构造自定义的 Extract 拦截器并提取 Module
    auto extract_selector = [&](const HloInstruction* inst) -> bool {
      if (inst->opcode() == HloOpcode::kConstant) {
        return true;
      }
      if (inst->parent() == entry_comp) {
        return valid_set.contains(inst);
      }
      return whitelisted_comps.contains(inst->parent());
    };

    auto replace_type_selector = [](const HloInstruction* inst) {
      return ReplaceType::kReplaceParam; 
    };

    auto extracted_module = ExtractModule(
        /*instruction=*/root_inst, 
        /*height=*/-1, 
        /*extract_selector=*/extract_selector, 
        /*replace_type_selector=*/replace_type_selector, 
        /*cross_computation=*/true);

    int useful_op_count = 0;
    for (const HloInstruction* inst : extracted_module->entry_computation()->instructions()) {
      if (inst->opcode() != HloOpcode::kParameter && inst->opcode() != HloOpcode::kConstant) {
        useful_op_count++;
      }
    }
    
    // 如果没有实质性的计算逻辑，直接丢弃该切片
    if (useful_op_count <= 1) {
      continue;
    }

    // ==========================================
    // 利用 HloGumgraph 计算 Fingerprint 并进行去重
    // ==========================================

    // Clone the extracted module for fingerprinting only; the original is
    // preserved for output with its actual (non-normalized) slice indices.
    auto fp_module = extracted_module->Clone("fp");

    // Normalize kSlice start indices to 0 in the entry computation.
    // This makes slices that differ only in their starting offset
    // (e.g., [0:1] vs [1:2]) hash to the same fingerprint.
    // Whitelisted sub-computations are guaranteed to contain no kSlice.
    for (HloInstruction* inst :
         fp_module->entry_computation()->instructions()) {
      if (inst->opcode() == HloOpcode::kSlice) {
        auto* slice = Cast<HloSliceInstruction>(inst);
        auto* starts = slice->mutable_slice_starts();
        auto* limits = slice->mutable_slice_limits();
        for (int i = 0; i < static_cast<int>(starts->size()); ++i) {
          (*limits)[i] -= (*starts)[i];
          (*starts)[i] = 0;
        }
      }
    }

    // Normalize constant values to zero across all computations.
    // Programs differing only in constant values (e.g., constant(0.340488)
    // vs constant(-0.494451523)) will hash to the same fingerprint.
    for (HloComputation* comp : fp_module->computations()) {
      for (HloInstruction* inst : comp->instructions()) {
        if (inst->opcode() == HloOpcode::kConstant) {
          *Cast<HloConstantInstruction>(inst)->mutable_literal() =
              Literal::CreateFromShape(inst->shape());
        }
      }
    }

    xla::hlo_diff::HloGumgraphFingerprintOptions fp_options;
    fp_options.ignore_shape = true;
    fp_options.ignore_backend_config = true;

    auto graph_or_status = xla::hlo_diff::HloGumgraph::Create(
        fp_module.get(), fp_options, /*precompute_instruction_dependencies=*/false);

    if (!graph_or_status.ok()) {
      std::cerr << "Warning: Failed to create Gumgraph for slice rooted at "
                << root_inst->name() << ". Skipping deduplication and saving directly. Error: "
                << graph_or_status.status() << std::endl;
    } else {
      std::unique_ptr<const xla::hlo_diff::HloGumgraph> graph = *std::move(graph_or_status);
      uint64_t fingerprint = graph->GetRoot().props.subgraph_fingerprint;
      int64_t param_elements = ComputeParameterElements(extracted_module.get());

      auto it = best_slices.find(fingerprint);
      if (it == best_slices.end()) {
        fingerprint_order.push_back(fingerprint);
        best_slices.emplace(fingerprint,
            std::make_pair(std::move(extracted_module), param_elements));
      } else if (param_elements > it->second.second) {
        it->second.first = std::move(extracted_module);
        it->second.second = param_elements;
      }
    }
  }

  // 第五步：将每个指纹对应的最优切片序列化并输出到独立文件中
  for (uint64_t fp : fingerprint_order) {
    auto& [mod, param_elem_count] = best_slices[fp];
    std::string output_filename =
        absl::StrCat(opts.name, "_slice_", slice_index, ".hlo");
    if (!opts.output_dir.empty()) {
      tsl::Env::Default()->RecursivelyCreateDir(opts.output_dir);
      output_filename = tsl::io::JoinPath(opts.output_dir, output_filename);
    }
    std::ofstream out_file(output_filename);
    if (out_file.is_open()) {
      xla::HloPrintOptions print_options;
      print_options.set_print_metadata(false);
      out_file << mod->ToString(print_options);
      out_file.close();
      std::cout << "Successfully saved slice to: " << output_filename
                << " (ParameterElements: " << param_elem_count << ")" << std::endl;
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
                "  stablehlo : StableHLO in textual or bytecode format"),
      tsl::Flag("depth", &opts.depth, "The maximum depth for BFS traversal when slicing."),
      tsl::Flag("max_inst_count", &opts.max_inst_count, "The maximum number of instructions to include in each slice."),
      tsl::Flag("output_dir", &opts.output_dir, "The directory to save the sliced modules. Defaults to current directory."),
      tsl::Flag("name", &opts.name, "A name prefix for the output files. Defaults to 'default'.")};
      
  const std::string kUsageString =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[ 0 ], flag_list));

  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  if (!parse_ok) {
    std::cerr << kUsageString;
    return 1;
  }
  tsl::port::InitMain(kUsageString.c_str(), &argc, &argv);

  opts.input_file = argv[ 1 ];

  absl::Status status = xla::RunXlaSlicer(opts);
  if (!status.ok()) {
    std::cerr << status << std::endl;
    return 1;
  }
  return 0;
}