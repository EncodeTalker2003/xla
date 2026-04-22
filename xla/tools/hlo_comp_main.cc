#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/error_spec.h"
#include "xla/literal.h"
#include "xla/literal_comparison.h"
#include "xla/service/compiler.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/platform_util.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/test_utils.h"
#include "xla/tools/hlo_comp.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/path.h"

namespace {
const char* const kUsage = R"(
HLO Equivalence Checker
Usage:
  bazel run //xla/tools:hlo_comp -- \
    --input_format=[hlo|mhlo|pb|pbtxt|stablehlo] \
    --lhs_file=path/to/lhs_module \
    --rhs_file=path/to/rhs_module \
    --platform=[CPU|CUDA|Interpreter] \
    --iterations=10
)";
}  // namespace

namespace xla {
namespace {

// 简化版的比对失败回调，直接打印维度和概要信息
void OnMiscompare(const LiteralSlice& expected, const LiteralSlice& actual,
                  const LiteralSlice& mismatches,
                  const ShapeIndex& /*shape_index*/,
                  const literal_comparison::ErrorBuckets& /*error_buckets*/) {
  std::cerr << "\n>>> Mismatch Detected <<<\n"
            << "LHS (Expected): " << ShapeUtil::HumanString(expected.shape()) << "\n"
            << "RHS (Actual):   " << ShapeUtil::HumanString(actual.shape()) << "\n";
}

absl::Status RunHloComp(const HloCompConfig& opts) {
  std::string format = opts.input_format;
  if (format.empty()) {
    format = std::string(tsl::io::Extension(opts.lhs_file));
  }
  
  std::cerr << "Loading LHS module...\n";
  TF_ASSIGN_OR_RETURN(auto lhs_module, LoadModuleFromFile(opts.lhs_file, format));
  HloVerifier verifier(
      HloVerifierOpts{}.WithLayoutSensitive(false).WithAllowMixedPrecision(true));
  TF_RETURN_IF_ERROR(verifier.Run(lhs_module.get()).status());

  std::cerr << "Loading RHS module...\n";
  TF_ASSIGN_OR_RETURN(auto rhs_module, LoadModuleFromFile(opts.rhs_file, format));
  TF_RETURN_IF_ERROR(verifier.Run(rhs_module.get()).status());

  // 1. 签名一致性静态检查
  const auto* lhs_entry = lhs_module->entry_computation();
  const auto* rhs_entry = rhs_module->entry_computation();
  if (lhs_entry->num_parameters() != rhs_entry->num_parameters()) {
    return absl::InvalidArgumentError("LHS and RHS entry computations have a different number of parameters.");
  }
  for (int i = 0; i < lhs_entry->num_parameters(); ++i) {
    if (!ShapeUtil::Equal(lhs_entry->parameter_instruction(i)->shape(),
                          rhs_entry->parameter_instruction(i)->shape())) {
      return absl::InvalidArgumentError("LHS and RHS entry computation parameter shapes do not match.");
    }
  }

  // 2. 深拷贝以保留签名信息用于随机数据生成
  std::unique_ptr<HloModule> signature_module = lhs_module->Clone();

  // 3. 平台与 Runner 初始化
  TF_ASSIGN_OR_RETURN(se::Platform* platform, PlatformUtil::GetPlatform(opts.platform));
  HloRunner runner(platform);

  // --- 插入阶段：Cost Model 分析与耗时预估 ---
  TF_ASSIGN_OR_RETURN(auto compiler, xla::Compiler::GetForPlatform(platform));
  TF_ASSIGN_OR_RETURN(se::StreamExecutor* executor, platform->ExecutorForDevice(0));

  // 定义一个 Lambda 函数来复用 Cost Analysis 的逻辑
  auto run_cost_analysis = [&](const HloModule* module, const std::string& name) {
    std::cerr << "Running HLO Passes for " << name << " Cost Analysis...\n";
    std::unique_ptr<HloModule> cost_module = module->Clone();
    xla::Compiler::CompileOptions compile_options;
    auto pass_status = compiler->RunHloPasses(std::move(cost_module), executor, compile_options);

    if (pass_status.ok()) {
      std::unique_ptr<HloModule> optimized_module = std::move(pass_status).value();
      // 使用标准的紧凑内存计算函数初始化 Cost Analysis
      HloCostAnalysis cost_analysis([](const xla::Shape& shape) { 
        return xla::ShapeUtil::ByteSizeOf(shape); 
      });
      if (optimized_module->entry_computation()->Accept(&cost_analysis).ok()) {
        double total_flops = cost_analysis.flop_count();
        double total_bytes = cost_analysis.bytes_accessed();

        std::cerr << "[" << name << "] Total FLOPs: " << total_flops << "\n"
                  << "[" << name << "] Total Bytes Accessed: " << total_bytes << "\n";

        // 提取底层硬件信息
        const se::DeviceDescription& dev_desc = executor->GetDeviceDescription();
        int64_t memory_bandwidth = dev_desc.memory_bandwidth();
        double clock_rate_ghz = dev_desc.clock_rate_ghz();
        int core_count = dev_desc.core_count();

        // 如果无法提取有效硬件参数则打印警告并跳过预估
        if (memory_bandwidth <= 0 || clock_rate_ghz <= 0 || core_count <= 0) {
          std::cerr << "Warning: Unable to fetch hardware peak parameters for Roofline model. Skipping estimated time calculation for " << name << ".\n";
        } else {
          // 假设现代 GPU 架构每个核心每个时钟周期可执行约 128 次浮点运算
          double peak_flops_per_sec = clock_rate_ghz * 1e9 * core_count * 128.0;
          double peak_bandwidth_bytes_per_sec = static_cast<double>(memory_bandwidth);

          double time_compute_s = total_flops / peak_flops_per_sec;
          double time_memory_s = total_bytes / peak_bandwidth_bytes_per_sec;

          // 根据 Roofline 模型，执行时间由算力或访存的短板决定
          double estimated_time_ms = std::max(time_compute_s, time_memory_s) * 1000.0;
          std::cerr << "[" << name << "] Estimated execution time: " << estimated_time_ms << "ms\n";
        }
      }
    } else {
      std::cerr << "Warning: Failed to run HLO passes for " << name << " cost analysis.\n";
    }
  };

  // 分别对 LHS 和 RHS 执行耗时评估
  run_cost_analysis(lhs_module.get(), "LHS");
  std::cerr << "-\n";
  run_cost_analysis(rhs_module.get(), "RHS");
  std::cerr << "-------------------------------------------\n";

  // 4. AOT 提前编译阶段 (消耗原始 Module)
  std::cerr << "Compiling LHS Executable...\n";
  auto compile_start = std::chrono::high_resolution_clock::now();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<OpaqueExecutable> lhs_exec,
                      runner.CreateExecutable(std::move(lhs_module), /*run_hlo_passes=*/true));
  auto compile_mid = std::chrono::high_resolution_clock::now();
  std::cerr << "LHS compiled in " 
            << std::chrono::duration<double>(compile_mid - compile_start).count() << "s.\n";

  std::cerr << "Compiling RHS Executable...\n";
  TF_ASSIGN_OR_RETURN(std::unique_ptr<OpaqueExecutable> rhs_exec,
                      runner.CreateExecutable(std::move(rhs_module), /*run_hlo_passes=*/true));
  auto compile_end = std::chrono::high_resolution_clock::now();
  std::cerr << "RHS compiled in " 
            << std::chrono::duration<double>(compile_end - compile_mid).count() << "s.\n";

  // 5. 随机测试主循环
  std::minstd_rand0 engine; 
  ErrorSpec error_spec(1e-3, 1e-3); // 默认误差阈值设置
  double total_lhs_time = 0.0;
  double total_rhs_time = 0.0;

  std::cerr << "\nStarting fuzzing loop for " << opts.iterations << " iterations...\n";
  for (int i = 0; i < opts.iterations + 2; ++i) {
    ExecutionProfile lhs_profile, rhs_profile;
    // 根据拷贝的 Module 生成随机张量
    TF_ASSIGN_OR_RETURN(std::vector<Literal> args,
                        MakeFakeArguments(signature_module.get(), &engine, 
                                          /*use_large_range=*/false, 
                                          /*treat_gte_as_data_formatting=*/false));

    std::vector<const Literal*> arg_ptrs;
    arg_ptrs.reserve(args.size());
    for (const auto& arg : args) {
      arg_ptrs.push_back(&arg);
    }

    // 执行并获取结果
    TF_ASSIGN_OR_RETURN(Literal lhs_result, runner.ExecuteWithExecutableAndProfile(lhs_exec.get(), arg_ptrs, &lhs_profile));
    TF_ASSIGN_OR_RETURN(Literal rhs_result, runner.ExecuteWithExecutableAndProfile(rhs_exec.get(), arg_ptrs, &rhs_profile));

    // 对比容差
    absl::Status comparison_status = literal_comparison::Near(
        lhs_result, rhs_result, error_spec, /*detailed_message=*/true, &OnMiscompare);

    if (i < 2) {
      std::cerr << "Warm-up iteration " << i + 1 << " completed.\n";
      continue; // 前两轮作为 warm-up，不计入时间统计和最终对比结果
    }

    double lhs_time = static_cast<double>(lhs_profile.compute_time_ns()) / 1e6;
    double rhs_time = static_cast<double>(rhs_profile.compute_time_ns()) / 1e6;
    total_lhs_time += lhs_time;
    total_rhs_time += rhs_time;
    
    std::cerr << "Iteration " << i + 1 << ": LHS execution time = " 
              << lhs_time << "ms, "
              << "RHS execution time = " 
              << rhs_time << "ms.\n";

    if (!comparison_status.ok()) {
      std::cerr << "Mismatch detected at iteration " << i + 1 << "!\n";
      return comparison_status;
    }
  }

  std::cerr << "\nSuccess! LHS and RHS are equivalent across " << opts.iterations << " random inputs.\n";
  std::cerr << "Average LHS execution time: " << total_lhs_time / opts.iterations << "ms\n";
  std::cerr << "Average RHS execution time: " << total_rhs_time / opts.iterations << "ms\n";
  return absl::OkStatus();
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  xla::HloCompConfig opts;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("input_format", &opts.input_format, "The format of the input file."),
      tsl::Flag("lhs_file", &opts.lhs_file, "Path to the left-hand-side HLO module."),
      tsl::Flag("rhs_file", &opts.rhs_file, "Path to the right-hand-side HLO module."),
      tsl::Flag("platform", &opts.platform, "The test platform (gpu, cpu, etc)."),
      tsl::Flag("iterations", &opts.iterations, "The number of times to run the module.")};
        
  const std::string kUsageString =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[ 0 ], flag_list));

  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  if (!parse_ok) {
    std::cerr << kUsageString;
    return 1;
  }
  tsl::port::InitMain(kUsageString.c_str(), &argc, &argv);

  absl::Status status = xla::RunHloComp(opts);

  if (!status.ok()) {
    std::cerr << status << std::endl;
    return 1;
  }
  return 0;
}