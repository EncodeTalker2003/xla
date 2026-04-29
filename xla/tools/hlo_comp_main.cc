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
#include "xla/service/hlo_runner.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/platform_util.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/test_utils.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
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

  // MLIRContext must outlive all GpuPerformanceModelOwning instances.
  mlir::MLIRContext mlir_ctx;

  // 定义一个 Lambda 函数来复用 Cost Analysis 的逻辑
  auto run_cost_analysis = [&](const HloModule* module, const std::string& name) {
    std::cerr << "Running HLO Passes for " << name << " Cost Analysis...\n";
    std::unique_ptr<HloModule> cost_module = module->Clone();
    cost_module->mutable_config().mutable_debug_options().add_xla_disable_hlo_passes("algsimp");
    xla::Compiler::CompileOptions compile_options;
    auto pass_status = compiler->RunHloPasses(std::move(cost_module), executor, compile_options);

    if (!pass_status.ok()) {
      std::cerr << "Warning: Failed to run HLO passes for " << name << " cost analysis.\n";
      return;
    }

    std::unique_ptr<HloModule> optimized_module = std::move(pass_status).value();
    const se::DeviceDescription& dev = executor->GetDeviceDescription();

    // Run GpuHloCostAnalysis to populate per-instruction utilization data.
    gpu::GpuHloCostAnalysis::Options ca_opts{
        [](const xla::Shape& shape) { return xla::ShapeUtil::ByteSizeOf(shape, 8); }};
    gpu::GpuHloCostAnalysis cost_analysis(ca_opts, dev);
    if (!optimized_module->entry_computation()->Accept(&cost_analysis).ok()) {
      std::cerr << "Warning: GpuHloCostAnalysis failed for " << name << ".\n";
      return;
    }

    // Use GpuPerformanceModel to estimate per-instruction runtime and sum up.
    gpu::GpuPerformanceModelOwning gpu_model(dev, &mlir_ctx);
    int64_t total_flops = 0, total_bytes_read = 0, total_bytes_written = 0;
    absl::Duration total_compute_time, total_memory_time, total_exec_time;
    int64_t num_kernels = 0;
    for (const HloInstruction* instr :
         optimized_module->entry_computation()->instructions()) {
      gpu::EstimateRunTimeData rt =
          gpu_model.Get().EstimateRunTimeForInstruction(instr, &cost_analysis);
      total_flops         += rt.flops;
      total_bytes_read    += rt.bytes_read;
      total_bytes_written += rt.bytes_written;
      total_compute_time  += rt.compute_time;
      total_memory_time   += rt.read_time + rt.write_time;
      total_exec_time     += rt.exec_time;
      if (instr->opcode() == HloOpcode::kFusion) ++num_kernels;
    }
    // Add 1µs kernel launch overhead per GPU kernel (kFusion instruction).
    total_exec_time += gpu::GpuPerformanceModelBase::kKernelLaunchOverhead * num_kernels;

    double compute_ms = absl::ToDoubleMilliseconds(total_compute_time);
    double memory_ms  = absl::ToDoubleMilliseconds(total_memory_time);
    double exec_ms    = absl::ToDoubleMilliseconds(total_exec_time);
    std::cerr << "[" << name << "] Total FLOPs: " << total_flops << "\n"
              << "[" << name << "] Bytes read: " << total_bytes_read << "\n"
              << "[" << name << "] Bytes written: " << total_bytes_written << "\n"
              << "[" << name << "] Compute time: " << compute_ms << "ms\n"
              << "[" << name << "] Memory time: " << memory_ms << "ms\n"
              << "[" << name << "] Estimated exec time: " << exec_ms
              << "ms (GpuPerformanceModel)\n";
    if (compute_ms > memory_ms) {
      std::cerr << ">>> Compute-bound <<<\n";
    } else {
      std::cerr << ">>> Memory-bound <<<\n";
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
  lhs_module->mutable_config().mutable_debug_options().add_xla_disable_hlo_passes("algsimp");
  TF_ASSIGN_OR_RETURN(std::unique_ptr<OpaqueExecutable> lhs_exec,
                      runner.CreateExecutable(std::move(lhs_module), /*run_hlo_passes=*/true));
  auto compile_mid = std::chrono::high_resolution_clock::now();
  std::cerr << "LHS compiled in " 
            << std::chrono::duration<double>(compile_mid - compile_start).count() << "s.\n";

  std::cerr << "Compiling RHS Executable...\n";
  rhs_module->mutable_config().mutable_debug_options().add_xla_disable_hlo_passes("algsimp");
  TF_ASSIGN_OR_RETURN(std::unique_ptr<OpaqueExecutable> rhs_exec,
                      runner.CreateExecutable(std::move(rhs_module), /*run_hlo_passes=*/true));
  auto compile_end = std::chrono::high_resolution_clock::now();
  std::cerr << "RHS compiled in " 
            << std::chrono::duration<double>(compile_end - compile_mid).count() << "s.\n";

  // 5. Generate all N+2 input sets upfront so LHS and RHS see identical inputs.
  std::minstd_rand0 engine;
  ErrorSpec error_spec(1e-3, 1e-3);
  const int total_iters = opts.iterations + 2;  // 2 warm-up + N measured

  std::cerr << "\nGenerating " << total_iters << " input sets...\n";
  std::vector<std::vector<Literal>> all_args(total_iters);
  for (int i = 0; i < total_iters; ++i) {
    TF_ASSIGN_OR_RETURN(all_args[i],
                        MakeFakeArguments(signature_module.get(), &engine,
                                          /*use_large_range=*/false,
                                          /*treat_gte_as_data_formatting=*/false));
  }

  // Helper: build a const-pointer span over one input set.
  auto make_ptrs = [](const std::vector<Literal>& args) {
    std::vector<const Literal*> ptrs;
    ptrs.reserve(args.size());
    for (const auto& a : args) ptrs.push_back(&a);
    return ptrs;
  };

  // 6. LHS block: warm-up then N measured runs.
  std::cerr << "\nRunning LHS block...\n";
  std::vector<Literal> lhs_results(opts.iterations);
  std::vector<double>  lhs_times(opts.iterations);
  for (int i = 0; i < total_iters; ++i) {
    ExecutionProfile profile;
    auto ptrs = make_ptrs(all_args[i]);
    TF_ASSIGN_OR_RETURN(Literal result,
                        runner.ExecuteWithExecutableAndProfile(lhs_exec.get(), ptrs, &profile));
    if (i < 2) {
      std::cerr << "LHS warm-up iteration " << i + 1 << " completed.\n";
      continue;
    }
    lhs_results[i - 2] = std::move(result);
    lhs_times[i - 2] = static_cast<double>(profile.compute_time_ns()) / 1e6;
		std::cerr << "LHS iteration " << i - 1 << " completed in " << lhs_times[i - 2] << "ms.\n";
  }

  // 7. RHS block: warm-up then N measured runs; compare on-the-fly with stored LHS results.
  std::cerr << "\nRunning RHS block...\n";
  std::vector<double> rhs_times(opts.iterations);
  for (int i = 0; i < total_iters; ++i) {
    ExecutionProfile profile;
    auto ptrs = make_ptrs(all_args[i]);
    TF_ASSIGN_OR_RETURN(Literal rhs_result,
                        runner.ExecuteWithExecutableAndProfile(rhs_exec.get(), ptrs, &profile));
    if (i < 2) {
      std::cerr << "RHS warm-up iteration " << i + 1 << " completed.\n";
      continue;
    }
    rhs_times[i - 2] = static_cast<double>(profile.compute_time_ns()) / 1e6;
    absl::Status cmp = literal_comparison::Near(
        lhs_results[i - 2], rhs_result, error_spec, /*detailed_message=*/true, &OnMiscompare);
    if (!cmp.ok()) {
      std::cerr << "Mismatch detected at iteration " << i - 1 << "!\n";
      return cmp;
    }
		std::cerr << "RHS iteration " << i - 1 << " completed in " << rhs_times[i - 2] << "ms.\n";
  }

  // 8. Report per-iteration times and summary.
  double total_lhs_time = 0.0, total_rhs_time = 0.0;
  for (int i = 0; i < opts.iterations; ++i) {
    total_lhs_time += lhs_times[i];
    total_rhs_time += rhs_times[i];
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