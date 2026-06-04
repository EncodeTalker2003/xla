#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <optional>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
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
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/autotuning.pb.h"
#include "xla/autotune_results.pb.h"
#include "xla/service/gpu/autotuning/autotune_cache_key.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/tools/hlo_comp.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/env.h"
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

// ── Conclusion types ──────────────────────────────────────────────────────────

enum class HloCompConclusion {
  kGrammarError,  // LHS or RHS failed to load / verify
  kRuntimeError,  // LHS or RHS failed to compile or execute
  kNotEquivalent, // outputs differ, or incompatible signatures
  kRhsNotBetter,  // equivalent but RHS is not more profitable
  kRhsBetter,     // equivalent and RHS is more profitable
};

// Prints the final conclusion to stdout.
// `side` should be "LHS" or "RHS" for kGrammarError and kRuntimeError.
// `error_detail` is appended to the conclusion line for those two kinds so
// callers can extract both the category and the raw error in one line.
void PrintConclusion(HloCompConclusion conclusion,
                     std::string_view side = "",
                     std::string_view error_detail = "") {
  auto append_detail = [&]() {
    if (!error_detail.empty()) std::cout << ": " << error_detail;
  };
  switch (conclusion) {
    case HloCompConclusion::kGrammarError:
      std::cout << "CONCLUSION: " << side << " has grammar issues" << std::endl;
      append_detail();
      std::cout << "\n";
      break;
    case HloCompConclusion::kRuntimeError:
      std::cout << "CONCLUSION: " << side << " has runtime errors" << std::endl;
      append_detail();
      std::cout << "\n";
      break;
    case HloCompConclusion::kNotEquivalent:
      std::cout << "CONCLUSION: Original HLO Program and HLO program proposed by LLM are not equivalent" << std::endl;
      append_detail();
      break;
    case HloCompConclusion::kRhsNotBetter:
      std::cout << "CONCLUSION: HLO program proposed by LLM is not more profitable than Original HLO Program" << std::endl;
      append_detail();
      break;
    case HloCompConclusion::kRhsBetter:
      std::cout << "CONCLUSION: HLO program proposed by LLM is more profitable than Original HLO Program" << std::endl;
      break;
  }
  std::cout.flush();
}

// ── Data types ────────────────────────────────────────────────────────────────

void OnMiscompare(const LiteralSlice& expected, const LiteralSlice& actual,
                  const LiteralSlice& mismatches,
                  const ShapeIndex& /*shape_index*/,
                  const literal_comparison::ErrorBuckets& /*error_buckets*/) {
  std::cerr << "\n>>> Mismatch Detected <<<\n"
            << "LHS (Expected): " << ShapeUtil::HumanString(expected.shape()) << "\n"
            << "RHS (Actual):   " << ShapeUtil::HumanString(actual.shape()) << "\n";
}

struct CostMetrics {
  int64_t kernel_count{0};
  int64_t total_flops{0};
  absl::Duration total_exec_time;
  int64_t instruction_count{0};
  int64_t total_hbm_bytes{0};
};

// ── Verdict determination ─────────────────────────────────────────────────────

// Determines whether RHS is more profitable than LHS using a priority-order
// comparison. When static metrics are unavailable, falls back to real exec time
// only (>5% threshold required for kRhsBetter).
HloCompConclusion DetermineVerdict(const std::optional<CostMetrics>& lhs_opt,
                                   const std::optional<CostMetrics>& rhs_opt,
                                   double avg_lhs_ms, double avg_rhs_ms,
                                   int64_t lhs_raw_inst_count,
                                   int64_t rhs_raw_inst_count) {
  constexpr double kThreshold = 0.05;

  if (lhs_opt && rhs_opt) {
    const CostMetrics& lm = *lhs_opt;
    const CostMetrics& rm = *rhs_opt;
    double lhs_static_ms = absl::ToDoubleMilliseconds(lm.total_exec_time);
    double rhs_static_ms = absl::ToDoubleMilliseconds(rm.total_exec_time);
    double lhbm = static_cast<double>(lm.total_hbm_bytes);
    double rhbm = static_cast<double>(rm.total_hbm_bytes);

    

		if (rhs_static_ms < lhs_static_ms * 0.98 || rhs_static_ms < lhs_static_ms - 0.01) {
			return HloCompConclusion::kRhsBetter;
		} else if (lhs_static_ms < rhs_static_ms * 0.98 || lhs_static_ms < rhs_static_ms - 0.01) {
			return HloCompConclusion::kRhsNotBetter;
		} else if (rhs_raw_inst_count < lhs_raw_inst_count * 0.7) {
			return HloCompConclusion::kRhsBetter;
		} else {
			return HloCompConclusion::kRhsNotBetter;
		}
	}
    
		/*
		// Priority order: FLOPs → HBM → kernel count → real time → static time →
    // instruction count.
    if (lm.total_flops != rm.total_flops)
      return (rm.total_flops < lm.total_flops) ? HloCompConclusion::kRhsBetter
                                               : HloCompConclusion::kRhsNotBetter;
    if (rhbm < lhbm * (1.0 - kThreshold))
      return HloCompConclusion::kRhsBetter;
    if (lhbm < rhbm * (1.0 - kThreshold))
      return HloCompConclusion::kRhsNotBetter;
    if (lm.kernel_count != rm.kernel_count)
      return (rm.kernel_count < lm.kernel_count) ? HloCompConclusion::kRhsBetter
                                                 : HloCompConclusion::kRhsNotBetter;
    if (avg_rhs_ms < avg_lhs_ms * (1.0 - kThreshold))
      return HloCompConclusion::kRhsBetter;
    if (avg_lhs_ms < avg_rhs_ms * (1.0 - kThreshold))
      return HloCompConclusion::kRhsNotBetter;
    if (rhs_static_ms < lhs_static_ms * (1.0 - kThreshold))
      return HloCompConclusion::kRhsBetter;
    if (lhs_static_ms < rhs_static_ms * (1.0 - kThreshold))
      return HloCompConclusion::kRhsNotBetter;
    if (lm.instruction_count != rm.instruction_count)
      return (rm.instruction_count < lm.instruction_count)
                 ? HloCompConclusion::kRhsBetter
                 : HloCompConclusion::kRhsNotBetter;
    return HloCompConclusion::kRhsNotBetter;  // all metrics tied
  }
		*/

  // Static metrics unavailable — fall back to real exec time only.
  if (avg_rhs_ms < avg_lhs_ms * 0.98 || avg_rhs_ms < avg_lhs_ms - 0.01)
    return HloCompConclusion::kRhsBetter;
  return HloCompConclusion::kRhsNotBetter;
}

// ── Comparison table (pure printer, no verdict) ───────────────────────────────

void PrintComparisonTable(const std::optional<CostMetrics>& lhs_opt,
                          const std::optional<CostMetrics>& rhs_opt,
                          double avg_lhs_ms, double avg_rhs_ms,
                          int64_t lhs_raw_inst_count,
                          int64_t rhs_raw_inst_count) {
  // Mirrors the threshold in DetermineVerdict exactly.
  auto time_winner = [](double lv, double rv) -> const char* {
    if (rv < lv * 0.98 || rv < lv - 0.01) return "LLM";
    if (lv < rv * 0.98 || lv < rv - 0.01) return "Original";
    return "tie";
  };
  auto abs_winner = [](auto lv, auto rv) -> const char* {
    if (lv < rv) return "Original";
    if (rv < lv) return "LLM";
    return "tie";
  };
	auto inst_winner = [](auto lv, auto rv) -> const char* {
    if (lv < rv * 0.7) return "Original";
    if (rv < lv * 0.7) return "LLM";
    return "tie";
  };

  std::cout << "\n=== Comparison Table ===\n"
            << std::left
            << std::setw(24) << "Metric"
            << std::setw(16) << "Original"
            << std::setw(16) << "LLM Proposed"
            << "Winner\n"
            << std::string(62, '-') << "\n";

  if (lhs_opt && rhs_opt) {
    const CostMetrics& lm = *lhs_opt;
    const CostMetrics& rm = *rhs_opt;
    double lhs_static_ms = absl::ToDoubleMilliseconds(lm.total_exec_time);
    double rhs_static_ms = absl::ToDoubleMilliseconds(rm.total_exec_time);
    double lhbm_mb = lm.total_hbm_bytes / 1.0e6;
    double rhbm_mb = rm.total_hbm_bytes / 1.0e6;

    // Decision metrics in priority order (match DetermineVerdict).
    std::cout << std::fixed << std::setprecision(6)
              << std::setw(24) << "Static exec time (ms)"
              << std::setw(16) << lhs_static_ms
              << std::setw(16) << rhs_static_ms
              << time_winner(lhs_static_ms, rhs_static_ms) << "\n"
              << std::defaultfloat
              << std::setw(24) << "Raw inst count"
              << std::setw(16) << lhs_raw_inst_count
              << std::setw(16) << rhs_raw_inst_count
              << inst_winner(lhs_raw_inst_count, rhs_raw_inst_count) << "\n"
              << std::setw(24) << "Inst count (opt)"
              << std::setw(16) << lm.instruction_count
              << std::setw(16) << rm.instruction_count
              << "--\n";

    // Informational metrics (Winner not applicable — does not drive verdict).
    std::cout << std::setw(24) << "Total FLOPs"
              << std::setw(16) << lm.total_flops
              << std::setw(16) << rm.total_flops
              << "--\n"
              << std::fixed << std::setprecision(4)
              << std::setw(24) << "HBM traffic (MB)"
              << std::setw(16) << lhbm_mb
              << std::setw(16) << rhbm_mb
              << "--\n"
              << std::defaultfloat
              << std::setw(24) << "Kernel count"
              << std::setw(16) << lm.kernel_count
              << std::setw(16) << rm.kernel_count
              << "--\n"
              << std::fixed << std::setprecision(6)
              << std::setw(24) << "Real exec time (ms)"
              << std::setw(16) << avg_lhs_ms
              << std::setw(16) << avg_rhs_ms
              << "--\n"
              << std::defaultfloat;
  } else {
    // Static metrics unavailable — real exec time is the decision metric.
    std::cout << std::fixed << std::setprecision(6)
              << std::setw(24) << "Real exec time (ms)"
              << std::setw(16) << avg_lhs_ms
              << std::setw(16) << avg_rhs_ms
              << time_winner(avg_lhs_ms, avg_rhs_ms) << "\n"
              << std::defaultfloat;

    // Raw inst count is still available even when static metrics failed.
    std::cout << std::setw(24) << "Raw inst count"
              << std::setw(16) << lhs_raw_inst_count
              << std::setw(16) << rhs_raw_inst_count
              << abs_winner(lhs_raw_inst_count, rhs_raw_inst_count) << "\n";
    // All other static metrics are N/A — show as informational.
    std::cout << std::setw(24) << "Static exec time (ms)"
              << std::setw(16) << "N/A" << std::setw(16) << "N/A" << "--\n"
              << std::setw(24) << "Inst count (opt)"
              << std::setw(16) << "N/A" << std::setw(16) << "N/A" << "--\n"
              << std::setw(24) << "Total FLOPs"
              << std::setw(16) << "N/A" << std::setw(16) << "N/A" << "--\n"
              << std::setw(24) << "HBM traffic (MB)"
              << std::setw(16) << "N/A" << std::setw(16) << "N/A" << "--\n"
              << std::setw(24) << "Kernel count"
              << std::setw(16) << "N/A" << std::setw(16) << "N/A" << "--\n";
  }
}

// ── Main comparison logic ─────────────────────────────────────────────────────

absl::Status RunHloComp(const HloCompConfig& opts) {
  std::string format = opts.input_format;
  if (format.empty()) {
    format = std::string(tsl::io::Extension(opts.lhs_file));
  }

  // ── Phase 1: Load and verify LHS ─────────────────────────────────────────
  std::cerr << "Loading LHS module...\n";
  auto lhs_module_or = LoadModuleFromFile(opts.lhs_file, format);
  if (!lhs_module_or.ok()) {
    std::cerr << "Grammar error (LHS): " << lhs_module_or.status().message() << "\n";
    PrintConclusion(HloCompConclusion::kGrammarError, "Original HLO Program",
                    lhs_module_or.status().message());
    return lhs_module_or.status();
  }
  std::unique_ptr<HloModule> lhs_module = std::move(lhs_module_or).value();

  HloVerifier verifier(
      HloVerifierOpts{}.WithLayoutSensitive(false).WithAllowMixedPrecision(true));
  auto lhs_verify = verifier.Run(lhs_module.get());
  if (!lhs_verify.ok()) {
    std::cerr << "Grammar error (LHS): " << lhs_verify.status().message() << "\n";
    PrintConclusion(HloCompConclusion::kGrammarError, "Original HLO Program",
                    lhs_verify.status().message());
    return lhs_verify.status();
  }
  int64_t lhs_raw_inst_count =
      lhs_module->entry_computation()->instruction_count();
  std::cerr << "[LHS] Raw instruction count: " << lhs_raw_inst_count << "\n";

  // ── Phase 1b: Load and verify RHS ────────────────────────────────────────
  std::cerr << "Loading RHS module...\n";
  auto rhs_module_or = LoadModuleFromFile(opts.rhs_file, format);
  if (!rhs_module_or.ok()) {
    std::cerr << "Grammar error (RHS): " << rhs_module_or.status().message() << "\n";
    PrintConclusion(HloCompConclusion::kGrammarError, "HLO program proposed by LLM",
                    rhs_module_or.status().message());
    return rhs_module_or.status();
  }
  std::unique_ptr<HloModule> rhs_module = std::move(rhs_module_or).value();

  auto rhs_verify = verifier.Run(rhs_module.get());
  if (!rhs_verify.ok()) {
    std::cerr << "Grammar error (RHS): " << rhs_verify.status().message() << "\n";
    PrintConclusion(HloCompConclusion::kGrammarError, "HLO program proposed by LLM",
                    rhs_verify.status().message());
    return rhs_verify.status();
  }
  int64_t rhs_raw_inst_count =
      rhs_module->entry_computation()->instruction_count();
  std::cerr << "[RHS] Raw instruction count: " << rhs_raw_inst_count << "\n";

  // ── Phase 2: Signature check ──────────────────────────────────────────────
  const auto* lhs_entry = lhs_module->entry_computation();
  const auto* rhs_entry = rhs_module->entry_computation();
  if (lhs_entry->num_parameters() != rhs_entry->num_parameters()) {
    PrintConclusion(HloCompConclusion::kNotEquivalent, "", "Parameter count mismatch");
    return absl::InvalidArgumentError(
        "LHS and RHS entry computations have a different number of parameters.");
  }
  for (int i = 0; i < lhs_entry->num_parameters(); ++i) {
    if (!ShapeUtil::Equal(lhs_entry->parameter_instruction(i)->shape(),
                          rhs_entry->parameter_instruction(i)->shape())) {
      PrintConclusion(HloCompConclusion::kNotEquivalent, "", "Parameter shape mismatch");
      return absl::InvalidArgumentError(
          "LHS and RHS entry computation parameter shapes do not match.");
    }
  }

  // Preserve signature for random-input generation (phases 5–6 consume the
  // original modules).
  std::unique_ptr<HloModule> signature_module = lhs_module->Clone();

  // ── Phase 3: Platform and runner init ────────────────────────────────────
  TF_ASSIGN_OR_RETURN(se::Platform* platform,
                      PlatformUtil::GetPlatform(opts.platform));
  HloRunner runner(platform);
  TF_ASSIGN_OR_RETURN(auto compiler, xla::Compiler::GetForPlatform(platform));
  TF_ASSIGN_OR_RETURN(se::StreamExecutor* executor,
                      platform->ExecutorForDevice(0));

  // MLIRContext must outlive all GpuPerformanceModelOwning instances.
  mlir::MLIRContext mlir_ctx;

  // ── Phase 3: Cost analysis (soft failure — nullopt continues) ────────────
  auto run_cost_analysis = [&](const HloModule* module,
                               const std::string& name)
      -> std::optional<CostMetrics> {
    std::cerr << "Running HLO Passes for " << name << " Cost Analysis...\n";
    std::unique_ptr<HloModule> cost_module = module->Clone();
    cost_module->mutable_config()
        .mutable_debug_options()
        .add_xla_disable_hlo_passes("algsimp");
    xla::Compiler::CompileOptions compile_options;
    auto pass_status = compiler->RunHloPasses(std::move(cost_module), executor,
                                              compile_options);
    if (!pass_status.ok()) {
      std::cerr << "Warning: Failed to run HLO passes for " << name
                << " cost analysis.\n";
      return std::nullopt;
    }
    std::unique_ptr<HloModule> optimized_module = std::move(pass_status).value();

    if (!opts.dump_dir.empty()) {
      std::string path =
          tsl::io::JoinPath(opts.dump_dir, name + "_optimized.hlo");
      absl::Status s = tsl::WriteStringToFile(tsl::Env::Default(), path,
                                              optimized_module->ToString());
      if (s.ok())
        std::cerr << "Dumped optimized " << name << " HLO to " << path << "\n";
      else
        std::cerr << "Warning: failed to dump " << name << " HLO: " << s << "\n";
    }

    const se::DeviceDescription& dev = executor->GetDeviceDescription();
    gpu::GpuHloCostAnalysis::Options ca_opts{
        [](const xla::Shape& shape) {
          return xla::ShapeUtil::ByteSizeOf(shape, 8);
        }};
    gpu::GpuHloCostAnalysis cost_analysis(ca_opts, dev);
    if (!optimized_module->entry_computation()->Accept(&cost_analysis).ok()) {
      std::cerr << "Warning: GpuHloCostAnalysis failed for " << name << ".\n";
      return std::nullopt;
    }

    // Harvest the measured run_times stored in the process-global autotune cache
    // (populated during RunHloPasses for cuDNN conv, cuBLAS GEMM, Triton GEMM).
    // The autotuner keys its results on the PRE-optimization instruction (before
    // the chosen algorithm / scratch buffer is baked into a conv, and before a
    // Triton gemm fusion is re-nested), so an AutotuneCacheKey lookup on the
    // optimized instruction misses. Instead we read all stored results and bucket
    // them by autotuner result type, then match them to optimized kernels by type
    // (and output shape when more than one candidate exists) below.
    struct MeasuredKernel {
      std::string hlo;
      absl::Duration run_time;
    };
    std::vector<MeasuredKernel> conv_times, triton_times, cublas_times;
    {
      AutotuneResults harvested;
      if (gpu::AutotunerUtil::SerializeAutotuneResults(&harvested).ok()) {
        for (const auto& e : harvested.results()) {
          const AutotuneResult& r = e.result();
          if (!r.has_run_time()) continue;               // no measurement stored
					std::cout << "Find sth interesting" << std::endl;
          absl::Duration t = absl::Seconds(r.run_time().seconds()) +
                             absl::Nanoseconds(r.run_time().nanos());
          if (r.has_algorithm() || r.has_cuda_conv_plan()) {
						std::cout << "Push convolution time" << std::endl;
            conv_times.push_back({e.hlo(), t});
          } else if (r.has_triton()) {
						std::cout << "Push triton time" << std::endl;
            triton_times.push_back({e.hlo(), t});
          } else if (r.has_gemm()) {
						std::cout << "Push gemm time" << std::endl;
            cublas_times.push_back({e.hlo(), t});
          }
        }
      }
    }

    // Returns the measured autotuned exec time for `instr` from `bucket`, or
    // nullopt. With a single candidate the match is unambiguous; otherwise we
    // disambiguate by checking the kernel's primary (non-scratch) output shape
    // appears in the cached entry's canonical HLO, and give up if still unclear.
    auto match_measured_time =
        [](const HloInstruction* instr,
           const std::vector<MeasuredKernel>& bucket)
        -> std::optional<absl::Duration> {
      if (bucket.empty()) return std::nullopt;
      if (bucket.size() == 1) return bucket.front().run_time;
      const Shape& primary = instr->shape().IsTuple()
                                 ? instr->shape().tuple_shapes(0)
                                 : instr->shape();
      std::string shape_str = primary.ToString(/*print_layout=*/true);
      const MeasuredKernel* hit = nullptr;
      for (const auto& k : bucket) {
        if (absl::StrContains(k.hlo, shape_str)) {
          if (hit != nullptr) return std::nullopt;  // ambiguous
          hit = &k;
        }
      }
      if (hit != nullptr) return hit->run_time;
      return std::nullopt;
    };

    gpu::GpuPerformanceModelOwning gpu_model(dev, &mlir_ctx);
    int64_t total_flops = 0, total_bytes_read = 0, total_bytes_written = 0;
    absl::Duration total_compute_time, total_memory_time, total_exec_time;
    int64_t num_kernels = 0, instruction_count = 0;
    for (const HloInstruction* instr :
         optimized_module->entry_computation()->instructions()) {
      ++instruction_count;
      // Only kFusion and kCustomCall dispatch actual GPU kernels. Skip all
      // other instructions (kParameter, kTuple, kGetTupleElement, kBitcast,
      // etc.) — they are zero-cost at runtime, and their memory would be
      // double-counted since the consuming kernels already account for reading
      // those bytes in their own bytes_read.
      const bool is_kernel = instr->opcode() == HloOpcode::kFusion ||
                             instr->opcode() == HloOpcode::kCustomCall;
      if (!is_kernel) continue;
      ++num_kernels;

      gpu::EstimateRunTimeData rt;
      // Reuse the autotuner's measured run_time for conv/gemm kernels. The
      // roofline model cannot distinguish kernels with identical FLOPs/bytes but
      // very different real speed (e.g. a cuDNN grouped conv vs a Triton GEMM),
      // so for those we substitute the measured time harvested above. Select the
      // matching bucket by kernel type.
      const std::vector<MeasuredKernel>* bucket = nullptr;
      const char* bucket_kind = nullptr;
      if (gpu::IsCustomCallToDnnConvolution(*instr)) {
        bucket = &conv_times;
        bucket_kind = "conv";
      } else if (gpu::IsCublasGemm(*instr)) {
        bucket = &cublas_times;
        bucket_kind = "cublas-gemm";
      } else if (instr->opcode() == HloOpcode::kFusion &&
                 instr->fusion_kind() == HloInstruction::FusionKind::kCustom) {
        auto cfg = instr->backend_config<gpu::GpuBackendConfig>();
        if (cfg.ok()) {
          absl::string_view k = cfg->fusion_backend_config().kind();
          if (k == gpu::kTritonGemmFusionKind ||
              k == gpu::kTritonNestedGemmFusionKind) {
            bucket = &triton_times;
            bucket_kind = "triton-gemm";
          }
        }
      }

      std::optional<absl::Duration> measured;
      if (bucket != nullptr) {
        measured = match_measured_time(instr, *bucket);
        if (!measured.has_value()) {
          std::cerr << "  [autotuner] WARNING: " << instr->name() << " ("
                    << bucket_kind
                    << ") had no matched measured time; using roofline\n";
        }
      }

      if (measured.has_value()) {
        int64_t flops     = cost_analysis.flop_count(*instr);
        int64_t bytes_out = cost_analysis.output_bytes_accessed(*instr);
        int64_t bytes_in  = cost_analysis.bytes_accessed(*instr) - bytes_out;
        absl::Duration read_time =
            absl::Seconds(1.0 * bytes_in / dev.memory_bandwidth());
        absl::Duration write_time =
            gpu::GpuPerformanceModelBase::WriteTime(dev, bytes_out);
        absl::Duration compute_time = gpu::GpuPerformanceModelBase::ComputeTime(
            dev, flops, dev.core_count(), dev.fpus_per_core());
        rt = {flops, bytes_in, bytes_out,
              read_time, write_time, compute_time, *measured};
        std::cerr << "  [autotuner] " << instr->name() << " (" << bucket_kind
                  << "): " << absl::ToDoubleMilliseconds(*measured)
                  << "ms (measured)\n";
      } else if (gpu::IsCustomCallToDnnConvolution(*instr) ||
          gpu::IsCublasGemm(*instr)) {
        // EstimateRunTimeForInstruction falls through to kLoop emitter for
        // kCustomCall, producing wrong launch dimensions. Use a full-occupancy
        // roofline instead: both cuDNN and cuBLAS saturate all SMs.
        int64_t flops     = cost_analysis.flop_count(*instr);
        int64_t bytes_out = cost_analysis.output_bytes_accessed(*instr);
        int64_t bytes_in  = cost_analysis.bytes_accessed(*instr) - bytes_out;
        const int64_t num_blocks        = dev.core_count();
        const int64_t threads_per_block = dev.fpus_per_core();
        absl::Duration compute_time = gpu::GpuPerformanceModelBase::ComputeTime(
            dev, flops, num_blocks, threads_per_block);
        absl::Duration write_time =
            gpu::GpuPerformanceModelBase::WriteTime(dev, bytes_out);
        absl::Duration read_time =
            absl::Seconds(1.0 * bytes_in / dev.memory_bandwidth());
        absl::Duration exec_time =
            gpu::GpuPerformanceModelBase::CombineComputeAndMemoryAccessTime(
                compute_time, read_time + write_time);
        rt = {flops, bytes_in, bytes_out,
              read_time, write_time, compute_time, exec_time};
      } else {
        rt = gpu_model.Get().EstimateRunTimeForInstruction(instr, &cost_analysis);

        // CoalescingAnalysis marks reads of kInput transpose/reduce kernels and
        // Triton "__triton" fusions as non-coalesced, applying a heavy penalty
        // (element_size / cache_line_size ≈ 1/16 for f32). However:
        //   - kInput fusions compile to tiled-transpose or warp-shuffle-reduce
        //     kernels that buffer tiles in shared memory, so global reads ARE
        //     effectively coalesced.
        //   - kCustom "__triton" (non-GEMM) fusions let Triton handle memory
        //     layout internally, also achieving coalesced global access.
        // Re-estimate read_time at peak (coalesced) bandwidth for these cases.
        bool needs_coalesced_read_override = false;
        if (instr->fusion_kind() == HloInstruction::FusionKind::kInput) {
          needs_coalesced_read_override = true;
        } else if (instr->fusion_kind() == HloInstruction::FusionKind::kCustom) {
          auto cfg = instr->backend_config<gpu::GpuBackendConfig>();
          if (cfg.ok() &&
              cfg->fusion_backend_config().kind() == gpu::kTritonFusionKind) {
            needs_coalesced_read_override = true;
          }
        }
        if (needs_coalesced_read_override) {
          absl::Duration coalesced_read_time = absl::Seconds(
              1.0 * rt.bytes_read / dev.memory_bandwidth());
          rt.read_time = coalesced_read_time;
          rt.exec_time = gpu::GpuPerformanceModelBase::CombineComputeAndMemoryAccessTime(
              rt.compute_time, coalesced_read_time + rt.write_time);
        }
      }
      total_flops         += rt.flops;
      total_bytes_read    += rt.bytes_read;
      total_bytes_written += rt.bytes_written;
      total_compute_time  += rt.compute_time;
      total_memory_time   += rt.read_time + rt.write_time;
      total_exec_time     += rt.exec_time;
    }
    total_exec_time +=
        gpu::GpuPerformanceModelBase::kKernelLaunchOverhead * num_kernels;

    double compute_ms = absl::ToDoubleMilliseconds(total_compute_time);
    double memory_ms  = absl::ToDoubleMilliseconds(total_memory_time);
    double exec_ms    = absl::ToDoubleMilliseconds(total_exec_time);
    std::cerr << "[" << name << "] Kernel count: " << num_kernels << "\n"
              << "[" << name << "] Instruction count: " << instruction_count << "\n"
              << "[" << name << "] Total FLOPs: " << total_flops << "\n"
              << "[" << name << "] Bytes read: " << total_bytes_read / 1.0e6 << " MB\n"
              << "[" << name << "] Bytes written: " << total_bytes_written / 1.0e6 << " MB\n"
              << "[" << name << "] Total HBM traffic: "
              << (total_bytes_read + total_bytes_written) / 1.0e6 << " MB\n"
              << "[" << name << "] Compute time: " << compute_ms << "ms\n"
              << "[" << name << "] Memory time: " << memory_ms << "ms\n"
              << "[" << name << "] Estimated exec time: " << exec_ms
              << "ms (GpuPerformanceModel)\n";
    std::cerr << (compute_ms > memory_ms ? ">>> Compute-bound <<<\n"
                                         : ">>> Memory-bound <<<\n");

    CostMetrics metrics;
    metrics.kernel_count      = num_kernels;
    metrics.total_flops       = total_flops;
    metrics.total_exec_time   = total_exec_time;
    metrics.instruction_count = instruction_count;
    metrics.total_hbm_bytes   = total_bytes_read + total_bytes_written;
    return metrics;
  };

  auto lhs_metrics = run_cost_analysis(lhs_module.get(), "LHS");
  std::cerr << "-\n";
  auto rhs_metrics = run_cost_analysis(rhs_module.get(), "RHS");
  std::cerr << "-------------------------------------------\n";

  // ── Phase 4: Compile LHS ──────────────────────────────────────────────────
  std::cerr << "Compiling LHS Executable...\n";
  auto compile_start = std::chrono::high_resolution_clock::now();
  lhs_module->mutable_config()
      .mutable_debug_options()
      .add_xla_disable_hlo_passes("algsimp");
  auto lhs_exec_or =
      runner.CreateExecutable(std::move(lhs_module), /*run_hlo_passes=*/true);
  if (!lhs_exec_or.ok()) {
    std::cerr << "Runtime error (LHS): " << lhs_exec_or.status().message() << "\n";
    PrintConclusion(HloCompConclusion::kRuntimeError, "Original HLO Program",
                    lhs_exec_or.status().message());
    return lhs_exec_or.status();
  }
  std::unique_ptr<OpaqueExecutable> lhs_exec = std::move(lhs_exec_or).value();
  auto compile_mid = std::chrono::high_resolution_clock::now();
  std::cerr << "LHS compiled in "
            << std::chrono::duration<double>(compile_mid - compile_start).count()
            << "s.\n";

  // ── Phase 4b: Compile RHS ─────────────────────────────────────────────────
  std::cerr << "Compiling RHS Executable...\n";
  rhs_module->mutable_config()
      .mutable_debug_options()
      .add_xla_disable_hlo_passes("algsimp");
  auto rhs_exec_or =
      runner.CreateExecutable(std::move(rhs_module), /*run_hlo_passes=*/true);
  if (!rhs_exec_or.ok()) {
    std::cerr << "Runtime error (RHS): " << rhs_exec_or.status().message() << "\n";
    PrintConclusion(HloCompConclusion::kRuntimeError, "HLO program proposed by LLM",
                    rhs_exec_or.status().message());
    return rhs_exec_or.status();
  }
  std::unique_ptr<OpaqueExecutable> rhs_exec = std::move(rhs_exec_or).value();
  auto compile_end = std::chrono::high_resolution_clock::now();
  std::cerr << "RHS compiled in "
            << std::chrono::duration<double>(compile_end - compile_mid).count()
            << "s.\n";

  // ── Phase 5: Generate inputs ──────────────────────────────────────────────
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

  auto make_ptrs = [](const std::vector<Literal>& args) {
    std::vector<const Literal*> ptrs;
    ptrs.reserve(args.size());
    for (const auto& a : args) ptrs.push_back(&a);
    return ptrs;
  };

  // ── Phase 5: Run LHS ──────────────────────────────────────────────────────
  std::cerr << "\nRunning LHS block...\n";
  std::vector<Literal> lhs_results(opts.iterations);
  std::vector<double>  lhs_times(opts.iterations);
  for (int i = 0; i < total_iters; ++i) {
    ExecutionProfile profile;
    auto ptrs = make_ptrs(all_args[i]);
    auto result_or = runner.ExecuteWithExecutableAndProfile(
        lhs_exec.get(), ptrs, &profile);
    if (!result_or.ok()) {
      std::cerr << "Runtime error (LHS): " << result_or.status().message() << "\n";
      PrintConclusion(HloCompConclusion::kRuntimeError, "Original HLO Program",
                      result_or.status().message());
      return result_or.status();
    }
    if (i < 2) {
      std::cerr << "LHS warm-up iteration " << i + 1 << " completed.\n";
      continue;
    }
    lhs_results[i - 2] = std::move(result_or).value();
    lhs_times[i - 2] = static_cast<double>(profile.compute_time_ns()) / 1e6;
    std::cerr << "LHS iteration " << i - 1 << " completed in "
              << lhs_times[i - 2] << "ms.\n";
  }

  // ── Phase 6: Run RHS and compare ─────────────────────────────────────────
  std::cerr << "\nRunning RHS block...\n";
  std::vector<double> rhs_times(opts.iterations);
  for (int i = 0; i < total_iters; ++i) {
    ExecutionProfile profile;
    auto ptrs = make_ptrs(all_args[i]);
    auto result_or = runner.ExecuteWithExecutableAndProfile(
        rhs_exec.get(), ptrs, &profile);
    if (!result_or.ok()) {
      std::cerr << "Runtime error (RHS): " << result_or.status().message() << "\n";
      PrintConclusion(HloCompConclusion::kRuntimeError, "HLO program proposed by LLM",
                      result_or.status().message());
      return result_or.status();
    }
    if (i < 2) {
      std::cerr << "RHS warm-up iteration " << i + 1 << " completed.\n";
      continue;
    }
    Literal rhs_result = std::move(result_or).value();
    rhs_times[i - 2] = static_cast<double>(profile.compute_time_ns()) / 1e6;

    absl::Status cmp = literal_comparison::Near(
        lhs_results[i - 2], rhs_result, error_spec,
        /*detailed_message=*/true, &OnMiscompare);
    if (!cmp.ok()) {
      std::cerr << "Mismatch detected at iteration " << i - 1 << "!\n";
      PrintConclusion(HloCompConclusion::kNotEquivalent);
      return cmp;
    }
    std::cerr << "RHS iteration " << i - 1 << " completed in "
              << rhs_times[i - 2] << "ms.\n";
  }

  // ── Phase 7: Verdict ──────────────────────────────────────────────────────
  double total_lhs_time = 0.0, total_rhs_time = 0.0;
  for (int i = 0; i < opts.iterations; ++i) {
    total_lhs_time += lhs_times[i];
    total_rhs_time += rhs_times[i];
  }
  double avg_lhs_ms = total_lhs_time / opts.iterations;
  double avg_rhs_ms = total_rhs_time / opts.iterations;

  std::cerr << "\nSuccess! LHS and RHS are equivalent across "
            << opts.iterations << " random inputs.\n";

  HloCompConclusion verdict =
      DetermineVerdict(lhs_metrics, rhs_metrics, avg_lhs_ms, avg_rhs_ms,
                       lhs_raw_inst_count, rhs_raw_inst_count);
  PrintConclusion(verdict);
  if (verdict == HloCompConclusion::kRhsBetter ||
      verdict == HloCompConclusion::kRhsNotBetter) {
    PrintComparisonTable(lhs_metrics, rhs_metrics, avg_lhs_ms, avg_rhs_ms,
                         lhs_raw_inst_count, rhs_raw_inst_count);
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  xla::HloCompConfig opts;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("input_format", &opts.input_format,
                "The format of the input file."),
      tsl::Flag("lhs_file", &opts.lhs_file,
                "Path to the left-hand-side HLO module."),
      tsl::Flag("rhs_file", &opts.rhs_file,
                "Path to the right-hand-side HLO module."),
      tsl::Flag("platform", &opts.platform,
                "The test platform (gpu, cpu, etc)."),
      tsl::Flag("iterations", &opts.iterations,
                "The number of times to run the module."),
      tsl::Flag("dump_dir", &opts.dump_dir,
                "If set, dump post-optimization HLO for LHS and RHS into this "
                "directory.")};

  const std::string kUsageString =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[0], flag_list));

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
