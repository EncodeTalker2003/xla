#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/literal.h"
#include "xla/literal_comparison.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/test_utils.h"
#include "xla/tools/hlo_cse.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/path.h"

namespace {
const char* const kUsage = R"(
HLO Common Subexpression Eliminator

Finds structurally identical subexpressions in both LHS and RHS HLO programs
and replaces them with shared parameters, preserving equivalence throughout.

Usage:
  bazel run //xla/tools:hlo_cse -- \
    --lhs_file=path/to/lhs.hlo \
    --rhs_file=path/to/rhs.hlo \
    --platform=CUDA \
    --iterations=5 \
    --output_dir=/tmp/cse_output
)";
}  // namespace

namespace xla {
namespace {

void OnMiscompare(const LiteralSlice& expected, const LiteralSlice& actual,
                  const LiteralSlice& /*mismatches*/,
                  const ShapeIndex& /*shape_index*/,
                  const literal_comparison::ErrorBuckets& /*error_buckets*/) {
  std::cerr << "\n>>> Mismatch Detected <<<\n"
            << "LHS: " << ShapeUtil::HumanString(expected.shape()) << "\n"
            << "RHS: " << ShapeUtil::HumanString(actual.shape()) << "\n";
}

// Compute a position-independent token for a single instruction.
// Strips the canonical "name = " prefix from the Fingerprint output so that
// structurally identical instructions at different topological positions
// (and thus different canonical names) yield the same token.
uint64_t InstructionToken(const HloInstruction* instr) {
  std::string s = instr->ToString(HloPrintOptions::Fingerprint());
  size_t eq_pos = s.find(" = ");
  if (eq_pos != std::string::npos) {
    s = s.substr(eq_pos + 3);
  }
  return tsl::Fingerprint64(s);
}

// Recursive subgraph fingerprint: operand-order-sensitive, name-independent.
// Parameters are identified by number+shape; operations by opcode+shape+children.
uint64_t SubgraphFp(const HloInstruction* instr,
                    absl::flat_hash_map<const HloInstruction*, uint64_t>& cache) {
  if (auto it = cache.find(instr); it != cache.end()) return it->second;
  uint64_t fp = InstructionToken(instr);
  for (const HloInstruction* op : instr->operands()) {
    fp = tsl::FingerprintCat64(fp, SubgraphFp(op, cache));
  }
  cache[instr] = fp;
  return fp;
}

// Compute the depth (height) of an instruction's subgraph (0 for leaves).
int64_t SubgraphHeight(
    const HloInstruction* instr,
    absl::flat_hash_map<const HloInstruction*, int64_t>& cache) {
  if (auto it = cache.find(instr); it != cache.end()) return it->second;
  int64_t h = 0;
  for (const HloInstruction* op : instr->operands()) {
    h = std::max(h, SubgraphHeight(op, cache) + 1);
  }
  cache[instr] = h;
  return h;
}

struct CandidateInfo {
  uint64_t fingerprint;
  Shape shape;
  int64_t height;
};

// Build a fingerprint→CandidateInfo map for all non-leaf, non-dead, non-fused
// instructions in the entry computation.  Only one entry per fingerprint
// (the first seen); the shape is captured for later parameter creation.
absl::flat_hash_map<uint64_t, CandidateInfo> CollectCandidates(
    HloModule* module) {
  HloComputation* entry = module->entry_computation();
  absl::flat_hash_map<const HloInstruction*, uint64_t> fp_cache;
  absl::flat_hash_map<const HloInstruction*, int64_t> height_cache;
  absl::flat_hash_map<uint64_t, CandidateInfo> result;

  for (HloInstruction* instr : entry->instructions()) {
    switch (instr->opcode()) {
      case HloOpcode::kParameter:
      case HloOpcode::kConstant:
      case HloOpcode::kFusion:
      case HloOpcode::kWhile:
      case HloOpcode::kCall:
      case HloOpcode::kConditional:
        continue;
      default:
        break;
    }
    if (instr->IsDead()) continue;

    uint64_t fp = SubgraphFp(instr, fp_cache);
    int64_t h = SubgraphHeight(instr, height_cache);

    if (!result.contains(fp)) {
      result[fp] = {fp, instr->shape(), h};
    }
  }
  return result;
}

// Replace all live, eligible instructions in comp that have fingerprint
// target_fp with new_param.  Handles the root instruction specially by
// calling set_root_instruction rather than ReplaceAllUsesWith.
void ReplaceAllMatchingWithParam(HloComputation* comp,
                                  uint64_t target_fp,
                                  HloInstruction* new_param) {
  absl::flat_hash_map<const HloInstruction*, uint64_t> fp_cache;
  std::vector<HloInstruction*> to_replace;

  for (HloInstruction* instr : comp->instructions()) {
    if (instr == new_param) continue;
    switch (instr->opcode()) {
      case HloOpcode::kParameter:
      case HloOpcode::kConstant:
        continue;
      default:
        break;
    }
    if (instr->IsDead() && instr != comp->root_instruction()) continue;

    if (SubgraphFp(instr, fp_cache) == target_fp) {
      to_replace.push_back(instr);
    }
  }

  for (HloInstruction* instr : to_replace) {
    if (instr == comp->root_instruction()) {
      // Root has no users; redirect it by setting the new param as root.
      comp->set_root_instruction(new_param);
    } else {
      CHECK_OK(instr->ReplaceAllUsesWith(new_param));
    }
  }
}

// Run a random equivalence test between lhs and rhs.
// Compiles CLONES (does not consume the originals).
// Uses signature_module to determine the parameter shapes for random arg gen.
absl::Status TestEquivalence(HloModule* lhs, HloModule* rhs,
                              HloModule* signature,
                              const HloCseConfig& opts,
                              HloRunner& runner) {
  TF_ASSIGN_OR_RETURN(auto lhs_exec,
      runner.CreateExecutable(lhs->Clone(), /*run_hlo_passes=*/true));
  TF_ASSIGN_OR_RETURN(auto rhs_exec,
      runner.CreateExecutable(rhs->Clone(), /*run_hlo_passes=*/true));

  std::minstd_rand0 engine;
  ErrorSpec error_spec(1e-3, 1e-3);

  for (int i = 0; i < opts.iterations; ++i) {
    ExecutionProfile profile;
    TF_ASSIGN_OR_RETURN(std::vector<Literal> args,
        MakeFakeArguments(signature, &engine,
                          /*use_large_range=*/false,
                          /*treat_gte_as_data_formatting=*/false));

    std::vector<const Literal*> arg_ptrs;
    arg_ptrs.reserve(args.size());
    for (const auto& arg : args) arg_ptrs.push_back(&arg);

    TF_ASSIGN_OR_RETURN(Literal lhs_result,
        runner.ExecuteWithExecutableAndProfile(
            lhs_exec.get(), arg_ptrs, &profile));
    TF_ASSIGN_OR_RETURN(Literal rhs_result,
        runner.ExecuteWithExecutableAndProfile(
            rhs_exec.get(), arg_ptrs, &profile));

    TF_RETURN_IF_ERROR(literal_comparison::Near(
        lhs_result, rhs_result, error_spec,
        /*detailed_message=*/true, &OnMiscompare));
  }
  return absl::OkStatus();
}

// Remove all dead non-parameter instructions from comp, repeating until stable.
absl::Status RemoveDeadInstructions(HloComputation* comp) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      if (instr->opcode() == HloOpcode::kParameter) continue;
      if (instr->IsDead()) {
        TF_RETURN_IF_ERROR(comp->RemoveInstruction(instr));
        changed = true;
        break;  // restart with fresh post-order after each removal
      }
    }
  }
  return absl::OkStatus();
}

// Remove parameters that are dead in BOTH lhs and rhs entry computations.
// Processes in decreasing index order to avoid renumbering issues.
absl::Status RemoveDeadParamsSynchronized(HloComputation* lhs_entry,
                                           HloComputation* rhs_entry) {
  int64_t N = lhs_entry->num_parameters();
  CHECK_EQ(rhs_entry->num_parameters(), N);

  std::vector<int64_t> to_remove;
  for (int64_t i = 0; i < N; ++i) {
    if (lhs_entry->parameter_instruction(i)->IsDead() &&
        rhs_entry->parameter_instruction(i)->IsDead()) {
      to_remove.push_back(i);
    }
  }

  // Process in DECREASING index order: RemoveParameter(i) renumbers params
  // at indices > i, so lower indices remain stable as we work down.
  for (int64_t ri = static_cast<int64_t>(to_remove.size()) - 1; ri >= 0;
       --ri) {
    int64_t idx = to_remove[ri];
    TF_RETURN_IF_ERROR(lhs_entry->RemoveParameter(idx));
    TF_RETURN_IF_ERROR(rhs_entry->RemoveParameter(idx));
  }
  return absl::OkStatus();
}

absl::Status RunHloCse(const HloCseConfig& opts) {
  // ---- Phase 1: Load, Verify, Signature Check ----
  std::string format = opts.input_format;
  if (format.empty()) {
    format = std::string(tsl::io::Extension(opts.lhs_file));
  }

  std::cerr << "Loading LHS module...\n";
  TF_ASSIGN_OR_RETURN(auto current_lhs,
      LoadModuleFromFile(opts.lhs_file, format));

  std::cerr << "Loading RHS module...\n";
  TF_ASSIGN_OR_RETURN(auto current_rhs,
      LoadModuleFromFile(opts.rhs_file, format));

  HloVerifier verifier(
      HloVerifierOpts{}
          .WithLayoutSensitive(false)
          .WithAllowMixedPrecision(true));
  TF_RETURN_IF_ERROR(verifier.Run(current_lhs.get()).status());
  TF_RETURN_IF_ERROR(verifier.Run(current_rhs.get()).status());

  const auto* lhs_entry = current_lhs->entry_computation();
  const auto* rhs_entry = current_rhs->entry_computation();

  if (lhs_entry->num_parameters() != rhs_entry->num_parameters()) {
    return absl::InvalidArgumentError(
        "LHS and RHS have different numbers of parameters.");
  }
  for (int i = 0; i < lhs_entry->num_parameters(); ++i) {
    if (!ShapeUtil::Equal(lhs_entry->parameter_instruction(i)->shape(),
                          rhs_entry->parameter_instruction(i)->shape())) {
      return absl::InvalidArgumentError(
          absl::StrCat("Parameter ", i, " shapes differ between LHS and RHS."));
    }
  }

  // ---- Phase 2: Initial Equivalence Check ----
  TF_ASSIGN_OR_RETURN(se::Platform* platform,
      PlatformUtil::GetPlatform(opts.platform));
  HloRunner runner(platform);

  std::cerr << "Running initial equivalence check...\n";
  TF_RETURN_IF_ERROR(TestEquivalence(current_lhs.get(), current_rhs.get(),
                                      current_lhs.get(), opts, runner));
  std::cerr << "Initial equivalence check PASSED.\n";

  // ---- Phase 3: CSE Substitution Loop ----
  int cse_count = 0;

  for (int outer_round = 0; outer_round < opts.max_cse_rounds; ++outer_round) {
    auto lhs_candidates = CollectCandidates(current_lhs.get());
    auto rhs_candidates = CollectCandidates(current_rhs.get());

    // Intersect fingerprint sets; verify shapes agree (should always be true
    // when fingerprint includes shape).
    std::vector<CandidateInfo> common;
    for (auto& [fp, info] : lhs_candidates) {
      auto it = rhs_candidates.find(fp);
      if (it != rhs_candidates.end() &&
          ShapeUtil::Equal(info.shape, it->second.shape)) {
        common.push_back(info);
      }
    }

    if (common.empty()) {
      std::cerr << "No common subexpressions found in round " << outer_round
                << ". Stopping.\n";
      break;
    }

    // Sort by subgraph height descending: try larger subexpressions first.
    std::sort(common.begin(), common.end(),
              [](const CandidateInfo& a, const CandidateInfo& b) {
                return a.height > b.height;
              });

    bool committed = false;
    for (const CandidateInfo& cand : common) {
      std::cerr << "  Trying CSE for subgraph height=" << cand.height
                << " shape=" << ShapeUtil::HumanString(cand.shape) << "...\n";

      auto test_lhs = current_lhs->Clone();
      auto test_rhs = current_rhs->Clone();

      int64_t new_param_idx =
          test_lhs->entry_computation()->num_parameters();
      std::string param_name = absl::StrCat("cse_p_", cse_count);

      HloInstruction* P_lhs =
          test_lhs->entry_computation()->AddEntryComputationParameter(
              HloInstruction::CreateParameter(
                  new_param_idx, cand.shape, param_name));
      HloInstruction* P_rhs =
          test_rhs->entry_computation()->AddEntryComputationParameter(
              HloInstruction::CreateParameter(
                  new_param_idx, cand.shape, param_name));

      ReplaceAllMatchingWithParam(
          test_lhs->entry_computation(), cand.fingerprint, P_lhs);
      ReplaceAllMatchingWithParam(
          test_rhs->entry_computation(), cand.fingerprint, P_rhs);

      // Use test_lhs as the signature for MakeFakeArguments (not consumed).
      absl::Status test_status = TestEquivalence(
          test_lhs.get(), test_rhs.get(), test_lhs.get(), opts, runner);

      if (test_status.ok()) {
        std::cerr << "  CSE substitution ACCEPTED (round=" << outer_round
                  << " total=" << cse_count + 1 << ").\n";
        current_lhs = std::move(test_lhs);
        current_rhs = std::move(test_rhs);
        ++cse_count;
        committed = true;
        break;
      } else {
        std::cerr << "  Rejected (equivalence not preserved): "
                  << test_status.message() << "\n";
      }
    }

    if (!committed) {
      std::cerr << "No valid CSE found in round " << outer_round
                << ". Stopping.\n";
      break;
    }
  }

  std::cerr << "\nTotal CSE substitutions performed: " << cse_count << "\n";

  // ---- Phase 4: Dead Code Cleanup ----
  std::cerr << "Removing dead instructions...\n";
  TF_RETURN_IF_ERROR(
      RemoveDeadInstructions(current_lhs->entry_computation()));
  TF_RETURN_IF_ERROR(
      RemoveDeadInstructions(current_rhs->entry_computation()));

  std::cerr << "Removing dead parameters (synchronized)...\n";
  TF_RETURN_IF_ERROR(RemoveDeadParamsSynchronized(
      current_lhs->entry_computation(),
      current_rhs->entry_computation()));

  // ---- Phase 5: Output ----
  if (opts.output_dir.empty()) {
    return absl::InvalidArgumentError("--output_dir must be specified.");
  }
  TF_RETURN_IF_ERROR(
      tsl::Env::Default()->RecursivelyCreateDir(opts.output_dir));

  HloPrintOptions print_opts;
  print_opts.set_print_metadata(false);

  // Derive output filename from the input basename, replacing extension with
  // _cse.hlo.
  auto make_out_path = [&](const std::string& input_path) {
    std::string base = std::string(tsl::io::Basename(input_path));
    size_t dot = base.rfind('.');
    if (dot != std::string::npos) base = base.substr(0, dot);
    return tsl::io::JoinPath(opts.output_dir, base + "_cse.hlo");
  };

  std::string lhs_out = make_out_path(opts.lhs_file);
  std::string rhs_out = make_out_path(opts.rhs_file);

  {
    std::ofstream f(lhs_out);
    if (!f.is_open()) {
      return absl::InternalError(
          absl::StrCat("Cannot open ", lhs_out, " for writing."));
    }
    f << current_lhs->ToString(print_opts);
    std::cerr << "Written LHS to: " << lhs_out << "\n";
  }
  {
    std::ofstream f(rhs_out);
    if (!f.is_open()) {
      return absl::InternalError(
          absl::StrCat("Cannot open ", rhs_out, " for writing."));
    }
    f << current_rhs->ToString(print_opts);
    std::cerr << "Written RHS to: " << rhs_out << "\n";
  }

  std::cerr << "LHS final parameter count: "
            << current_lhs->entry_computation()->num_parameters() << "\n";
  std::cerr << "RHS final parameter count: "
            << current_rhs->entry_computation()->num_parameters() << "\n";

  return absl::OkStatus();
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  xla::HloCseConfig opts;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("input_format", &opts.input_format,
                "Format of the input files (hlo|pb|pbtxt|stablehlo)."),
      tsl::Flag("lhs_file", &opts.lhs_file,
                "Path to the left-hand-side HLO module."),
      tsl::Flag("rhs_file", &opts.rhs_file,
                "Path to the right-hand-side HLO module."),
      tsl::Flag("platform", &opts.platform,
                "Execution platform (CUDA|CPU|Interpreter)."),
      tsl::Flag("iterations", &opts.iterations,
                "Random test iterations per CSE candidate."),
      tsl::Flag("max_cse_rounds", &opts.max_cse_rounds,
                "Maximum number of outer CSE substitution rounds."),
      tsl::Flag("output_dir", &opts.output_dir,
                "Directory to write the output HLO files."),
  };

  const std::string kUsageString =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[0], flag_list));
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  if (!parse_ok) {
    std::cerr << kUsageString;
    return 1;
  }
  tsl::port::InitMain(kUsageString.c_str(), &argc, &argv);

  absl::Status status = xla::RunHloCse(opts);
  if (!status.ok()) {
    std::cerr << status << "\n";
    return 1;
  }
  return 0;
}
