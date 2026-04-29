#ifndef XLA_TOOLS_HLO_CSE_H_
#define XLA_TOOLS_HLO_CSE_H_

#include <string>

namespace xla {

struct HloCseConfig {
  std::string lhs_file{""};
  std::string rhs_file{""};
  std::string input_format{""};
  std::string platform{"CUDA"};
  int iterations{5};        // random test iterations per candidate
  int max_cse_rounds{10};  // max outer substitution rounds
  std::string output_dir{""};
};

}  // namespace xla

#endif  // XLA_TOOLS_HLO_CSE_H_
