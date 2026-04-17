#ifndef XLA_TOOLS_HLO_COMP_H_
#define XLA_TOOLS_HLO_COMP_H_

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {


struct HloCompConfig {
  std::string lhs_file{""};
  std::string rhs_file{""};
  std::string input_format{""};
  std::string platform;
  int iterations{1};
};

}  // namespace xla

#endif  // XLA_TOOLS_HLO_COMP_H_