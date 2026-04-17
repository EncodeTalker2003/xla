#ifndef XLA_TOOLS_HLO_COMP_H_
#define XLA_TOOLS_HLO_COMP_H_

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {


struct XlaSlicerConfig {
  std::string lhs_file{""};
  std::string rhs_file{""};
  std::string input_format{""};
  // Slicing depth, etc ...
};

}  // namespace xla

#endif  // XLA_TOOLS_HLO_COMP_H_