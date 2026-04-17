#ifndef XLA_TOOLS_XLA_SLICER_H_
#define XLA_TOOLS_XLA_SLICER_H_

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {


struct XlaSlicerConfig {
  std::string input_file{""};
  std::string input_format{""};
  // Slicing depth, etc ...
};

}  // namespace xla

#endif  // XLA_TOOLS_XLA_SLICER_H_