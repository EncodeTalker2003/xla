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
	int depth{5};
	int max_inst_count{10};
	std::string output_dir{"./"};
	std::string name{"default"};
  // Slicing depth, etc ...
};

}  // namespace xla

#endif  // XLA_TOOLS_XLA_SLICER_H_