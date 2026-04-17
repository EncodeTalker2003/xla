#include "xla/tools/xla_slicer.h"

#include <iostream>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/xla_data.pb.h"
#include "xla/tools/hlo_slicer.h"

namespace xla {

}  // namespace xla