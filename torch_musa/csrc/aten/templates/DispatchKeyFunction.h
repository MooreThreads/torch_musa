// clang-format off
#ifndef BUILD_TORCH_MUSA_CODEGEN_ATEN_OPS_${upper_root_name}_${upper_dispatch_namespace}_DISPATCH_H_
#define BUILD_TORCH_MUSA_CODEGEN_ATEN_OPS_${upper_root_name}_${upper_dispatch_namespace}_DISPATCH_H_

// ${generated_comment}

// NB: The implementing C++ file is RegisterDispatchKey.cpp

// The only #includes we need are for custom classes that have defaults in the C++ API
#include <c10/core/MemoryFormat.h>
#include <c10/core/Scalar.h>
#include <ATen/core/Reduction.h>

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>

namespace at {

namespace ${dispatch_namespace} {

${dispatch_namespaced_declarations}

} // namespace ${dispatch_namespace}
} // namespace at

#endif // BUILD_TORCH_MUSA_CODEGEN_ATEN_OPS_${upper_root_name}_${upper_dispatch_namespace}_DISPATCH_H_
