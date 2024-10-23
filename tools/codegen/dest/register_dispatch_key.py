"""torch_musa register_dispatch_key, borrowed from pytorch/torchgen"""

# pylint: disable=C0115,C0116,C0301
import itertools
import textwrap
from typing import List, Optional, Tuple, Union

from typing_extensions import Literal  # Python 3.8+

from torchgen.api import structured
from torchgen.api.translate import translate
from torchgen.api.types import (
    BaseCType,
    Binding,
    ConstRefCType,
    CppSignature,
    CppSignatureGroup,
    Expr,
    MutRefCType,
    NamedCType,
    NativeSignature,
    tensorT,
)
from torchgen.context import method_with_native_function
from torchgen.model import (
    BackendIndex,
    NativeFunction,
    SchemaKind,
)
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import assert_never, Target

from codegen.model import MUSA_DISPATCH_KEYS, MUSA_STRUCTURED_DISPATCH_KEY


def gen_registration_headers(
    backend_index: BackendIndex,
) -> List[str]:
    assert backend_index.dispatch_key in MUSA_DISPATCH_KEYS
    headers = ["#include <ATen/ops/as_strided_native.h>"]
    headers.append("#include <torch_musa/csrc/aten/ops/TensorFactory.h>")
    return headers


def musa_gen_empty_impl_names(
    backend_index: BackendIndex,
) -> Tuple[Optional[str], Optional[str]]:
    empty_impl = None
    empty_strided_impl = None

    assert backend_index.dispatch_key in MUSA_DISPATCH_KEYS
    empty_impl = "at::detail::empty_musa"
    empty_strided_impl = "at::detail::empty_strided_musa"

    return empty_impl, empty_strided_impl


def musa_create_register_dispatch_key(
    backend_index: BackendIndex,
    target: Union[
        Literal[Target.ANONYMOUS_DEFINITION],
        Literal[Target.NAMESPACED_DEFINITION],
        Literal[Target.NAMESPACED_DECLARATION],
        Literal[Target.REGISTRATION],
    ],
    selector: SelectiveBuilder,
    symint: bool,
    class_method_name: Optional[str],
    skip_dispatcher_op_registration: bool,
):
    from torchgen.dest import RegisterDispatchKey  # pylint: disable=C0415

    rdk = RegisterDispatchKey(
        backend_index=backend_index,
        target=target,
        selector=selector,
        rocm=False,
        symint=symint,
        class_method_name=class_method_name,
        skip_dispatcher_op_registration=skip_dispatcher_op_registration,
    )
    return rdk


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           STRUCTURED
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def musa_gen_class_set_output_body(
    self, k: SchemaKind, maybe_create_proxy: bool
) -> str:
    if self.backend_index.dispatch_key is MUSA_STRUCTURED_DISPATCH_KEY:
        maybe_set_guard = """
auto current_device = guard_.current_device();
if (C10_UNLIKELY(current_device.has_value())) {
  TORCH_INTERNAL_ASSERT(*current_device == options.device(),
    "structured kernels don't support multi-device outputs");
} else {
  guard_.reset_device(options.device());
}
"""
        maybe_set_guard_line = maybe_set_guard + "\n"
    else:
        maybe_set_guard_line = maybe_set_guard = ""

    if maybe_create_proxy:
        create_proxy = """
auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);
if (C10_UNLIKELY(maybe_proxy.has_value())) {
    proxy_outputs_[output_idx] = std::move(maybe_proxy).value();
}
"""
    else:
        create_proxy = ""

    if k is SchemaKind.functional:
        assert self.backend_index.dispatch_key is MUSA_STRUCTURED_DISPATCH_KEY
        return f"""{maybe_set_guard_line}
outputs_[output_idx] = create_out(sizes, strides, options);"""
    if k is SchemaKind.inplace:
        return f"""{maybe_set_guard_line}
const auto& out = outputs_[output_idx].get();
check_inplace(out, sizes, options);
{create_proxy}"""
    if k is SchemaKind.out:
        return f"""{maybe_set_guard_line}
const auto& out = outputs_[output_idx].get();
resize_out(out, sizes, strides, options);
{create_proxy}"""
    if k is SchemaKind.mutable or k is SchemaKind.scratch:
        raise AssertionError(f"{k} structured operators are currently not supported")
    assert_never(k)


def musa_gen_class(
    self,
    f: NativeFunction,
    k: SchemaKind,
    *,
    class_name: str,
    parent_class: str,
    generate_super: bool,
) -> str:
    if k is SchemaKind.functional:
        output_type = "Tensor"
        output_value = "outputs_[output_idx]"
        proxy_field = ""
    elif k is SchemaKind.inplace:
        output_type = "std::reference_wrapper<Tensor>"
        output_value = "proxy_outputs_[output_idx].has_value() ? *proxy_outputs_[output_idx] : outputs_[output_idx].get()"
        proxy_field = (
            f"std::array<c10::optional<Tensor>, {len(f.func.returns)}> proxy_outputs_;"
        )
    elif k is SchemaKind.out:
        output_type = "std::reference_wrapper<Tensor>"
        output_value = "proxy_outputs_[output_idx].has_value() ? *proxy_outputs_[output_idx] : outputs_[output_idx].get()"
        proxy_field = (
            f"std::array<c10::optional<Tensor>, {len(f.func.returns)}> proxy_outputs_;"
        )

    if self.backend_index.dispatch_key == MUSA_STRUCTURED_DISPATCH_KEY:
        guard_field = "c10::musa::OptionalMUSAGuard guard_;"
    else:
        guard_field = ""

    indent = " " * 4
    class_ctor_str = self.gen_class_ctor(k, class_name, len(f.func.returns))
    lines = (
        f"struct {class_name} final : public {parent_class} {{",
        f"{textwrap.indent(class_ctor_str, indent)}",
        f"{textwrap.indent(self.gen_class_set_output_functions(k, parent_class, generate_super), indent)}",
        "    const Tensor& maybe_get_output(int64_t output_idx) override {",
        f"      return {output_value};\n",
        "    }",
        f"    std::array<{output_type}, {len(f.func.returns)}> outputs_;",
        f"{textwrap.indent(proxy_field, indent)}",
        f"{textwrap.indent(guard_field, indent)}",
        "};",
    )
    return "\n".join(line for line in lines if line)


@method_with_native_function
def musa_gen_one(self, f: NativeFunction) -> Optional[str]:
    from torchgen.dest import RegisterDispatchKey  # pylint: disable=C0415

    assert not f.manual_kernel_registration

    if (
        self.target is Target.REGISTRATION
        and not self.selector.is_native_function_selected(f)
    ):
        return None

    cpp_sig_group = CppSignatureGroup.from_native_function(
        f, method=False, fallback_binding=False
    )

    # Signature of the wrapper function we'll register to the dispatcher
    kern = self.backend_index.get_kernel(f)
    sig = NativeSignature(
        f.func,
        prefix=f"wrapper_{self.backend_index.dispatch_key}_",
        symint=kern is not None and kern.supports_symint(),
    )

    if self.target is Target.NAMESPACED_DECLARATION:
        result = ""
        for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
            result += f"TORCH_API {cpp_sig.decl()};\n"
        return result

    if self.target is Target.NAMESPACED_DEFINITION:

        def generate_defn(cpp_sig: CppSignature) -> str:
            return f"""
{cpp_sig.defn()} {{
return {sig.name()}({', '.join(e.expr for e in translate(cpp_sig.arguments(), sig.arguments()))});
}}
"""

        result = ""
        for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
            result += generate_defn(cpp_sig)
        return result

    if self.target is Target.ANONYMOUS_DEFINITION:
        k = f.func.kind()

        # Construct the body of the wrapper function with signature sig
        sig_body = []
        # We'll use context to keep track of any variables we've brought
        # into scope while generating code
        context: List[Union[Binding, Expr]] = list(sig.arguments())

        # Initialize the class corresponding to this structured
        # operator; feeding it the output argument(s) if it is known
        metadata = self.backend_index.get_kernel(self.g)
        assert metadata is not None
        class_name = f"structured_{metadata.kernel}_{k.name}"
        parent_class = f"{metadata.cpp_namespace}::structured_{metadata.kernel}"

        if self.backend_index.device_guard:
            device_check_args = itertools.chain(
                f.func.arguments.out, f.func.arguments.flat_positional
            )
            sig_body.append(
                RegisterDispatchKey.gen_device_check(
                    f.device_check, list(device_check_args), sig.name()
                )
            )

        if k is SchemaKind.functional:
            sig_body.append(f"{class_name} op;")
        elif k is SchemaKind.inplace:
            sig_body.append(f"{class_name} op(self);")
        elif k is SchemaKind.out:
            out_args_str = ", ".join(a.name for a in f.func.arguments.out)
            sig_body.append(f"{class_name} op({out_args_str});")

        # Translate the input native arguments into structured
        # arguments for the meta call
        meta_exprs = ", ".join(
            e.expr
            for e in translate(context, structured.meta_arguments(self.g), method=False)
        )

        if self.g.out.precomputed:
            # If this function group has precomputed elements, the meta function
            # returns a struct containing them which must be saved so that it
            # can be unpacked when generating code to call the impl.
            sig_body.append(f"auto precompute = op.meta({meta_exprs});")

            # Put all of the contents of the precompute struct into the context
            # so that translate will be able to return the correct args for the
            # call to the impl.
            precomputed_values = [
                *self.g.out.precomputed.replace.values(),
                self.g.out.precomputed.add,
            ]
            for precomputed_elems in precomputed_values:
                for arg in precomputed_elems:
                    context.append(
                        Expr(
                            expr=f"precompute.{arg.name}",
                            type=structured.argument_type(arg, binds=arg.name),
                        )
                    )

            # Add a use of the precompute struct so FB internal compilers don't
            # complain that there is an unused variable.
            sig_body.append("(void)precompute;")
        else:
            sig_body.append(f"op.meta({meta_exprs});")

        # After running meta, op.outputs_ is guaranteed to be valid;
        # add it to the context
        out_args = structured.out_arguments(self.g)
        for i, out_arg in enumerate(out_args):
            assert ConstRefCType(BaseCType(tensorT)) == out_arg.nctype.type

            if k is SchemaKind.out:
                expr = f"op.maybe_get_output({i})"
            else:
                expr = f"op.outputs_[{i}]"

            context.append(
                Expr(
                    expr=expr,
                    # TODO: Stop hardcoding that the output type is a Tensor.  Note
                    # that for the codegen here this is fine because outputs_ is
                    # hardcoded to be tensor already
                    type=NamedCType(
                        out_arg.nctype.name, MutRefCType(BaseCType(tensorT))
                    ),
                )
            )

        # With the expanded context, do the impl call (if not a meta
        # function)
        impl_exprs = ", ".join(
            e.expr
            for e in translate(context, structured.impl_arguments(self.g), method=False)
        )
        sig_body.append(f"op.impl({impl_exprs});")

        # Go over each output, and check if there is a proxy created for it.
        # If so, copy it over to the original output.
        if k is SchemaKind.out or k is SchemaKind.inplace:
            for i in range(len(f.func.returns)):
                sig_body.append(
                    f"if (op.proxy_outputs_[{i}].has_value()) op.outputs_[{i}].get().copy_(*op.proxy_outputs_[{i}]);"
                )

        # Destructively return the final tensors
        # TODO: Do this in translate instead
        if k is SchemaKind.functional:
            if len(f.func.returns) == 1:
                ret_expr = "std::move(op.outputs_[0])"  # small optimization
            else:
                moved = ", ".join(
                    f"std::move(op.outputs_[{i}])" for i in range(len(f.func.returns))
                )
                ret_expr = f"std::make_tuple({moved})"
        elif k is SchemaKind.inplace:
            ret_expr = "self"
        elif k is SchemaKind.out:
            if len(f.func.returns) == 1:
                ret_expr = f.func.arguments.out[0].name
            else:
                refs = ", ".join(a.name for a in f.func.arguments.out)
                ret_expr = f"std::forward_as_tuple({refs})"
        sig_body.append(f"return {ret_expr};")

        sig_body_str = "\n".join(sig_body)

        # For an overview of what this template code looks like, see
        # https://github.com/pytorch/rfcs/pull/9
        return f"""\
{self.gen_class(
f, k,
class_name=class_name,
parent_class=parent_class,
generate_super=self.g.out.structured_inherits is not None
)}

{sig.defn()} {{
{sig_body_str}
}}
"""

    if self.target is Target.REGISTRATION:
        return f'm.impl("{f.func.name}", TORCH_FN({sig.name()}));'

    assert_never(self.target)


def init_for_musa_codegen() -> None:
    from torchgen.dest import register_dispatch_key  # pylint: disable=C0415

    setattr(register_dispatch_key, "gen_empty_impl_names", musa_gen_empty_impl_names)

    module_class = register_dispatch_key.StructuredRegisterDispatchKey
    setattr(module_class, "gen_class_set_output_body", musa_gen_class_set_output_body)
    setattr(module_class, "gen_class", musa_gen_class)
    setattr(module_class, "gen_one", musa_gen_one)
