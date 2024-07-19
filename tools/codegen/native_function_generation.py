"""Extension of native function generation based on torchgen"""

from torchgen.native_function_generation import (
    functional_to_out_signature,
    mutable_to_out_signature,
    self_to_out_signature,
)
from torchgen.model import (
    FunctionSchema,
    BaseOperatorName,
    OperatorName,
    SchemaKind,
)


def autogen_signature(
    schema: FunctionSchema,
    op_name: str,
) -> FunctionSchema:
    """Generate native function schema for the corresponding kind based on the input op name"""
    input_kind: SchemaKind = schema.kind()
    if "out" in op_name:
        if input_kind == SchemaKind.inplace:
            return self_to_out_signature(schema)
        if input_kind == SchemaKind.mutable:
            return mutable_to_out_signature(schema)
        if input_kind == SchemaKind.functional:
            return functional_to_out_signature(schema)
        raise AssertionError(
            "Only generate out= functions from either inplace or mutable or functional variants"
        )
    functional_schema = schema.signature(keep_return_names=True).with_name(
        OperatorName(
            name=BaseOperatorName(
                base=schema.name.name.base,
                inplace=False,
                dunder_method=schema.name.name.dunder_method,
                functional_overload=schema.kind() == SchemaKind.mutable,
            ),
            overload_name=schema.name.overload_name,
        )
    )
    return functional_schema
