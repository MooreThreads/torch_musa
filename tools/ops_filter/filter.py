import argparse
from collections import OrderedDict
import logging

from torchgen.model import (
    OperatorName,
    DispatchKey,
)

from .consts import OpKind
from .export import export_csv
from .rules import clean_torch_ops
from .utils import (
    get_torch_ops,
    get_musa_ops,
    this_dir,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kinds",
        type=str,
        default="cuda_custom,same_cpu_cuda",
        help=f"Comma-separated list of torch op types, choises are {OpKind.choices()}",
    )
    args = parser.parse_args()

    kinds = args.kinds.split(",")
    for i, old_k in enumerate(kinds):
        new_k = old_k.strip()
        if new_k not in OpKind.choices():
            logger.error(
                f"Invalid torch op kind `{old_k}`, must be one of {OpKind.choices()}"
            )
            exit()
        kinds[i] = OpKind.parse(new_k)
    setattr(args, "kinds", kinds)

    return args


def group_by_overload(torch_ops):
    overload_groups = OrderedDict()
    unstructured_list = []

    tmp = OrderedDict()
    for op in torch_ops:
        op_name = OperatorName.parse(op["func"])
        base_name = op_name.name.base
        group = tmp.get(base_name, [])
        group.append(op)
        tmp[base_name] = group

    for base_name, group in tmp.items():
        if len(group) > 1:
            overload_groups[base_name] = group
        else:
            unstructured_list.append(group[0])

    return overload_groups, unstructured_list


def process_overload(overload_groups):
    structured_list = []
    unstructured_extra = []

    tmp = OrderedDict()
    for group in overload_groups.values():
        for op in group:
            op_name = op["func"]
            structured = op.get("structured", False)
            structured_delegate = op.get("structured_delegate", None)

            if structured or structured_delegate:
                if structured:
                    key = op_name
                else:
                    key = structured_delegate.strip()
                overloads = tmp.get(key, [])
                overloads.append(op)
                tmp[key] = overloads
            else:
                unstructured_extra.append(op)
    structured_list = list(tmp.values())
    return structured_list, unstructured_extra


def parse_kind(op):
    flatten_dispatch = op.get("dispatch", {})
    have_cuda = DispatchKey.CUDA in flatten_dispatch
    have_cpu = DispatchKey.CPU in flatten_dispatch
    have_comp_exp = DispatchKey.CompositeExplicitAutograd in flatten_dispatch
    have_comp_imp = DispatchKey.CompositeImplicitAutograd in flatten_dispatch

    kind = OpKind.OTHERS

    if have_cuda:
        if have_cpu:
            if flatten_dispatch[DispatchKey.CPU] == flatten_dispatch[DispatchKey.CUDA]:
                kind = OpKind.SAME_CPU_CUDA
            else:
                kind = OpKind.CUDA_CUSTOM
        else:
            ret = OpKind.CUDA_CUSTOM
    else:
        if have_cpu:
            kind = OpKind.ONLY_CPU
        elif have_comp_exp or have_comp_imp:
            kind = OpKind.ONLY_COMPOSITE
        else:
            kind = OpKind.OTHERS

    return kind


def process_structured(structured_list, musa_ops_set, args):
    kind_groups = OrderedDict()
    for group in structured_list:
        assert len(group) > 1
        prim = None
        for op in group:
            if op.get("structured", False):
                prim = op
                break
        assert prim is not None

        kind = parse_kind(prim)
        groups = kind_groups.get(kind, [])
        groups.append(group)
        kind_groups[kind] = groups

    logger.info("=========== Structured info ===========")
    for kind, groups in kind_groups.items():
        logger.info(f"{kind}: {len(groups)} groups")

    columns = ["functions", "binding"]
    for kind, groups in kind_groups.items():
        if kind not in args.kinds:
            continue
        if not kind.is_exportable():
            logger.warning(
                f"Ignore `{kind}` ops, since they are not important for torch_musa."
            )
            continue
        topic = f"Structured_{kind}"
        records = []
        for group in groups:
            names = ""
            impl = None
            record = {}
            miss = False
            for op in group:
                op_name = op["func"]
                if names != "":
                    names += "\n"
                names += op_name
                if op_name not in musa_ops_set:
                    names += "(missing)"
                    miss = True
                if op.get("structured", False):
                    impl = kind.binding_impl(op["dispatch"])
            assert isinstance(impl, str) and impl != ""
            if miss:
                record["functions"] = names
                record["binding"] = impl
                records.append(record)
        export_csv(this_dir(), topic, columns, records)


def process_unstructured(unstructured_list, musa_ops_set, args):
    kind_groups = OrderedDict()
    for op in unstructured_list:
        kind = parse_kind(op)
        groups = kind_groups.get(kind, [])
        groups.append(op)
        kind_groups[kind] = groups

    logger.info("=========== Unstructured info ===========")
    for kind, groups in kind_groups.items():
        logger.info(f"{kind}: {len(groups)} groups")

    columns = ["functions", "binding"]
    for kind, groups in kind_groups.items():
        if kind not in args.kinds:
            continue
        if not kind.is_exportable():
            logger.warning(
                f"Ignore `{kind}` ops, since they are not important for torch_musa."
            )
            continue
        topic = f"Unstructured_{kind}"
        records = []
        for op in groups:
            op_name = op["func"]
            if op_name in musa_ops_set:
                continue
            impl = kind.binding_impl(op["dispatch"])
            assert isinstance(impl, str) and impl != ""
            record = {
                "functions": op_name,
                "binding": impl,
            }
            records.append(record)
        export_csv(this_dir(), topic, columns, records)


def main(args):
    assert isinstance(args.kinds, list)
    torch_ops_list = get_torch_ops()
    logger.info(f"Total torch ops: {len(torch_ops_list)}")
    musa_ops_set = get_musa_ops()
    logger.info(f"Total musa ops: {len(musa_ops_set)}")

    torch_ops_list = clean_torch_ops(torch_ops_list)
    logger.info(
        f"After cleaning, there are {len(torch_ops_list)} torch ops left to be further classified."
    )
    num_ops = len(torch_ops_list)
    overload_groups, unstructured_list = group_by_overload(torch_ops_list)
    structured_list, unstructured_extra = process_overload(overload_groups)
    unstructured_list += unstructured_extra

    structured_ops = 0
    for lst in structured_list:
        structured_ops += len(lst)
    unstructured_ops = len(unstructured_list)
    assert structured_ops + unstructured_ops == num_ops
    logger.info(
        f"Total {len(structured_list)} structured groups with {structured_ops} ops."
    )
    logger.info(f"Total {unstructured_ops} unstructured ops.")

    process_structured(structured_list, musa_ops_set, args)
    process_unstructured(unstructured_list, musa_ops_set, args)


if __name__ == "__main__":
    args = get_args()
    main(args)
