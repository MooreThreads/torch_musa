"""A utility for converting training/inference scripts from CUDA to MUSA"""

# pylint: disable=invalid-name, unused-argument, missing-function-docstring

import os
import os.path as osp
import sys
from collections import namedtuple
from argparse import ArgumentParser
import logging
from typing import Dict, List

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm module not found, please install tqdm by `pip install tqdm`")
    sys.exit(1)

try:
    import libcst as cst
    import libcst.matchers as m
except ImportError:
    print(
        "libcst module not found, please install libcst by `pip install libcst==1.1.0`"
    )
    sys.exit(1)


def string2cst_name(name: str) -> cst.Name:
    """cast str to libcst.Name"""
    return cst.Name(name)


def string2cst_attribute(attr: str) -> cst.Attribute:
    """cast str to libcst.Attribute"""
    attr_str_lst: List[str] = attr.split(".")
    assert len(attr_str_lst) >= 2, "length of attr_str_lst must greater than 2"

    def recursive_fn(str_lst):
        if len(str_lst) == 2:
            return cst.Attribute(
                value=string2cst_name(str_lst[0]), attr=string2cst_name(str_lst[1])
            )

        return cst.Attribute(
            value=recursive_fn(str_lst[:-1]), attr=string2cst_name(str_lst[-1])
        )

    return recursive_fn(attr_str_lst)


MAPPING_RULES = namedtuple(
    "MAPPING_RULES", ["NameAndCommentAndSimpleString", "ImportFrom", "Attribute"]
)(
    NameAndCommentAndSimpleString={
        "CUDA": "MUSA",
        "cuda": "musa",
        "NCCL": "MCCL",
        "nccl": "mccl",
        "nvcc": "mcc",
    },
    ImportFrom={
        string2cst_attribute("torch.utils.cpp_extension"): string2cst_attribute(
            "torch_musa.utils.musa_extension"
        ),
    },
    # Attribute node only modified in `ImportFrom` or `Import` scope
    Attribute={
        string2cst_attribute("torch.musa"): string2cst_name("torch_musa"),
        string2cst_attribute("torch.cuda"): string2cst_name("torch_musa"),
    },
)


class MUSAReplacementTransformer(cst.CSTTransformer):
    """
    A Transformer for converting CUDA-related strings and APIs to the MUSA platform.

    This class implement custom logic to convert CUDA-specific code to MUSA compatible code.
    It traverses the Concrete Syntax Tree(CST) of the code and performs the necessary
    transformations, such as replacing CUDA identifiers and function calls with their
    corresponding MUSA counterparts.
    """

    def __init__(self):
        self.in_Import = False
        self.in_ImportFrom = False

    def leave_Comment(
        self, original_node: cst.Comment, updated_node: cst.Comment
    ) -> cst.Comment:
        return self.leave_NameOrCommentOrSimpleString(original_node, updated_node)

    def leave_SimpleString(
        self, original_node: cst.SimpleString, updated_node: cst.SimpleString
    ) -> cst.SimpleString:
        return self.leave_NameOrCommentOrSimpleString(original_node, updated_node)

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:
        return self.leave_NameOrCommentOrSimpleString(original_node, updated_node)

    def leave_FormattedStringText(
        self,
        original_node: cst.FormattedStringText,
        updated_node: cst.FormattedStringText,
    ) -> cst.FormattedStringText:
        return self.leave_NameOrCommentOrSimpleString(original_node, updated_node)

    def leave_NameOrCommentOrSimpleString(self, original_node, updated_node):
        new_str = updated_node.value
        for k, v in MAPPING_RULES.NameAndCommentAndSimpleString.items():
            if k in new_str:
                new_str = new_str.replace(k, v)

        return updated_node.with_changes(value=new_str)

    def visit_Import(self, node: cst.Import) -> bool:
        self.in_Import = True
        return True

    def leave_Import(
        self, original_node: cst.Import, updated_node: cst.Import
    ) -> cst.Import:
        self.in_Import = False
        return updated_node

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
        self.in_ImportFrom = True
        return True

    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> cst.ImportFrom:
        self.in_ImportFrom = False
        rules = getattr(MAPPING_RULES, "ImportFrom")
        assert isinstance(rules, Dict)
        for k in rules:
            v = rules[k]
            if m.matches(getattr(updated_node, "module"), k):
                return updated_node.with_changes(module=v)

        return updated_node

    def leave_Attribute(
        self, original_node: cst.Attribute, updated_node: cst.Attribute
    ) -> cst.Attribute:
        """
        Attribute shrinking, i.e., torch.musa ---> torch_musa

        the modification only happend within the `Import` and `ImportFrom` scope,
        which ensured by flags of `self.in_Import` and `self.in_ImportFrom`
        """
        if self.in_Import or self.in_ImportFrom:
            rules = getattr(MAPPING_RULES, "Attribute")
            assert isinstance(rules, Dict)
            for k in rules:
                v = rules[k]
                if m.matches(updated_node, k):
                    return v

        return updated_node


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def parse_args() -> ArgumentParser:
    """parse the command line options"""
    parser = ArgumentParser(
        description="A utility for converting training/inference scripts from CUDA to MUSA"
    )
    parser.add_argument(
        "-r",
        "--root_path",
        type=str,
        help="root path to scripts to be converted, this could be a path of directory or file",
        required=True,
    )
    parser.add_argument(
        "-l",
        "--launch_path",
        type=str,
        default=None,
        help="launch file path of training/inference scripts, "
        "will add `import torch_musa` if specified",
    )
    parser.add_argument(
        "-e",
        "--excluded_path",
        type=str,
        nargs="*",
        default=[],
        help="path(s) will be excluded during convertion",
    )

    args = parser.parse_args()
    return args


class BaseMUSAConverter:
    """Base class of MUSAConverter"""

    def __init__(self, args: ArgumentParser):
        self.args = args

        self.root_path = args.root_path
        self.launch_file_path = args.launch_path
        if args.launch_path is None:
            logging.warning(
                "the file path of launch script was not provided, you can also add "
                "import torch_musa in this file manually"
            )
        __excluded_path = []
        if len(args.excluded_path):
            for path in args.excluded_path:
                if not osp.isfile(path):
                    logging.error(
                        "the path of %s not exits, "
                        "please check the inputs that passed into  -e/--excluded_path",
                        path,
                    )
                    sys.exit(1)
                __excluded_path.append(osp.abspath(path))
        self.excluded_path = __excluded_path

    def is_legal_file(self, path: str) -> bool:
        assert osp.isabs(path), "absolute path needed here"
        file_suffix = osp.splitext(path)[-1]

        return file_suffix in [".py"] and path not in self.excluded_path

    def setup_files(self) -> List[str]:
        """
        prepare path of files to be converted, absolute path will be used during convertion.
        """
        file_list = []
        if osp.isfile(self.root_path):
            path = osp.abspath(self.root_path)
            if self.is_legal_file(path):
                file_list.append(path)
        elif osp.isdir(self.root_path):
            for root, _, files in os.walk(self.root_path):
                for file in files:
                    file_path = osp.abspath(osp.join(root, file))
                    if self.is_legal_file(file_path):
                        file_list.append(file_path)
        else:
            raise RuntimeError("illegal root_path")

        return file_list

    def run(self):
        """subclass must implement run()"""
        raise NotImplementedError


class AdvancedMUSAConverter(BaseMUSAConverter):
    """
    A CST-based convertion tool for converting CUDA-related strings and APIs to the MUSA platform.
    """

    def __init__(self, args: ArgumentParser):
        super().__init__(args)

        self.visitors = [
            MUSAReplacementTransformer(),
        ]

    def collect_unsupported_torch_ops(self):
        raise NotImplementedError

    def convert_file(self, file: str) -> None:
        """convert file"""
        with open(file, "r+", encoding="utf-8") as f:
            src_code = f.read()
            module = cst.parse_module(src_code)
            for visitor in self.visitors:
                module = module.visit(visitor)
            new_code = module.code
            f.seek(0)
            f.write(new_code)
            f.truncate()

    def add_torch_musa_into_launch_script(self, file: str) -> None:
        # NOTE: sometimes running the script after adding `import torch_musa`
        # at the begining of the .py file might not work as excepted, for example,
        # the ENV of MUSA_VISIBLE_DEVICES set in the script may be invalidated
        with open(file, "r+", encoding="utf-8") as f:
            src_code = f.read()
            module = cst.parse_module(src_code)
            import_statement = cst.SimpleStatementLine(
                (cst.Import([cst.ImportAlias(cst.Name("torch_musa"))]),)
            )
            module_body_list = list(module.body)
            module_body_list.insert(0, import_statement)

            module = module.with_changes(body=module_body_list)
            new_code = module.code
            f.seek(0)
            f.write(new_code)
            f.truncate()

    def run(self):
        """run"""
        logging.info("Start to convert the scripts...")
        file_list = self.setup_files()
        for file in tqdm(file_list):
            self.convert_file(file)

        if self.launch_file_path:
            self.add_torch_musa_into_launch_script(self.launch_file_path)
        logging.info("Scripts conversion done!!!")


def main():
    args = parse_args()
    musa_converter = AdvancedMUSAConverter(args)
    musa_converter.run()


if __name__ == "__main__":
    main()
