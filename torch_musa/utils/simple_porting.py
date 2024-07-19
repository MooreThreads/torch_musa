"""simple porting tool for porting cuda files"""

import argparse
import json
import os.path
import shutil
import sys
from os import makedirs
from os.path import realpath, dirname, join, split, exists, relpath
from typing import NoReturn

from .logger_util import LOGGER

EXT_REPLACED_MAPPING = {"cuh": "muh", "cu": "mu"}


def read_json(json_file_path: str) -> dict:
    """Read json data"""
    with open(json_file_path, encoding="utf-8") as f:
        return json.load(f)


class SimplePorting:
    """Simple porting tool, transform cuda files to musa files"""

    def __init__(
        self,
        cuda_dir_path: str = None,
        ignore_dir_paths: list = None,
        mapping_rule: dict = None,
        drop_default_mapping: bool = False,
        mapping_dir_path: str = None,
    ):
        self.overwrite_default_mapping = drop_default_mapping
        self.mapping_rule = mapping_rule
        self.mapping_dir_path = mapping_dir_path
        self.cuda_dir_path = cuda_dir_path
        self.ignore_dir_paths = ignore_dir_paths

        if not self.cuda_dir_path:
            self.cuda_dir_path = realpath(join(dirname(__file__), "cuda"))

        cuda_dirname = split(self.cuda_dir_path)[-1]
        self.musa_dir_path = realpath(
            join(join(self.cuda_dir_path, ".."), cuda_dirname + "_musa")
        )
        if exists(self.musa_dir_path):
            shutil.rmtree(self.musa_dir_path)
        makedirs(self.musa_dir_path)

    def load_replaced_mapping(self) -> NoReturn:
        """Load all mapping files"""
        if self.mapping_rule and isinstance(self.mapping_rule, str):
            try:
                self.mapping_rule = json.loads(self.mapping_rule)
            except RuntimeError:
                LOGGER.warning("json loads mapping_rule failed. %s", self.mapping_rule)

        if not self.mapping_rule:
            self.mapping_rule = {}

        if self.overwrite_default_mapping:
            if not (self.mapping_rule and isinstance(self.mapping_rule, dict)):
                LOGGER.error(
                    "set overwrite_default_mapping True but not give valid mapping_rule. "
                    "got mapping_rule=%s",
                    self.mapping_rule,
                )
                sys.exit(-1)
        else:
            if not self.mapping_dir_path:
                self.mapping_dir_path = realpath(join(dirname(__file__), "mapping"))
            for name in os.listdir(self.mapping_dir_path):
                if name.endswith(".json") and name != "general.json":
                    self.mapping_rule.update(
                        read_json(join(self.mapping_dir_path, name))
                    )
                    LOGGER.info("loading %s...", name)
        self.mapping_rule = sorted(
            self.mapping_rule.items(), key=lambda x: len(x[0]), reverse=True
        )
        print(self.mapping_rule)
        LOGGER.info("Loading all mapping files success.")

    def modify_file(self, cuda_filepath: str, musa_filepath: str) -> NoReturn:
        """Modify file via mapping files"""
        with open(cuda_filepath, encoding="utf-8") as f:
            with open(musa_filepath, "w", encoding="utf-8") as f_musa:
                for line in f.readlines():
                    if line.startswith("*") or line.startswith("/") or line == "":
                        f_musa.write(line)
                        continue
                    for k, v in self.mapping_rule:
                        # header files in cub library are suffixed with ".cuh" instead of ".muh",
                        # which is not consistent with other musa libraries. So here we need to skip
                        # header files replacement of cub library.
                        if "cub/" not in line:
                            line = line.replace(k, v)
                    f_musa.write(line)

    def change_filename(self, name) -> str:
        """Change filename to musa related file."""
        name_splits = name.split(".")
        filename = ".".join(name_splits[:-1])
        ext = name_splits[-1]
        replaced_filename = filename
        return replaced_filename + "." + EXT_REPLACED_MAPPING.get(ext, ext)

    def process_ignore_dir_paths(self):
        """Process ignore dirs"""
        if self.ignore_dir_paths is None:
            self.ignore_dir_paths = []

        if self.ignore_dir_paths and isinstance(self.ignore_dir_paths, str):
            try:
                self.ignore_dir_paths = json.loads(self.ignore_dir_paths)
            except RuntimeError:
                LOGGER.warning(
                    "json loads ignore_dir_paths failed. %s", self.ignore_dir_paths
                )

        self.ignore_dir_paths = [realpath(path) for path in self.ignore_dir_paths]
        LOGGER.info("Processing ignore dir paths success.")

    def run(self):
        """Start porting cuda files to musa"""
        # load mapping rule
        self.load_replaced_mapping()
        # process ignore dirs
        self.process_ignore_dir_paths()
        # process files
        path_iter = list(os.walk(self.cuda_dir_path))
        total_len = len(path_iter)
        for i, (root, _, files) in enumerate(path_iter):
            if realpath(root) in self.ignore_dir_paths:
                LOGGER.info("%s is in no need of porting and skip it.", realpath(root))
                continue
            for file in files:
                musa_file = self.change_filename(file)
                cuda_filepath = realpath(join(root, file))
                rel_path = relpath(root, self.cuda_dir_path)
                dir_path = join(self.musa_dir_path, rel_path)
                makedirs(dir_path, exist_ok=True)
                musa_filepath = realpath(join(dir_path, musa_file))
                self.modify_file(cuda_filepath, musa_filepath)
                LOGGER.info(
                    "modify %s -> %s %s/%s content success.",
                    cuda_filepath,
                    musa_filepath,
                    i + 1,
                    total_len,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Porting")
    parser.add_argument(
        "--cuda-dir-path",
        help="Specify cuda directory to port, "
        "e.g. /home/vision/torchvision/csrc/ops/cuda",
        default="cuda/",
    )
    parser.add_argument(
        "--ignore-dir-paths",
        help="Specify cuda directory to port, "
        "e.g. /home/vision/torchvision/csrc/ops/cuda",
        default=[],
    )
    parser.add_argument(
        "--mapping-rule",
        help="Specify mapping rule" 'e.g. {"cuda":"musa"}',
        default="{}",
    )
    parser.add_argument(
        "--drop-default-mapping",
        action="store_true",
        help="Specify whether to drop default mapping",
    )
    parser.add_argument(
        "--mapping-dir-path",
        help="Specify where mapping directory locate" "e.g. mapping/",
        default=None,
    )
    args = parser.parse_args()
    print(vars(args))
    simple_porting_test = SimplePorting(
        args.cuda_dir_path,
        args.ignore_dir_paths,
        args.mapping_rule,
        args.drop_default_mapping,
        args.mapping_dir_path,
    )
    simple_porting_test.run()
