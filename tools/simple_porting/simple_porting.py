import argparse
import json
import os.path
from os import makedirs
from os.path import realpath, dirname, join, split, exists
from typing import NoReturn

from tools.simple_porting.logger_util import LOGGER

EXT_REPLACED_MAPPING = {"cuh": "muh", "cu": "mu"}


def read_json(json_file_path: str) -> dict:
    """Read json data"""
    with open(json_file_path) as f:
        return json.load(f)


class SimplePorting:
    """Simple porting tool, transform cuda files to musa files"""

    def __init__(self, cuda_dir_path: str = None, mapping_rule: dict = None,
                 drop_default_mapping: bool = False, mapping_dir_path: str = None):
        self.overwrite_default_mapping = drop_default_mapping
        self.mapping_rule = mapping_rule
        self.mapping_dir_path = mapping_dir_path
        self.cuda_dir_path = cuda_dir_path
        if not self.cuda_dir_path:
            self.cuda_dir_path = realpath(join(dirname(__file__), "cuda"))
        self.musa_dir_path = realpath(join(join(self.cuda_dir_path, ".."), "musa"))
        if not exists(self.musa_dir_path):
            makedirs(self.musa_dir_path)

    def load_replaced_mapping(self) -> NoReturn:
        """Load all mapping files"""
        if self.mapping_rule and isinstance(self.mapping_rule, str):
            try:
                self.mapping_rule = json.loads(self.mapping_rule)
            except RuntimeError:
                LOGGER.warning(f"json loads mapping_rule failed. {self.mapping_rule}")

        if not self.mapping_rule:
            self.mapping_rule = {}

        if self.overwrite_default_mapping:
            if not (self.mapping_rule and isinstance(self.mapping_rule, dict)):
                LOGGER.error("set overwrite_default_mapping True but not give valid mapping_rule. "
                             f"got mapping_rule={self.mapping_rule}")
                exit(-1)
        else:
            if not self.mapping_dir_path:
                self.mapping_dir_path = realpath(join(dirname(__file__), "mapping"))
            for name in os.listdir(self.mapping_dir_path):
                if name.endswith(".json"):
                    self.mapping_rule.update(read_json(join(self.mapping_dir_path, name)))
                    LOGGER.info(f"loading {name}...")
        self.mapping_rule = sorted(self.mapping_rule.items(), key=lambda x: len(x[0]), reverse=True)
        LOGGER.info("loading all mapping files success.")

    def modify_file(self, filepath: str) -> NoReturn:
        """Modify file via mapping files"""
        with open(filepath) as f:
            musa_filepath = self.get_musa_filepath(filepath)
            with open(musa_filepath, "w") as f_musa:
                for line in f.readlines():
                    if line.startswith("*") or line.startswith("/"):
                        continue
                    for k, v in self.mapping_rule:
                        line = line.replace(k, v)
                    f_musa.write(line)

    def get_musa_filepath(self, cuda_filepath: str) -> str:
        """Get musa filename"""
        name = split(cuda_filepath)[-1]
        filename, ext = name.split(".")

        replaced_filename = filename
        if filename.startswith("cuda_"):
            replaced_filename = filename.replace("cuda_", "musa_")
        if filename.endswith("_cuda"):
            replaced_filename = filename.replace("_cuda", "_musa")
        if "_cuda_" in filename:
            replaced_filename = filename.replace("_cuda_", "_musa_")
        return join(self.musa_dir_path,
                    replaced_filename + "." + EXT_REPLACED_MAPPING.get(ext, ext))

    def run(self):
        # load mapping rule
        self.load_replaced_mapping()
        # process files
        files = os.listdir(self.cuda_dir_path)
        for i, name in enumerate(files):
            full_path = join(self.cuda_dir_path, name)
            self.modify_file(full_path)
            LOGGER.info(f"modify {name} {i + 1}/{len(files)} content success.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple Porting")
    parser.add_argument("--cuda-dir", help="Specify cuda directory to port, "
                                           "e.g. /home/vision/torchvision/csrc/ops/cuda",
                        default="cuda/")
    parser.add_argument("--mapping-rule", help="Specify mapping rule"
                                               "e.g. {\"cuda\":\"musa\"}",
                        default="{}")
    parser.add_argument("--drop-default-mapping", action='store_true',
                        help="Specify whether to drop default mapping")
    parser.add_argument("--mapping-dir-path", help="Specify where mapping directory locate"
                                                   "e.g. mapping/",
                        default=None)
    args = parser.parse_args()
    print(vars(args))
    simple_porting_test = SimplePorting(args.cuda_dir, args.mapping_rule,
                                        args.drop_default_mapping, args.mapping_dir_path)
    simple_porting_test.run()
