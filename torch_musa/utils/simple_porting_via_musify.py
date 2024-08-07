"""simple porting tool for porting cuda files"""

import argparse
import glob
import json
import logging
import pathlib
import shutil
from os import rename, makedirs
from os.path import realpath, join, split, isfile, exists
from typing import List, Dict

from .logger_util import LOGGER
from .musify_text import init_ac_automaton, transform_file

EXT_REPLACED_MAPPING = {"cuh": "muh", "cu": "mu"}
SEFL_PATH = str(pathlib.Path(__file__).parent)


def read_json(json_file_path: str) -> dict:
    """Read json data"""
    with open(json_file_path, encoding="utf-8") as f:
        return json.load(f)


class SimplePortingViaMusify:
    """Simple porting tool, integrate with musify-text"""

    def __init__(
        self,
        cuda_dir_path: str,
        ignore_patterns: List[str],
        extra_mapping: Dict[str, str] = None,
        drop_default_mapping: bool = False,
        mapping: List[str] = None,
        output_method: str = "inplace",
        direction: str = "c2m",
        log_level: str = "INFO",
    ):
        self.cuda_dir_path = cuda_dir_path
        cuda_dirname = split(self.cuda_dir_path)[-1]
        self.musa_dir_path = realpath(
            join(join(self.cuda_dir_path, ".."), cuda_dirname + "_musa")
        )

        self.args = argparse.Namespace()
        if isinstance(extra_mapping, str):
            try:
                extra_mapping = json.loads(extra_mapping)
            except RuntimeError as exception:
                LOGGER.warning("%s. Got %s", exception, extra_mapping)

        if not isinstance(extra_mapping, dict):
            extra_mapping = {}

        self.args.extra_mapping = extra_mapping
        self.args.output_method = output_method
        self.args.direction = direction
        self.args.log_level = log_level
        self.ignore_patterns = ignore_patterns if ignore_patterns else []
        self.args.mapping = mapping
        if drop_default_mapping:
            self.args.mapping = []
        self._delete_srcs()
        self._copy_srcs()
        self._collect_srcs()

    def _delete_srcs(self):
        if exists(self.musa_dir_path):
            shutil.rmtree(self.musa_dir_path)
            LOGGER.info("Finish deleting all previous ported sources.")
        makedirs(self.musa_dir_path)

    def _copy_srcs(self):
        shutil.copytree(
            self.cuda_dir_path,
            self.musa_dir_path,
            ignore=shutil.ignore_patterns(*self.ignore_patterns),
            dirs_exist_ok=True,
        )
        LOGGER.info(
            "Finish copying sources from %s to %s and ignore %s.",
            self.cuda_dir_path,
            self.musa_dir_path,
            self.ignore_patterns,
        )

    def _collect_srcs(self):
        all_files = glob.glob(join(self.musa_dir_path, "**", "*"), recursive=True)
        self.args.srcs = [file for file in all_files if isfile(file)]
        LOGGER.info("Finish collecting all sources.")

    def _rename_ext(self):
        for file in self.args.srcs:
            rename(file, self._change_file_ext(file))
        LOGGER.info("Finish renaming all sources.")

    def _change_file_ext(self, name) -> str:
        """Change file extension"""
        name_splits = name.split(".")
        filename = ".".join(name_splits[:-1])
        ext = name_splits[-1]
        if not filename:
            return EXT_REPLACED_MAPPING.get(ext, ext)
        return ".".join([filename, EXT_REPLACED_MAPPING.get(ext, ext)])

    def run(self):
        """Start porting cuda files"""
        logging.basicConfig(
            format="[%(levelname)s] [%(asctime)s] %(message)s",
            level=self.args.log_level,
        )
        logging.debug("Options: %s", self.args)
        init_ac_automaton(self.args)

        for src in self.args.srcs:
            transform_file(src, self.args)
        LOGGER.info("Finish porting all sources.")

        self._rename_ext()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Porting 2")
    parser.add_argument(
        "--cuda-dir-path",
        help="Specify cuda directory to port, " "e.g. /home/mmcv/mmcv/ops/csrc",
        default="/home/mmcv/mmcv/ops/csrc",
    )
    parser.add_argument(
        "--ignore-patterns",
        help="Specify cuda directory to port, "
        'e.g. ["*parrots*", "*mlu*", "*npu*", '
        '"*mps*"]',
        default=["*parrots*", "*mlu*", "*npu*", "*mps*"],
    )
    parser.add_argument(
        "--extra-mapping",
        help="Specify mapping rule" 'e.g. {"cuda":"musa"}',
        default={},
    )
    parser.add_argument(
        "--drop-default-mapping",
        action="store_true",
        help="Specify whether to drop default mapping",
    )
    parser.add_argument(
        "--mapping",
        help="Specify where extra mapping files locate" "e.g. ['mapping.json']",
        default=glob.glob(join(SEFL_PATH, "mapping", "*.json")),
    )
    parser.add_argument(
        "--output-method",
        help="Specify output method. e.g. ",
        choices=["terminal", "create", "inplace"],
        default="inplace",
    )
    parser.add_argument(
        "--direction", help="convert direction", choices=["c2m", "m2c"], default="c2m"
    )
    parser.add_argument(
        "--log-level",
        help="lowest log level to display",
        choices=["DEBUG", "INFO", "WARNING"],
        default="INFO",
    )
    args = parser.parse_args()
    print(vars(args))
    simple_porting_test = SimplePortingViaMusify(
        args.cuda_dir_path,
        args.ignore_patterns,
        args.extra_mapping,
        args.drop_default_mapping,
        args.mapping,
    )
    simple_porting_test.run()
