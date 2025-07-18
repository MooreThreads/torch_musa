"""This file is mainly from cuda-porting tool 'Musify'
"""

import os
import re
import json
import pathlib
import argparse
import itertools
import contextlib
import ahocorasick
from typing import Dict


def init_ac_automaton(input_map_files: list) -> ahocorasick.Automaton:
    r"""Returns an instance of Automaton with specific information.

    Args:
        input_map_files (list[str]): json files which describe matching and replacing information.

    Returns:
        an instance of ahocorasick.Automaton.
    """
    automaton = ahocorasick.Automaton()

    def read_mapping(path):
        with open(path) as handle:
            return json.load(handle)

    map_iter = itertools.chain(*map(lambda p: read_mapping(p).items(), input_map_files))
    for cuda, musa in map_iter:
        automaton.add_word(cuda, (len(cuda), musa))

    automaton.make_automaton()
    return automaton


def is_word_char(chr: str) -> bool:
    r"""Checks whether all the characters in a given string are either alphabet/numeric (alphanumeric) characters or "_".

    Args:
        chr (str): a string.

    Returns:
        bool: True If all the characters are alphanumeric or "_".
    """
    return chr.isalnum() or chr == "_"


def is_word_boundary(code: str, start_idx: int, end_idx: int) -> bool:
    r"""Check whether it is a word boundary.

    Args:
        code (str): a code string.
        start_idx (int): start index.
        end_idx (int): end index.

    Returns:
        bool: whether it is a word boundary or not.
    """
    left_has_more = start_idx > 0 and is_word_char(code[start_idx - 1])
    right_has_more = end_idx + 1 < len(code) and is_word_char(code[end_idx + 1])
    return not (left_has_more or right_has_more)


def transform_line(
    file_path: str,
    line: str,
    automaton: ahocorasick.Automaton,
    replace_map: Dict[str, str] = {},
    excluded_files_mapping: Dict[str, tuple] = {},
) -> None:
    r"""Match and replace specific strings for a line in file.

    Args:
        file_path (str): path of file which will be transformed.
        line (str): a line in file which will be transformed.
        automaton (ahocorasick.Automaton): an instance of Automaton with matching and replacing information.
        replace_map (Dict[str, str]): an extra map used to match and replace which could not be handled by Automaton.
        excluded_files_mapping (Dict[str, tuple]): an map used to filter out files that we don't want to apply rules on.

    Returns: None.
    """
    new_line = ""
    last_end_idx = 0
    for end_idx, (cuda_len, musa_name) in automaton.iter_long(line):
        start_idx = end_idx - cuda_len + 1
        if is_word_boundary(line, start_idx, end_idx):
            new_line += line[last_end_idx:start_idx]
            new_line += musa_name
            last_end_idx = end_idx + 1

    new_line += line[last_end_idx:]
    base_filename = os.path.basename(file_path)
    for key, value in replace_map.items():

        # only filter __syncthreads() for now
        if base_filename in excluded_files_mapping.get(key, []):
            continue
        # Note: header files in cub library are suffixed with ".cuh" instead of ".muh",
        # which is not consistent with other musa libraries. So here we need to skip
        # header files replacement of cub library.
        if "cub/" not in new_line:
            pattern = re.compile(key)
            new_line = pattern.sub(value, new_line)

    return new_line


@contextlib.contextmanager
def writer(file_name: str = None) -> None:
    r"""Manage files when writing

    Args:
        file_name (str): file name is being written.

    Returns: None.
    """
    if file_name:
        writer = open(file_name, "w")
    else:
        writer = sys.stdout
    yield writer
    if file_name:
        writer.close()


def transform_file(
    path: str,
    automaton: ahocorasick.Automaton,
    replace_map: Dict[str, str] = {},
    excluded_files_mapping: Dict[str, tuple] = {},
) -> str:
    r"""Match and replace specific strings for cuda compatibility.

    Args:
        path (str): path of file which will be transformed.
        automaton (ahocorasick.Automaton): an instance of Automaton with matching and replacing information.
        replace_map (Dict[str, str]): an extra map used to match and replace which could not be handled by Automaton.
        excluded_files_mapping (Dict[str, tuple]): an map used to filter out files that we don't want to apply rules on.

    Returns:
        file_name (str): new file name after transforming.
    """
    write_path = path + ".mt"
    with open(path, "rb") as read_handle:
        with writer(write_path) as write_handle:
            if (
                path.endswith("cu")
                or path.endswith("cuh")
                or os.path.basename(path)
                in [
                    "DistributionTemplates.h",
                ]
            ) and ("THC" not in path):
                write_handle.write(
                    '#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"\n'
                )
            lines = read_handle.readlines()
            old_line = "null"
            for line in lines:
                try:
                    line = line.decode()
                except UnicodeDecodeError:
                    line = line.decode("unicode_escape")
                if (line == "constexpr DeviceType kCUDA = DeviceType::CUDA;\n") or (
                    "bool is_cuda() const noexcept" in old_line
                ):
                    new_line = line
                else:
                    new_line = transform_line(
                        path, line, automaton, replace_map, excluded_files_mapping
                    )
                write_handle.write(new_line)
                old_line = line
    file_name = os.path.basename(path)
    if (
        "cub/" not in file_name
        and file_name.endswith(".cu")
        or file_name.endswith(".cuh")
        or "CUDA" in file_name
        or "cuda" in file_name
    ):
        musa_file_name = (
            file_name.replace(".cu", ".mu")
            .replace("CUDA", "MUSA")
            .replace("cuda", "musa")
        )
        musa_file_path = path.replace(file_name, musa_file_name)
        os.remove(path)
    else:
        musa_file_path = path
    os.rename(write_path, musa_file_path)
    return musa_file_path
