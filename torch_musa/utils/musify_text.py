"""Musify-text tool"""

import contextlib
import itertools
import json
import logging
import os
import sys

from ahocorapy.keywordtree import KeywordTree


def is_overlap(x, y):
    """Judge if two string overlap"""
    x_name, x_start = x[0]
    y_name, y_start = y[0]
    x_len = len(x_name)
    y_len = len(y_name)
    x_end = x_start + x_len
    y_end = y_start + y_len
    return x_start < y_end and y_start < x_end


def should_replace(old, new):
    """Judge if string should replace"""
    o_name, o_start = old[0]
    n_name, n_start = new[0]
    if o_start > n_start:
        return True
    if o_start < n_start:
        return False

    o_len = len(o_name)
    n_len = len(n_name)

    return o_len < n_len


class KeywordMap:
    """
    Class definition of keyword map.
    """

    def __init__(self, case_insensitive=False):
        self.__inner = KeywordTree(case_insensitive=case_insensitive)
        self.__valuemap = {}

    def add(self, key, value):
        """Add new key-value pair to map"""
        self.__inner.add(key)
        self.__valuemap[key] = value

    def __transform_result(self, result):
        """Transform result"""
        key, _ = result
        return result, self.__valuemap[key]

    def search(self, text):
        """Search text"""
        return self.search_one(text)

    def search_one(self, text):
        """Search text once"""
        result = self.__inner.search_one(text)
        if result is None:
            return None

        return self.__transform_result(result)

    def search_all(self, text):
        """Search all text occurrence"""
        return map(self.__transform_result, self.__inner.search_all(text))

    def search_longest(self, text):
        """Search the longest text occurrence"""
        result_gen = self.search_all(text)
        try:
            candidate = next(result_gen)
        except StopIteration:
            return

        for result in result_gen:
            if not is_overlap(candidate, result):
                yield candidate
                candidate = result
            elif should_replace(candidate, result):
                candidate = result

        yield candidate

    def finalize(self):
        """Finalize"""
        self.__inner.finalize()


automaton = KeywordMap()

EXCL_FLAG = False


def exclusion_start(line_no):
    """Judge if the line at line_no is exclusion start"""
    global EXCL_FLAG
    if EXCL_FLAG:
        logging.warning(
            "open musify exclusion block when it is already opened at line %s", line_no
        )
    else:
        logging.info("musify exclusion block opened at line %s", line_no)
    EXCL_FLAG = True
    return False


def exclusion_stop(line_no):
    """Judge if the line at line_no is exclusion stop"""
    global EXCL_FLAG
    if not EXCL_FLAG:
        logging.warning(
            "close musify exclusion block when it is already closed at line %s", line_no
        )
    else:
        logging.info("musify exclusion block closed at line %s", line_no)
    EXCL_FLAG = False
    return False


def exclusion_line(line_no):
    """Judge if the line at line_no is exclusion line"""
    if EXCL_FLAG:
        logging.warning(
            "use musify single line exclusion when in exclusion block at line %s",
            line_no,
        )
    else:
        logging.info("musify single line exclusion at line %s", line_no)
    return True


excl_pattern = KeywordMap()
excl_pattern.add(b"MUSIFY_EXCL_START", exclusion_start)
excl_pattern.add(b"MUSIFY_EXCL_STOP", exclusion_stop)
excl_pattern.add(b"MUSIFY_EXCL_LINE", exclusion_line)
excl_pattern.finalize()


def init_ac_automaton(args):
    """Ac automaton initialization function"""

    def read_mapping(path):
        """Load mapping via json load"""
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    map_iter = map(
        lambda tup: (tup[0].encode(), tup[1].encode()),
        itertools.chain(*map(lambda p: read_mapping(p).items(), args.mapping)),
    )

    extra_map_iter = []
    if args.extra_mapping and isinstance(args.extra_mapping, dict):
        extra_map_iter = map(
            lambda tup: (tup[0].encode(), tup[1].encode()), args.extra_mapping.items()
        )
    if args.direction == "m2c":
        for cuda, musa in itertools.chain(map_iter, extra_map_iter):
            automaton.add(musa, cuda)
    else:
        for cuda, musa in itertools.chain(map_iter, extra_map_iter):
            automaton.add(cuda, musa)

    automaton.finalize()


def is_word_char(char_str):
    """Judge if ch belongs to a word"""
    return char_str.isalnum() or char_str == b"_"


def is_word_boundary(code, begin_idx, end_idx):
    """Judge if it is word boundary"""
    left_has_more = begin_idx > 0 and is_word_char(bytes([code[begin_idx - 1]]))
    right_has_more = end_idx < len(code) and is_word_char(bytes([code[end_idx]]))
    return not (left_has_more or right_has_more)


def transform_line(line, line_no):
    """Transform line"""
    for _, excl_fn in excl_pattern.search_all(line):
        if excl_fn(line_no):
            return line

    if EXCL_FLAG:
        return line

    new_line = b""
    last_end_idx = 0
    for (src_name, begin_idx), dst_name in automaton.search_longest(line):
        end_idx = begin_idx + len(src_name)
        if is_word_boundary(line, begin_idx, end_idx):
            new_line += line[last_end_idx:begin_idx]
            new_line += dst_name
            last_end_idx = end_idx

    new_line += line[last_end_idx:]

    return new_line


@contextlib.contextmanager
def writer(file_name=None):
    """File writer"""
    if file_name:
        file_writer = open(file_name, "wb")
    else:
        file_writer = sys.stdout
    yield file_writer
    if file_name:
        file_writer.close()


def transform_file(path, args):
    """Transform file"""
    logging.info("Processing %s", path)
    write_path = None
    if args.output_method != "terminal":
        write_path = path + ".mt"
    with open(path, "rb") as read_handle:
        with writer(write_path) as write_handle:
            lines = read_handle.readlines()
            for line_no, line in enumerate(lines, 1):
                new_line = transform_line(line, line_no)
                write_handle.write(new_line)
    if args.output_method == "inplace":
        os.rename(write_path, path)

    global EXCL_FLAG
    if EXCL_FLAG:
        logging.warning("musify exclusion block not closed when file ends.")
        EXCL_FLAG = False
