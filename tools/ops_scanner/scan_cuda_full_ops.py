import argparse
import os
import re
from itertools import chain

from tools.ops_scanner import ops_scanner_base


class CudaFullOpsScanner(ops_scanner_base.OpsScannerBase):
    def _get_op_name(self, file_path: str):
        file_name = os.path.split(file_path)[-1]
        with open(file_path) as f:
            for line in f.readlines():
                match_res: list = re.findall(self.ops_regex, line)
                if not match_res:
                    continue
                # remove duplicate
                original_ops_set = set(chain(*match_res))
                final_res = set()
                for op in original_ops_set:
                    # remove some derived ops like 'out' or 'nhwc'
                    op_: str = op.replace("_out", "").replace("_nhwc", "")
                    # and remove leading '_'
                    if op_.startswith("_"):
                        op_ = op_[1:]
                    # not empty
                    if op_:
                        final_res.add(op_)
                if final_res:
                    self.res[file_name].extend(final_res)
        # whole file level: remove duplicate ops
        self.res[file_name] = list(set(self.res[file_name]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Scan ops in PyTorch")
    parser.add_argument("--scan-dir", help="Specify directory to be scanned", default="./")
    parser.add_argument("--file-ext", help="Specify file extension", default=".cu")
    parser.add_argument("--regex", help="Specify search regular expression",
                        default=r"REGISTER_DISPATCH\((.*?)_stub,"
                                r"|#include <ATen/ops/(.*?)_native.h>"
                                r"|\"(.*?)_cuda\""
                                r"|REGISTER_CUDA_DISPATCH\((.*?)_stub,"
                        )
    parser.add_argument("--output-path", help="Specify ops' scanning result location", default="./cuda_full_ops.xlsx")

    args = parser.parse_args()
    test_dir = args.scan_dir
    test_file_ext = args.file_ext
    test_ops_regex = args.regex
    ops_scanner = CudaFullOpsScanner(test_dir, test_file_ext, test_ops_regex, output_path=args.output_path)
    ops_scanner.scan()
    ops_scanner.write_to_xlsx()
