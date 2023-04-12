import argparse
import os
import re

from tools.ops_scanner import ops_scanner_base


class MudnnSupportOpsScanner(ops_scanner_base.OpsScannerBase):
    def _get_op_name(self, file_path: str):
        file_name = os.path.split(file_path)[-1]
        with open(file_path) as f:
            for line in f.readlines():
                if line.strip().startswith("//"):
                    continue
                match_res: list = re.findall(self.ops_regex, line)
                if match_res:
                    self.res[file_name].extend(match_res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Scan ops in PyTorch")
    parser.add_argument("--scan-dir", help="Specify directory to be scanned", default="./")
    parser.add_argument("--file-ext", help="Specify file extension", default=".md")
    parser.add_argument("--regex", help="Specify search regular expression",
                        default=r"MUDNN_ITEM\((.*?)\)"
                        )
    parser.add_argument("--output-path", help="Specify ops' scanning result location",
                        default="./mudnn_support_ops.xlsx")

    args = parser.parse_args()
    test_dir = args.scan_dir
    test_file_ext = args.file_ext
    test_ops_regex = args.regex
    ops_scanner = MudnnSupportOpsScanner(test_dir, test_file_ext, test_ops_regex, output_path=args.output_path)
    ops_scanner.scan()
    ops_scanner.write_to_xlsx()
