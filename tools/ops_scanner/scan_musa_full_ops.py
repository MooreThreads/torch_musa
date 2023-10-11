import argparse
import os
import re
from itertools import chain
  
from tools.ops_scanner import ops_scanner_base
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Scan ops in PyTorch")
    parser.add_argument("--scan-dir", help="Specify directory to be scanned", default="/home/torch_musa/torch_musa/csrc/aten/ops")
    parser.add_argument("--file-ext", help="Specify file extension", default=".cpp")
    parser.add_argument("--regex", help="Specify search regular expression",
                        default=r"m\.impl\(\"(.*?)\"|ADVANCED_REGISTER\(\s*?aten,\s*?PrivateUse1,\s*?\"(.*?)\","
                        )
    parser.add_argument("--output-path", help="Specify ops' scanning result location", default="./torch_musa_full_ops.xlsx")

    args = parser.parse_args()
    test_dir = args.scan_dir
    test_file_ext = args.file_ext
    test_ops_regex = args.regex
    ops_scanner = ops_scanner_base.OpsScannerBase(test_dir, test_file_ext, test_ops_regex, output_path=args.output_path)
    ops_scanner.scan()
    ops_scanner.write_to_xlsx()
