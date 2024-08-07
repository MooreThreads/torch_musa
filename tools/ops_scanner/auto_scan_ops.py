from tools.ops_scanner.ops_scanner_base import OpsScannerBase
from tools.ops_scanner.parse_musa_functions import MusaFunctionsParser
from os.path import join, dirname, abspath
from itertools import chain

# scan all sources
scan_dir = join(
    dirname(dirname(dirname(abspath(__file__)))), "torch_musa", "csrc", "aten"
)
# scan all file types
scan_file_ext = ""
# filter ops by m.impl
aten_quantized_block_pattern = (
    r"TORCH_LIBRARY_IMPL\(aten, QuantizedPrivateUse1, m\) \{([\s\S]*?)\}"
)
quantized_quantized_block_pattern = (
    r"TORCH_LIBRARY_IMPL\(quantized, QuantizedPrivateUse1, m\) \{([\s\S]*?)\}"
)
regular_block_pattern = r"TORCH_LIBRARY_IMPL\(aten, PrivateUse1, m\) \{([\s\S]*?)\}"
torchvision_block_pattern = (
    r"TORCH_LIBRARY_IMPL\(torchvision, PrivateUse1, m\) \{([\s\S]*?)\}"
)

impl_pattern = 'm\.impl\(\s*?TORCH_SELECTIVE_NAME\(\s*?"(.*?)"|m\.impl\(\s*?"(.*?)"'

# catch quantized ops
quantized_ops = []
for block_pattern in (aten_quantized_block_pattern, quantized_quantized_block_pattern):
    ops_scanner = OpsScannerBase(
        scan_dir, scan_file_ext, impl_pattern, conditional_regex=block_pattern
    )
    ops_scanner.scan()
    quantized_ops.extend(chain(*ops_scanner.res.values()))

quantized_ops = ["`" + op + "` (quantized op)" for op in quantized_ops]

# catch extra regular and torchvision ops
ops_scanner = OpsScannerBase(
    scan_dir, scan_file_ext, impl_pattern, conditional_regex=regular_block_pattern
)
ops_scanner.scan()
musa_functions_parser = MusaFunctionsParser()
musa_functions_parser.parse_file()
regular_ops = chain(*ops_scanner.res.values(), musa_functions_parser.res)
regular_ops = ["`" + op + "`" for op in regular_ops]

ops_scanner = OpsScannerBase(
    scan_dir, scan_file_ext, impl_pattern, conditional_regex=torchvision_block_pattern
)
ops_scanner.scan()
torchvision_ops = chain(*ops_scanner.res.values())
torchvision_ops = ["`" + op + "` (torchvision op)" for op in torchvision_ops]

# write to markdown
with open(join(dirname(abspath(__file__)), "ops_list.md"), "w", encoding="UTF-8") as f:
    total = len(regular_ops) + len(torchvision_ops) + len(quantized_ops)
    f.write(f"Total amount of ops: {total}\n")
    index = 0
    for op_name in sorted(chain(regular_ops, torchvision_ops, quantized_ops)):
        f.write(f"- {op_name}\n")
        index += 1
    assert index == total, "Ops count not correct!"
    print(f"Total amount of ops: {index}")
