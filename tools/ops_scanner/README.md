# Ops scanner

## Files manifest
- ops_scanner_base.py
- scan_cuda_full_ops.py
- scan_musa_full_ops.py
- scan_mudnn_support_ops.py
- check_cuda_ops_in_mudnn.py
- parse_musa_functions.py
- README.md

## Usage
Please execute command under **torch_musa root** directory.

1.scan musa ops:

```
python -m tools.ops_scanner.scan_musa_full_ops  \
--scan-dir /home/torch_musa/torch_musa/csrc/aten/ops  \
--file-ext .cpp  \
--regex 'm\.impl\(\s*?TORCH_SELECTIVE_NAME\(\s*?\"(.*?)\"|m\.impl\(\s*?\"(.*?)\"|ADVANCED_REGISTER\(\s*?aten,\s*?PrivateUse1,\s*?\"(.*?)\",'  \
--output-path ./torch_musa_full_ops.xlsx
```

2.scan cuda ops:

```
python -m tools.ops_scanner.scan_cuda_full_ops  \
--scan-dir ./  \
--file-ext .cu  \
--regex 'REGISTER_DISPATCH\((.*?)_stub,|#include <ATen/ops/(.*?)_native.h>|\"(.*?)_cuda\"|REGISTER_CUDA_DISPATCH\((.*?)_stub,'  \
--output-path ./cuda_full_ops.xlsx
```

3.scan mudnn ops:

```
python -m tools.ops_scanner.scan_mudnn_support_ops  \
--scan-dir ./  \
--file-ext .md  \
--regex 'MUDNN_ITEM\((.*?)\)'  \
--output-path ./mudnn_support_ops.xlsx
```

4.check cuda ops in mudnn

```
python -m tools.ops_scanner.check_cuda_ops_in_mudnn  \
--cuda-ops-file-path ./cuda_full_ops.xlsx  \
--mudnn-ops-file-path ./mudnn_support_ops.xlsx  \
--output-path ./check_cuda_ops_in_mudnn.xlsx
```

5.parse musa_functions.yaml to collect all ops

```
python tools/ops_scanner/parse_musa_functions.py
```


## Extension

### Ops scan
1.Create a new class derived from **OpsScannerBase** in ops_scanner_base.py

2.Override method **_get_op_name** if you have customized requirements

### Ops check
1.modify method **match** in check_cuda_ops_in_mudnn.py