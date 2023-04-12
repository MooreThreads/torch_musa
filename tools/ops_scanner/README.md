# Ops scanner

## Files manifest
- ops_scanner_base.py
- scan_cuda_full_ops.py
- scan_mudnn_support_ops.py
- check_cuda_ops_in_mudnn.py
- README.md

## Usage
Please execute command under **torch_musa root** directory.

1.scan cuda ops:

```
python -m tools.ops_scanner.scan_cuda_full_ops  \
--scan-dir ./  \
--file-ext .cu  \
--regex 'REGISTER_DISPATCH\((.*?)_stub,|#include <ATen/ops/(.*?)_native.h>|\"(.*?)_cuda\"|REGISTER_CUDA_DISPATCH\((.*?)_stub,'  \
--output-path ./cuda_full_ops.xlsx
```

2.scan mudnn ops:

```
python -m tools.ops_scanner.scan_mudnn_support_ops  \
--scan-dir ./  \
--file-ext .md  \
--regex 'MUDNN_ITEM\((.*?)\)'  \
--output-path ./mudnn_support_ops.xlsx
```

3.check cuda ops in mudnn

```
python -m tools.ops_scanner.check_cuda_ops_in_mudnn  \
--cuda-ops-file-path ./cuda_full_ops.xlsx  \
--mudnn-ops-file-path ./mudnn_support_ops.xlsx  \
--output-path ./check_cuda_ops_in_mudnn.xlsx
```

## Extension

### Ops scan
1.Create a new class derived from **OpsScannerBase** in ops_scanner_base.py

2.Override method **_get_op_name** if you have customized requirements

### Ops check
1.modify method **match** in check_cuda_ops_in_mudnn.py