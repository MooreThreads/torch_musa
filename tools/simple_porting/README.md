# SimplePorting

Under torch_musa root directory, port cuda files to musa files via

`
python -m tools.simple_porting.simple_porting --cuda-dir cuda/
`

Please refer to simple_porting.py if you want more customizations.

Full command maybe:

`
python -m tools.simple_porting.simple_porting --cuda-dir cuda/ --mapping-rule {"cuda":"musa"} --drop-default-mapping --mapping-dir-path mapping/
`

If under WIN os then {"cuda":"musa"} should be '{\\"cuda\\":\\"musa\\"}'

If you want to integrate it to your own code then can use it like this:

```
import sys
sys.path.append("/home/torch_musa")
from tools.simple_porting.simple_porting import SimplePorting

xxx

SimplePorting(cuda_dir, mapping_rule, drop_default_mapping, mapping_dir_path).run()

xxx
```
