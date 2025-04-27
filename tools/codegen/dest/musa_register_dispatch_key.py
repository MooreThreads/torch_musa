"""Derived class of RegisterDispatchKey

Up to PT2.2, we have to add some extra patches at the THPVariable_op
level to ensure device is initialized when device related operators are invoked,
i.e., call musa_lazy_init(), use overwrited RegisterDispatchKey to avoid the intrusive way.

Not quite understand why MUSARegisterDispatchKey should be imported inside the function.
"""

import re
from typing import Optional
from torchgen.utils import Target
from torchgen.dest import RegisterDispatchKey
from torchgen.model import NativeFunction, NativeFunctionsGroup

from codegen.model import musa_get_func_extra_info


class MUSARegisterDispatchKey(RegisterDispatchKey):
    # we can remove overwrite function since PT2.5
    def gen_unstructured(
        self, f: NativeFunction, g: Optional[NativeFunctionsGroup] = None
    ) -> Optional[str]:
        base_result = super().gen_unstructured(f, g)
        if self.target is Target.ANONYMOUS_DEFINITION:
            if base_result and musa_get_func_extra_info(f).device_lazy_init:
                match = re.search("\) {", base_result)
                idx = match.start() + 3
                updated_result = (
                    base_result[:idx]
                    + "\n  torch::utils::musa_lazy_init();"
                    + base_result[idx:]
                )

                return updated_result

        return base_result
