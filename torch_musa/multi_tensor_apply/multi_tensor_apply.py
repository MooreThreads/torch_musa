# pylint: disable=missing-class-docstring
"""
A multi-tensor apply launch that batches the elementwise updates applied to
all the model's parameters into one or a few kernel launches
"""
from ..utils import ext_loader


class MultiTensorApply(object):
    available = False
    warned = False

    def __init__(self, chunk_size):
        try:
            ext_loader.load_ext("_ext")
            MultiTensorApply.available = True
            self.chunk_size = chunk_size
        except ImportError as err:
            MultiTensorApply.available = False
            MultiTensorApply.import_err = err

    def check_avail(self):
        if not MultiTensorApply.available:
            raise RuntimeError(
                "Attempted to call MultiTensorApply method, but MultiTensorApply "
                "is not available, original import error message:",
                MultiTensorApply.import_err,
            )

    def __call__(self, op, noop_flag_buffer, tensor_lists, *args):
        self.check_avail()

        return op(self.chunk_size, noop_flag_buffer, tensor_lists, *args)
