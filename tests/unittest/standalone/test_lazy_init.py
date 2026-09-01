"""test lazy init behavior"""

import gc


import torch
import torch_musa  # pylint: disable=unused-import


class TestLazyInit:
    """class of TestLazyInit"""

    def setup_method(self):
        pass

    def teardown_method(self):
        gc.collect()

    def test_lazy_init_flag(self):
        """`torch_musa._initialized` must track the real lazy-init state.

        PyTorch reads the flag off the backend module as
        `getattr(device_module, "_initialized", False)` -- `torch.utils.checkpoint`
        gates device-RNG save/restore on it -- so a stale re-exported copy silently
        disables features instead of failing loudly.
        """
        assert not torch_musa.core._lazy_init.is_initialized()
        x = torch.randn((8,), device="musa")
        assert torch_musa.core._lazy_init.is_initialized()
        assert torch_musa._initialized
        assert torch.musa._initialized

        del x

    def test_manual_seed(self):
        """test torch_musa's manual seed"""
        torch.musa.manual_seed(42)
        x_0 = torch.randn((1024,), device="musa")
        x_1 = torch.randn((1024,), device="musa")
        torch.musa.manual_seed(42)
        x_2 = torch.randn((1024,), device="musa")
        x_3 = torch.randn((1024,), device="musa")

        assert torch.allclose(x_0, x_2)
        assert torch.allclose(x_1, x_3)

        del x_0, x_1, x_2, x_3
