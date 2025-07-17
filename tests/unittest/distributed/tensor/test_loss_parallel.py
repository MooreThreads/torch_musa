"""Unit test for loss parallel"""

# pylint: disable=C0115,W0237,W0613,C0103,W0221
import torch
import torch.nn.functional as F


from torch.distributed.tensor import distribute_tensor, Shard
from torch.distributed.tensor.parallel import loss_parallel
from torch.distributed.device_mesh import init_device_mesh
from torch.utils._python_dispatch import TorchDispatchMode


from torch_musa.testing.common_fsdp import FSDPTest, skip_if_lt_x_gpu
from torch_musa.distributed.loss import parallel_fused_cross_entropy


NUM_GPUS_FOR_TESTING_LOSS_PARALLEL = 2


@skip_if_lt_x_gpu(NUM_GPUS_FOR_TESTING_LOSS_PARALLEL)
class TestLossParallel(FSDPTest):

    @property
    def world_size(self) -> int:
        return NUM_GPUS_FOR_TESTING_LOSS_PARALLEL

    def _device_mesh_plan(self, device="musa"):
        device_mesh = init_device_mesh(device, (self.world_size,))

        return device_mesh

    def test_cross_entropy_loss_parallel(self):
        """test cross entropy loss parallel logic under MUSA backend"""

        class OpDispatchMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                assert str(func) not in [
                    "aten._fused_cross_entropy_loss_2d_forward.default",
                    "aten._fused_cross_entropy_loss_2d_backward.default",
                ]
                return func(*args, **kwargs or {})

        device = torch.musa.current_device()
        device_mesh = init_device_mesh("musa", (self.world_size,))
        seed = 42
        torch.manual_seed(seed)
        torch.musa.manual_seed(seed)
        torch.musa.manual_seed_all(seed)

        bs, n_classes = 4, 16

        inp = torch.randn(bs, n_classes, device=device, requires_grad=True)
        target = torch.randint(n_classes, (bs,), device=device)

        inp_cpu = inp.detach().cpu()
        inp_cpu.requires_grad_(True)
        target_cpu = target.cpu()

        dist_inp = distribute_tensor(
            inp, device_mesh, placements=[Shard(1)]
        )  # shard at last dim

        with OpDispatchMode(), loss_parallel():
            loss = F.cross_entropy(dist_inp, target, reduction="mean")
            assert loss.device.type == "musa"
            loss.backward()

        loss_cpu = F.cross_entropy(inp_cpu, target_cpu, reduction="mean")
        loss_cpu.backward()

        assert torch.allclose(loss._local_tensor.cpu(), loss_cpu)

        local_inp_grad = dist_inp.grad._local_tensor
        rank = torch.distributed.get_rank()
        grad_chunks = inp_cpu.grad.chunk(self.world_size, dim=-1)
        assert local_inp_grad.shape[1] == n_classes // self.world_size
        assert torch.allclose(local_inp_grad.cpu(), grad_chunks[rank])

    def test_parallel_fused_cross_entropy_with_tp(self):
        self.run_subtests(
            {
                "logits_shape": [(4, 128256), (4, 1022), (3 * 128, 1022)],
                "reduction": ["mean", "sum"],
            },
            self._test_parallel_fused_cross_entropy_with_tp,
        )

    def _test_parallel_fused_cross_entropy_with_tp(self, logits_shape, reduction):
        bs, n_classes = logits_shape
        assert (
            n_classes % self.world_size == 0
        ), f"n_classes should be divisible by {self.world_size}"
        device = torch.musa.current_device()
        device_mesh = init_device_mesh("musa", (self.world_size,))

        seed = 42
        torch.manual_seed(seed)
        torch.musa.manual_seed(seed)
        torch.musa.manual_seed_all(seed)

        inp = torch.randn(bs, n_classes, device=device, requires_grad=True)
        target = torch.randint(n_classes, (bs,), device=device)

        inp_cpu = inp.detach().cpu()
        inp_cpu.requires_grad_(True)
        target_cpu = target.cpu()

        dist_inp = distribute_tensor(
            inp, device_mesh, placements=[Shard(1)]
        )  # shard at last dim

        loss = parallel_fused_cross_entropy(
            dist_inp, target, reduction=reduction, pg=device_mesh.get_group()
        )
        assert loss.device.type == "musa"
        loss.backward()

        loss_cpu = F.cross_entropy(inp_cpu, target_cpu, reduction=reduction)
        loss_cpu.backward()

        assert torch.allclose(loss._local_tensor.cpu(), loss_cpu)

        local_inp_grad = dist_inp.grad._local_tensor
        rank = torch.distributed.get_rank()
        grad_chunks = inp_cpu.grad.chunk(self.world_size, dim=-1)
        assert local_inp_grad.shape[1] == n_classes // self.world_size
        assert torch.allclose(local_inp_grad.cpu(), grad_chunks[rank])

        # print(f"loss_musa: {loss}")
        # print(f"loss_cpu: {loss_cpu}")

    def test_parallel_fused_cross_entropy_without_tp(self):
        self.run_subtests(
            {
                "logits_shape": [(4, 128256), (3, 1024), (4 * 128, 1025)],
                "reduction": ["mean", "sum"],
                "is_last_layer": [
                    True,
                    False,
                ],
            },
            self._test_parallel_fused_cross_entropy_without_tp,
        )

    def _test_parallel_fused_cross_entropy_without_tp(
        self, logits_shape, reduction, is_last_layer
    ):
        bs, n_classes = logits_shape
        device = torch.musa.current_device()

        inp = torch.randn(bs, n_classes, device=device, requires_grad=True)
        target = torch.randint(n_classes, (bs,), device=device)

        inp_cpu = inp.detach().cpu()
        inp_cpu.requires_grad_(True)
        target_cpu = target.cpu()

        loss = parallel_fused_cross_entropy(inp, target, reduction=reduction)
        assert loss.device.type == "musa"
        if not is_last_layer:
            loss *= 2.0
        loss.backward()

        loss_cpu = F.cross_entropy(inp_cpu, target_cpu, reduction=reduction)
        if not is_last_layer:
            loss_cpu *= 2.0
        loss_cpu.backward()

        # print(f"loss_musa: {loss.detach().cpu()}")
        # print(f"loss_cpu: {loss_cpu}")

        assert torch.allclose(loss.cpu(), loss_cpu)
        assert torch.allclose(inp.grad.cpu(), inp_cpu.grad)
