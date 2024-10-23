"""Test fsdp finetune"""

import io
import pytest

import torch
from torch import nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from torch.distributed._shard.sharded_tensor import ShardedTensor

import torch_musa
from torch_musa import testing
from torch_musa.testing.common_fsdp import FSDPTest


STATE_DICT_MAPPING = {
    "state_dict": StateDictType.FULL_STATE_DICT,
    "local_state_dict": StateDictType.LOCAL_STATE_DICT,
    "sharded_state_dict": StateDictType.SHARDED_STATE_DICT,
}


@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="Skip on arch's version older than 22"
)
class TestFSDPStateDict(FSDPTest):
    """class TestFSDPStateDict"""

    @property
    def world_size(self) -> int:
        return min(torch.musa.device_count(), 2)

    def _init_seq_module(self) -> nn.Module:
        modules = []
        for _ in range(2):
            modules += [nn.Linear(6, 6), nn.ReLU()]
        seq = nn.Sequential(*modules)
        return seq

    def test_save_and_load_state_dict(self):
        """Test case for saving and loading state_dict"""

        model = self._init_seq_module().to(torch_musa.current_device())
        fsdp_model = FSDP(model)
        with FSDP.state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT):
            state_dict = model.state_dict()
            checkpoint = io.BytesIO()
            torch.save(state_dict, checkpoint)
            checkpoint.seek(0)
            state_dict_saved = torch.load(checkpoint)

            for k, v in state_dict_saved.items():
                if isinstance(v, ShardedTensor):
                    assert (
                        v._local_shards[0].tensor.dtype
                        == state_dict[k]._local_shards[0].tensor.dtype
                    )
                    assert torch.allclose(
                        v._local_shards[0].tensor, state_dict[k]._local_shards[0].tensor
                    )
                else:
                    assert v.dtype == state_dict[k].dtype
                    assert torch.allclose(v, state_dict[k])
