"""test tensor property related operators"""

# pylint: disable=C0116,C0103
import torch
from torch_musa import testing


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_is_set_to():
    device = torch.musa.current_device()
    t1 = torch.empty(3, 4, 9, 10, device=device)
    t2 = torch.empty(3, 4, 9, 10, device=device)
    t3 = torch.tensor([], device=device).set_(t1)
    t4 = t3.clone().resize_(12, 90)
    assert not t1.is_set_to(t2)
    assert t1.is_set_to(t3)
    assert t3.is_set_to(t1), "is_set_to should be symmetric"
    assert not t1.is_set_to(t4)
    assert not torch.tensor([]).is_set_to(
        torch.tensor([])
    ), "Tensors with no storages should not appear to be set to each other"

    t1 = torch.tensor([True, True], dtype=torch.bool, device=device)
    t2 = torch.tensor([0], dtype=torch.bool, device=device).set_(t1)
    assert t1.is_set_to(t2)

    # test that sizes must match
    t1 = torch.empty([2, 3, 4], device=device)
    t2 = t1.view(4, 3, 2)
    assert not t1.is_set_to(t2)
    assert not t2.is_set_to(t1)

    # test that legacy empty size behavior used to be respected (i.e. all
    # empty tensors were logically collapsed to size [0]).
    t1 = torch.empty([2, 5, 0], device=device)
    t2 = t1.view([0])
    assert not t1.is_set_to(t2)
    assert not t2.is_set_to(t1)
