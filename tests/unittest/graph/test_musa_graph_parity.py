# pylint: disable=missing-module-docstring,missing-function-docstring

import pytest
import torch

from torch_musa import testing


DEVICE = "musa"


def _graph_supported():
    return testing.get_musa_arch() >= 22


pytestmark = pytest.mark.skipif(
    not _graph_supported(),
    reason="MUSAGraph is not supported on arch older than qy2",
)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_graph_is_current_stream_capturing():
    graph = torch.musa.MUSAGraph()
    static = torch.ones(4, device=DEVICE)

    assert not torch.musa.is_current_stream_capturing()
    with torch.musa.graph(graph):
        assert torch.musa.is_current_stream_capturing()
        static.add_(1)
    assert not torch.musa.is_current_stream_capturing()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_graph_capture_error_modes():
    for capture_error_mode in ("global", "thread_local", "relaxed"):
        graph = torch.musa.MUSAGraph()
        static = torch.ones(4, device=DEVICE)
        with torch.musa.graph(graph, capture_error_mode=capture_error_mode):
            static.add_(1)
        graph.replay()
        torch.musa.synchronize()
        assert testing.DefaultComparator(static.cpu(), torch.full((4,), 3.0))


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_graph_reset_and_recapture():
    graph = torch.musa.MUSAGraph()
    static = torch.ones(4, device=DEVICE)

    with torch.musa.graph(graph):
        static.add_(1)
    graph.replay()
    torch.musa.synchronize()
    assert testing.DefaultComparator(static.cpu(), torch.full((4,), 3.0))

    graph.reset()
    static.fill_(1)
    with torch.musa.graph(graph):
        static.mul_(3)
    graph.replay()
    torch.musa.synchronize()
    assert testing.DefaultComparator(static.cpu(), torch.full((4,), 9.0))


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_graph_raw_handle_error_paths():
    with pytest.raises(RuntimeError):
        torch.musa.MUSAGraph(keep_graph=True).instantiate()

    keep_graph = torch.musa.MUSAGraph(keep_graph=True)
    static = torch.ones(4, device=DEVICE)
    with torch.musa.graph(keep_graph):
        static.add_(1)

    with pytest.raises(RuntimeError):
        keep_graph.raw_musa_graph_exec()

    assert keep_graph.raw_musa_graph() != 0
    keep_graph.instantiate()
    assert keep_graph.raw_musa_graph_exec() != 0

    default_graph = torch.musa.MUSAGraph()
    with torch.musa.graph(default_graph):
        static.add_(1)

    with pytest.raises(RuntimeError):
        default_graph.raw_musa_graph()
    assert default_graph.raw_musa_graph_exec() != 0


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_graph_explicit_pool_sharing():
    pool = torch.musa.graph_pool_handle()
    x = torch.ones(4, device=DEVICE)
    y = torch.ones(4, device=DEVICE)
    graph1 = torch.musa.MUSAGraph()
    graph2 = torch.musa.MUSAGraph()

    with torch.musa.graph(graph1, pool=pool):
        x.add_(1)
    with torch.musa.graph(graph2, pool=pool):
        y.mul_(2)

    graph1.replay()
    graph2.replay()
    torch.musa.synchronize()
    assert testing.DefaultComparator(x.cpu(), torch.full((4,), 3.0))
    assert testing.DefaultComparator(y.cpu(), torch.full((4,), 4.0))


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_mgc_backward_matches_eager():
    torch.manual_seed(0)
    module = torch.nn.Linear(4, 4).to(DEVICE)
    eager = torch.nn.Linear(4, 4).to(DEVICE)
    eager.load_state_dict(module.state_dict())

    sample = torch.randn(8, 4, device=DEVICE, requires_grad=True)
    graphed = torch.musa.make_graphed_callables(module, (sample,))

    real_input = torch.randn(8, 4, device=DEVICE, requires_grad=True)
    eager_input = real_input.detach().clone().requires_grad_()

    graphed(real_input).sum().backward()
    eager.forward(eager_input).sum().backward()

    assert testing.DefaultComparator(real_input.grad.cpu(), eager_input.grad.cpu())
    assert testing.DefaultComparator(module.weight.grad.cpu(), eager.weight.grad.cpu())
    assert testing.DefaultComparator(module.bias.grad.cpu(), eager.bias.grad.cpu())


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_mgc_same_pool_tuple():
    x = torch.randn(8, 4, device=DEVICE)
    y = torch.randn(8, 4, device=DEVICE)

    graphed_add, graphed_mul = torch.musa.make_graphed_callables(
        (lambda inp: inp + 1, lambda inp: inp * 2),
        ((x,), (y,)),
    )

    with torch.no_grad():
        add_result = graphed_add(torch.full_like(x, 2))
        mul_result = graphed_mul(torch.full_like(y, 3))

    assert testing.DefaultComparator(add_result.cpu(), torch.full_like(x.cpu(), 3))
    assert testing.DefaultComparator(mul_result.cpu(), torch.full_like(y.cpu(), 6))


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_mgc_same_pool_explicit():
    pool = torch.musa.graph_pool_handle()
    x = torch.randn(8, 4, device=DEVICE)
    y = torch.randn(8, 4, device=DEVICE)

    graphed_add = torch.musa.make_graphed_callables(
        lambda inp: inp + 1, (x,), pool=pool
    )
    graphed_mul = torch.musa.make_graphed_callables(
        lambda inp: inp * 2, (y,), pool=pool
    )

    with torch.no_grad():
        add_result = graphed_add(torch.full_like(x, 2))
        mul_result = graphed_mul(torch.full_like(y, 3))

    assert testing.DefaultComparator(add_result.cpu(), torch.full_like(x.cpu(), 3))
    assert testing.DefaultComparator(mul_result.cpu(), torch.full_like(y.cpu(), 6))
