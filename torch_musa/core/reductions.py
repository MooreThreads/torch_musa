""" Add torch musa event to forking pickler."""
from multiprocessing.reduction import ForkingPickler
import torch


def rebuild_event(device, handle):
    return torch.musa.Event.from_ipc_handle(device, handle)


def reduce_event(event):
    handle = event.ipc_handle()
    return (rebuild_event, (event.device, handle))


def init_reductions():
    ForkingPickler.register(torch.musa.Event, reduce_event)
