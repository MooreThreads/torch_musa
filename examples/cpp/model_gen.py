#!/usr/bin/env python
# encoding: utf-8

"""
Generating toy model
"""

import torch
import torchvision
import torch_musa  # pylint: disable=W0611

model = torchvision.models.resnet50(pretrained=True)
model = model.eval().to("musa")  # pylint: disable=C0103

input_data = torch.rand(1, 3, 224, 224).to("musa")
model = torch.jit.trace(model, input_data, check_trace=False)

model.save("resnet50.pt")
print("save model to: resnet50.pt")
