#!/usr/bin/env python
# encoding: utf-8

import torch
import torch_musa
import torchvision

model = torchvision.models.resnet50(pretrained=True)
model = model.eval().to("musa")

input = torch.rand(1, 3, 224, 224).to("musa")
model = torch.jit.trace(model, input, check_trace=False)

model.save("resnet50.pt")
print("save model to: resnet50.pt")
