---
title: 简介
description: torch_musa 简介
hide_table_of_contents: False
---

# 简介

## MUSA概述

MUSA (Metaverse Unified System Architecture) 是摩尔线程公司为摩尔线程GPU推出的一种通用并行计算平台和编程模型。它提供了GPU编程的简易接口，用MUSA编程可以构建基于GPU计算的应用程序，利用GPUs的并行计算引擎来更加高效地解决比较复杂的计算难题。同时摩尔线程还推出了MUSA工具箱（MUSAToolkits），工具箱中包括GPU加速库，运行时库，编译器，调试和优化工具等。MUSAToolkits为开发人员在摩尔线程GPU上开发和部署高性能异构计算程序提供软件环境。

关于MUSA软件栈的更多内容，请参见MUSA官方文档。

## PyTorch概述

PyTorch是一款开源的深度学习编程框架，可以用于计算机视觉，自然语言处理，语音处理等领域。PyTorch使用动态计算，这在构建复杂架构时提供了更大的灵活性。PyTorch使用核心Python概念，如类、结构和条件循环，因此理解起来更直观，编程更容易。此外，PyTorch还具有可以轻松扩展、快速实现、生产部署稳定性强等优点。

关于PyTorch的更多内容，请参见PyTorch官方文档。

## torch_musa概述

为了摩尔线程GPU能支持开源框架PyTorch，摩尔线程公司开发了torch_musa。在PyTorch v2.0.0基础上，torch_musa以插件的形式来支持摩尔线程GPU，最大程度与PyTorch代码解耦，便于代码维护与升级。torch_musa利用PyTorch提供的第三方后端扩展接口，将摩尔线程高性能计算库动态注册到PyTorch上，从而使得PyTorch框架能够利用摩尔线程显卡的高性能计算单元。利用摩尔线程显卡CUDA兼容的特性，torch_musa内部引入了cuda兼容模块，使得PyTorch社区的CUDA kernels经过porting后就可以运行在摩尔线程显卡上，而且CUDA Porting的工作是在编译torch_musa的过程中自动进行，这大幅降低了torch_musa算子适配的成本，提高模型开发效率。同时，torch_musa在Python前端接口与PyTorch社区CUDA接口形式上基本保持一致，这极大地降低了用户的学习成本和模型的迁移成本。

本手册主要介绍了基于MUSA软件栈的torch_musa开发指南。

## torch_musa核心代码目录概述

- torch_musa/tests 测试文件。
- torch_musa/core 主要包含Python module，提供amp/device/memory/stream/event等模块的Python前端接口。
- torch_musa/csrc c++侧实现代码；
  - csrc/amp 提供混合精度模块的C++实现。
  - csrc/aten 提供C++ Tensor库，包括MUDNN算子适配，CUDA-Porting算子适配等等。
  - csrc/core 提供核心功能库，包括设备管理，内存分配管理，Stream管理，Events管理等。
  - csrc/distributed 提供分布式模块的C++实现。
