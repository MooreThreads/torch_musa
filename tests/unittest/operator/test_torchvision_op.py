"""Test torchvision operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import, not-callable
from collections import OrderedDict

import pytest
import torch
import torchvision

from torch_musa import testing

support_dtypes = [torch.float32]


def get_forge_data(num_boxes):
    boxes = torch.cat((torch.rand(num_boxes, 2), torch.rand(num_boxes, 2) + 10), dim=1)
    assert max(boxes[:, 0]) < min(boxes[:, 2])  # x1 < x2
    assert max(boxes[:, 1]) < min(boxes[:, 3])  # y1 < y2
    scores = torch.rand(num_boxes)
    idxs = torch.randint(0, 4, size=(num_boxes,))
    return boxes, scores, idxs


def make_rois(img_size, num_imgs, dtype, num_rois=1000):
    rois = torch.randint(0, img_size // 2, size=(num_rois, 5)).to(dtype)
    rois[:, 0] = torch.randint(0, num_imgs, size=(num_rois,))  # set batch index
    rois[:, 3:] += rois[:, 1:3]  # make sure boxes aren't degenerate
    return rois


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("num_boxes", [1000, 500, 10])
@pytest.mark.parametrize("iou_threshold", [0.9, 0.5, 0.1])
@pytest.mark.parametrize("dtype", support_dtypes)
def test_batched_nms(num_boxes, iou_threshold, dtype):
    boxes, scores, idxs = get_forge_data(num_boxes)
    input_args = {}
    input_args["boxes"] = boxes.to(dtype)
    input_args["scores"] = scores.to(dtype)
    input_args["idxs"] = idxs.to(dtype)
    input_args["iou_threshold"] = iou_threshold
    test = testing.OpTest(func=torchvision.ops.batched_nms, input_args=input_args)
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("size", [(3, 224, 224), (1, 300, 600), (10, 1000, 2000)])
@pytest.mark.parametrize("dtype", support_dtypes)
def test_masks_to_boxes(size, dtype):
    input_args = {}
    masks = torch.rand(*size).uniform_() > 0.8
    input_args["masks"] = masks.to(dtype)
    test = testing.OpTest(func=torchvision.ops.masks_to_boxes,
                          input_args=input_args)
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("num_boxes", [1000, 500, 10])
@pytest.mark.parametrize("iou_threshold", [0.9, 0.5, 0.1])
@pytest.mark.parametrize("dtype", support_dtypes)
def test_nms(num_boxes, iou_threshold, dtype):
    boxes, scores, _ = get_forge_data(num_boxes)
    input_args = {}
    input_args["boxes"] = boxes.to(dtype)
    input_args["scores"] = scores.to(dtype)
    input_args["iou_threshold"] = iou_threshold
    test = testing.OpTest(func=torchvision.ops.nms,
                          input_args=input_args)
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("spatial_scale", [0.5, 1.0, 1.5, 2.0])
@pytest.mark.parametrize("sampling_ratio", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("aligned", [True, False])
def test_roi_align(dtype, spatial_scale, sampling_ratio, aligned):
    pool_size = 5
    img_size = 10
    n_channels = 2
    num_imgs = 1

    x = torch.randint(50, 100, size=(num_imgs, n_channels, img_size, img_size)).to(dtype)
    rois = make_rois(img_size, num_imgs, dtype)
    input_args = {}
    input_args["input"] = x.to(dtype)
    input_args["boxes"] = rois.to(dtype)
    input_args["output_size"] = pool_size
    input_args["spatial_scale"] = spatial_scale
    input_args["sampling_ratio"] = sampling_ratio
    input_args["aligned"] = aligned
    test = testing.OpTest(func=torchvision.ops.roi_align,
                          input_args=input_args)
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("spatial_scale", [0.5, 1.0, 1.5, 2.0])
def test_roi_pool(dtype, spatial_scale):
    pool_size = 5
    img_size = 10
    n_channels = 2
    num_imgs = 1

    x = torch.randint(50, 100, size=(num_imgs, n_channels, img_size, img_size)).to(dtype)
    rois = make_rois(img_size, num_imgs, dtype)
    input_args = {}
    input_args["input"] = x.to(dtype)
    input_args["boxes"] = rois.to(dtype)
    input_args["output_size"] = pool_size
    input_args["spatial_scale"] = spatial_scale
    test = testing.OpTest(func=torchvision.ops.roi_pool,
                          input_args=input_args)
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("spatial_scale", [0.5, 1.0, 1.5, 2.0])
@pytest.mark.parametrize("sampling_ratio", [-2, -1, 0, 1, 2])
def test_ps_roi_align(dtype, spatial_scale, sampling_ratio):
    pool_size = 5
    img_size = 10
    n_channels = 25
    num_imgs = 1

    x = torch.randint(50, 100, size=(num_imgs, n_channels, img_size, img_size)).to(dtype)
    rois = make_rois(img_size, num_imgs, dtype)
    input_args = {}
    input_args["input"] = x.to(dtype)
    input_args["boxes"] = rois.to(dtype)
    input_args["output_size"] = pool_size
    input_args["spatial_scale"] = spatial_scale
    input_args["sampling_ratio"] = sampling_ratio
    test = testing.OpTest(func=torchvision.ops.ps_roi_align,
                          input_args=input_args)
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("spatial_scale", [0.5, 1.0, 1.5, 2.0])
def test_ps_roi_pool(dtype, spatial_scale):
    pool_size = 5
    img_size = 10
    n_channels = 25
    num_imgs = 1

    x = torch.randint(50, 100, size=(num_imgs, n_channels, img_size, img_size)).to(dtype)
    rois = make_rois(img_size, num_imgs, dtype)
    input_args = {}
    input_args["input"] = x.to(dtype)
    input_args["boxes"] = rois.to(dtype)
    input_args["output_size"] = pool_size
    input_args["spatial_scale"] = spatial_scale
    test = testing.OpTest(func=torchvision.ops.ps_roi_pool,
                          input_args=input_args)
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("in_channels_list ", [[10, 20, 30]])
@pytest.mark.parametrize("out_channels", [5])
@pytest.mark.parametrize("dtype", support_dtypes)
def test_feature_pyramid_network(in_channels_list, out_channels, dtype):
    m = torchvision.ops.FeaturePyramidNetwork(in_channels_list, out_channels)
    x = OrderedDict()
    x['feat0'] = torch.rand(1, 10, 64, 64).to(dtype)
    x['feat2'] = torch.rand(1, 20, 16, 16).to(dtype)
    x['feat3'] = torch.rand(1, 30, 8, 8).to(dtype)

    output_cpu = m(x)

    x['feat0'] = x['feat0'].to("musa")
    x['feat2'] = x['feat2'].to("musa")
    x['feat3'] = x['feat3'].to("musa")
    m = m.to("musa")

    output_musa = m(x)

    assert torch.allclose(output_cpu['feat0'], output_musa['feat0'].to("cpu"))
    assert torch.allclose(output_cpu['feat2'], output_musa['feat2'].to("cpu"))
    assert torch.allclose(output_cpu['feat3'], output_musa['feat3'].to("cpu"))


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("featmap_names", [['feat1', 'feat3']])
@pytest.mark.parametrize("output_size", [3])
@pytest.mark.parametrize("sampling_ratio", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("dtype", support_dtypes)
def test_multi_scale_roi_align(featmap_names, output_size, sampling_ratio, dtype):
    m = torchvision.ops.MultiScaleRoIAlign(featmap_names, output_size, sampling_ratio)
    i = OrderedDict()
    i['feat1'] = torch.rand(1, 5, 64, 64).to(dtype)
    i['feat2'] = torch.rand(1, 5, 32, 32).to(dtype)
    i['feat3'] = torch.rand(1, 5, 16, 16).to(dtype)

    # after https://jira.mthreads.com/browse/SW-26801 is solved then fake_data can be modified as
    # torch.rand(6, 4)
    fake_data = torch.tensor([
        [0.1678, 0.4586, 0.6704, 0.5342],
        [0.7101, 0.4826, 0.9034, 0.8015],
        [0.9226, 0.9472, 0.1856, 0.3235],
        [0.1684, 0.7336, 0.2768, 0.2755],
        [0.4598, 0.8958, 0.4561, 0.5798],
        [0.6150, 0.7043, 0.3712, 0.3182]
    ])
    boxes = fake_data * 256
    boxes[:, 2:] += boxes[:, :2]
    image_sizes = [(512, 512)]

    output_cpu = m(i, [boxes], image_sizes)

    i['feat1'] = i['feat1'].to("musa")
    i['feat2'] = i['feat2'].to("musa")
    i['feat3'] = i['feat3'].to("musa")
    boxes = boxes.to("musa")
    m = m.to("musa")

    output_musa = m(i, [boxes], image_sizes)

    assert torch.allclose(output_cpu, output_musa.to("cpu"))


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_box_area(dtype):
    img_size = 10
    num_imgs = 1
    rois = make_rois(img_size, num_imgs, dtype)
    input_args = {}
    input_args["boxes"] = rois.to(dtype)
    test = testing.OpTest(func=torchvision.ops.box_area,
                          input_args=input_args
                          )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("in_fmt", ["xyxy", "xywh", "cxcywh"])
@pytest.mark.parametrize("out_fmt", ["xyxy", "xywh", "cxcywh"])
def test_box_convert(dtype, in_fmt, out_fmt):
    boxes = torch.rand(6, 4) * 256
    boxes[:, 2:] += boxes[:, :2]
    input_args = {}
    input_args["boxes"] = boxes.to(dtype)
    input_args["in_fmt"] = in_fmt
    input_args["out_fmt"] = out_fmt
    test = testing.OpTest(func=torchvision.ops.box_convert, input_args=input_args)
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_box_iou(dtype):
    boxes1 = torch.rand(6, 4) * 256
    boxes1[:, 2:] += boxes1[:, :2]
    boxes2 = torch.rand(6, 4) * 256
    boxes2[:, 2:] += boxes2[:, :2]
    input_args = {}
    input_args["boxes1"] = boxes1.to(dtype)
    input_args["boxes2"] = boxes2.to(dtype)

    test = testing.OpTest(func=torchvision.ops.box_iou, input_args=input_args)
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("size", [(100, 100), (50, 200)])
def test_clip_boxes_to_image(dtype, size):
    boxes = torch.rand(6, 4) * 256
    boxes[:, 2:] += boxes[:, :2]
    input_args = {}
    input_args["boxes"] = boxes.to(dtype)
    input_args["size"] = size

    test = testing.OpTest(func=torchvision.ops.clip_boxes_to_image,
                          input_args=input_args
                          )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_complete_box_iou(dtype):
    boxes1 = torch.rand(6, 4) * 256
    boxes1[:, 2:] += boxes1[:, :2]
    boxes2 = torch.rand(6, 4) * 256
    boxes2[:, 2:] += boxes2[:, :2]
    input_args = {}
    input_args["boxes1"] = boxes1.to(dtype)
    input_args["boxes2"] = boxes2.to(dtype)
    input_args["eps"] = 1e-6

    test = testing.OpTest(func=torchvision.ops.complete_box_iou, input_args=input_args)
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_distance_box_iou(dtype):
    boxes1 = torch.rand(6, 4) * 256
    boxes1[:, 2:] += boxes1[:, :2]
    boxes2 = torch.rand(6, 4) * 256
    boxes2[:, 2:] += boxes2[:, :2]
    input_args = {}
    input_args["boxes1"] = boxes1.to(dtype)
    input_args["boxes2"] = boxes2.to(dtype)
    input_args["eps"] = 1e-6

    test = testing.OpTest(func=torchvision.ops.distance_box_iou, input_args=input_args)
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_generalized_box_iou(dtype):
    boxes1 = torch.rand(6, 4) * 256
    boxes1[:, 2:] += boxes1[:, :2]
    boxes2 = torch.rand(6, 4) * 256
    boxes2[:, 2:] += boxes2[:, :2]
    input_args = {}
    input_args["boxes1"] = boxes1.to(dtype)
    input_args["boxes2"] = boxes2.to(dtype)

    test = testing.OpTest(func=torchvision.ops.generalized_box_iou, input_args=input_args)
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_remove_small_boxes(dtype):
    boxes = torch.rand(6, 4) * 256
    boxes[:, 2:] += boxes[:, :2]
    input_args = {}
    input_args["boxes"] = boxes.to(dtype)
    input_args["min_size"] = 100

    test = testing.OpTest(func=torchvision.ops.remove_small_boxes, input_args=input_args)
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("reduction", ['none', 'mean', 'sum'])
@pytest.mark.parametrize("eps", [1e-4, 1e-5, 1e-7])
def test_complete_box_iou_loss(dtype, reduction, eps):
    boxes1 = torch.rand(6, 4) * 256
    boxes1[:, 2:] += boxes1[:, :2]
    boxes2 = torch.rand(6, 4) * 256
    boxes2[:, 2:] += boxes2[:, :2]
    input_args = {}
    input_args["boxes1"] = boxes1.to(dtype)
    input_args["boxes2"] = boxes2.to(dtype)
    input_args["reduction"] = reduction
    input_args["eps"] = eps

    test = testing.OpTest(func=torchvision.ops.complete_box_iou_loss,
                          input_args=input_args
                          )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("reduction", ['none', 'mean', 'sum'])
@pytest.mark.parametrize("eps", [1e-4, 1e-5, 1e-7])
def test_distance_box_iou_loss(dtype, reduction, eps):
    boxes1 = torch.rand(6, 4) * 256
    boxes1[:, 2:] += boxes1[:, :2]
    boxes2 = torch.rand(6, 4) * 256
    boxes2[:, 2:] += boxes2[:, :2]
    input_args = {}
    input_args["boxes1"] = boxes1.to(dtype)
    input_args["boxes2"] = boxes2.to(dtype)
    input_args["reduction"] = reduction
    input_args["eps"] = eps

    test = testing.OpTest(func=torchvision.ops.distance_box_iou_loss,
                          input_args=input_args
                          )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("reduction", ['none', 'mean', 'sum'])
@pytest.mark.parametrize("eps", [1e-4, 1e-5, 1e-7])
def test_generalized_box_iou_loss(dtype, reduction, eps):
    boxes1 = torch.rand(6, 4) * 256
    boxes1[:, 2:] += boxes1[:, :2]
    boxes2 = torch.rand(6, 4) * 256
    boxes2[:, 2:] += boxes2[:, :2]
    input_args = {}
    input_args["boxes1"] = boxes1.to(dtype)
    input_args["boxes2"] = boxes2.to(dtype)
    input_args["reduction"] = reduction
    input_args["eps"] = eps

    test = testing.OpTest(func=torchvision.ops.generalized_box_iou_loss,
                          input_args=input_args
                          )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("alpha", [-1, 0.25, 0.5, 0.85])
@pytest.mark.parametrize("gamma", [2, 3, 4])
@pytest.mark.parametrize("reduction", ['none', 'mean', 'sum'])
def test_sigmoid_focal_loss(dtype, alpha, gamma, reduction):
    inputs = torch.rand(6, 4) * 256
    inputs[:, 2:] += inputs[:, :2]
    targets = torch.rand(6, 4) * 256
    targets[:, 2:] += targets[:, :2]
    input_args = {}
    input_args["inputs"] = inputs.to(dtype)
    input_args["targets"] = targets.to(dtype)
    input_args["reduction"] = reduction
    input_args["alpha"] = alpha
    input_args["gamma"] = gamma
    test = testing.OpTest(func=torchvision.ops.sigmoid_focal_loss, input_args=input_args)
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype ", support_dtypes)
@pytest.mark.parametrize("in_channels", [10, 100])
@pytest.mark.parametrize("out_channels ", [5, 50])
@pytest.mark.parametrize("kernel_size ", [3, 5, 7])
@pytest.mark.parametrize("stride", [1, 2])
def test_conv2d_norm_activation(dtype, in_channels, out_channels, kernel_size, stride):
    m = torchvision.ops.Conv2dNormActivation(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride
    )
    input_data = torch.rand(1, in_channels, 10, 10).to(dtype)
    output_cpu = m(input_data)

    input_data = input_data.to("musa")
    m = m.to("musa")
    output_musa = m(input_data)

    assert torch.allclose(output_cpu, output_musa.to("cpu"), atol=1e-5)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype ", support_dtypes)
@pytest.mark.parametrize("in_channels", [10, 100])
@pytest.mark.parametrize("out_channels ", [5, 50])
@pytest.mark.parametrize("kernel_size ", [3, 5, 7])
@pytest.mark.parametrize("stride", [1, 2])
def test_conv3d_norm_activation(dtype, in_channels, out_channels, kernel_size, stride):
    m = torchvision.ops.Conv3dNormActivation(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride
    )
    input_data = torch.rand(1, in_channels, 10, 10, 10).to(dtype)
    output_cpu = m(input_data)

    input_data = input_data.to("musa")
    m = m.to("musa")
    output_musa = m(input_data)

    assert torch.allclose(output_cpu, output_musa.to("cpu"), atol=1e-4)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("num_features", [100, 300, 600])
@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("eps", [1e-1, 1e-3, 1e-5])
def test_frozen_batch_norm2d(num_features, dtype, eps):
    m = torchvision.ops.FrozenBatchNorm2d(
        num_features=num_features,
        eps=eps
    )
    input_data = torch.rand(3, num_features, 600, 400).to(dtype)
    output_cpu = m(input_data)

    input_data = input_data.to("musa")
    m = m.to("musa")
    output_musa = m(input_data)

    assert torch.allclose(output_cpu, output_musa.to("cpu"))


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("in_channels", [100, 200, 300])
@pytest.mark.parametrize("hidden_channels", [[50, 100, 150], [150, 200, 250]])
@pytest.mark.parametrize("dtype", support_dtypes)
def test_mlp(in_channels, hidden_channels, dtype):
    m = torchvision.ops.MLP(
        in_channels=in_channels,
        hidden_channels=hidden_channels
    )
    input_data = torch.rand(3, 3, 600, in_channels).to(dtype)
    output_cpu = m(input_data)

    input_data = input_data.to("musa")
    m = m.to("musa")
    output_musa = m(input_data)

    assert torch.allclose(output_cpu, output_musa.to("cpu"))


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("dims", [[0, 2, 1], [1, 2, 0]])
def test_permute(dtype, dims):
    m = torchvision.ops.Permute(
        dims=dims
    )
    input_data = torch.rand(3, 4, 5).to(dtype)
    output_cpu = m(input_data)

    input_data = input_data.to("musa")
    m = m.to("musa")
    output_musa = m(input_data)

    assert torch.allclose(output_cpu, output_musa.to("cpu"))


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_channels", [100, 200, 300])
@pytest.mark.parametrize("squeeze_channels", [25, 50, 75])
@pytest.mark.parametrize("dtype", support_dtypes)
def test_squeeze_excitation(input_channels, squeeze_channels, dtype):
    m = torchvision.ops.SqueezeExcitation(
        input_channels=input_channels,
        squeeze_channels=squeeze_channels
    )
    input_data = torch.rand(3, input_channels, 20, 40).to(dtype)
    output_cpu = m(input_data)

    input_data = input_data.to("musa")
    m = m.to("musa")
    output_musa = m(input_data)

    assert torch.allclose(output_cpu, output_musa.to("cpu"))


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("p", [0, 1])
@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("mode", ["batch", "row"])
@pytest.mark.parametrize("training", [True, False])
def test_stochastic_depth(p, dtype, mode, training):
    input_args = {}
    input_args["input"] = torch.rand(100, 200).to(dtype)
    input_args["p"] = p
    input_args["mode"] = mode
    input_args["training"] = training
    test = testing.OpTest(func=torchvision.ops.stochastic_depth,
                          input_args=input_args
                          )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_deform_conv2d(dtype):
    input_data = torch.rand(4, 3, 10, 10)
    kernel_height, kernel_width = 3, 3

    input_args = {}
    input_args["input"] = input_data.to(dtype)
    input_args["weight"] = torch.rand(5, 3, kernel_height, kernel_width).to(dtype)
    input_args["offset"] = torch.rand(4, 2 * kernel_height * kernel_width, 8, 8).to(dtype)
    input_args["mask"] = torch.rand(4, kernel_height * kernel_width, 8, 8).to(dtype)

    test = testing.OpTest(func=torchvision.ops.deform_conv2d,
                          input_args=input_args
                          )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("p", [0])
@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("block_size", [1, 3])
def test_drop_block2d(p, dtype, block_size):
    input_data = torch.rand(1, 3, 4, 4)

    input_args = {}
    input_args["input"] = input_data.to(dtype)
    input_args["p"] = p
    input_args["block_size"] = block_size
    test = testing.OpTest(func=torchvision.ops.drop_block2d,
                          input_args=input_args
                          )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("p", [0])
@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("block_size", [1, 3])
def test_drop_block3d(p, dtype, block_size):
    input_data = torch.rand(1, 3, 4, 4, 4)

    input_args = {}
    input_args["input"] = input_data.to(dtype)
    input_args["p"] = p
    input_args["block_size"] = block_size
    test = testing.OpTest(func=torchvision.ops.drop_block3d,
                          input_args=input_args,
                          comparators=testing.DefaultComparator(abs_diff=1e-6)
                          )
    test.check_result()
