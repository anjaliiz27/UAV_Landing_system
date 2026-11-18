"""Run from repo root: python -m pytest tests -q"""
import numpy as np
import torch
from models.segmentation_pipeline import SegNet
from src.preprocessing import (MAIN_W, MAIN_H, NUM_CLASSES, upscale_mask,
                               to_main, to_net_size, normalize)


def count_blocks(module, prefix):
    return len([n for n, _ in module.named_children() if n.startswith(prefix)])


def test_layer_budget():
    net = SegNet(NUM_CLASSES)
    assert count_blocks(net.encoder, "enc") == 4
    assert count_blocks(net.decoder, "dec") == 4


def test_forward_shape():
    out = SegNet(NUM_CLASSES)(torch.randn(1, 3, 256, 256))
    assert out.shape == (1, NUM_CLASSES, 256, 256)


def test_can_overfit_one_batch():
    torch.manual_seed(0)
    net, x = SegNet(NUM_CLASSES), torch.randn(2, 3, 256, 256)
    y = torch.randint(0, NUM_CLASSES, (2, 256, 256))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    first = None
    for _ in range(15):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(net(x), y)
        loss.backward(); opt.step()
        first = first or loss.item()
    assert loss.item() < first


def test_mask_upscale_keeps_labels_and_scale():
    m = np.random.randint(0, NUM_CLASSES, (256, 256)).astype(np.uint8)
    up = upscale_mask(m)
    assert up.shape == (MAIN_H, MAIN_W)
    assert set(np.unique(up)) <= set(np.unique(m))       # no blended/fake classes


def test_preprocessing_shapes():
    img = np.random.randint(0, 255, (4000, 6000, 3), dtype=np.uint8)
    main = to_main(img)
    assert main.shape == (MAIN_H, MAIN_W, 3)
    assert normalize(to_net_size(main)).shape == (3, 256, 256)
