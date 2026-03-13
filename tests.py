"""
Unit tests for SSMD components.
Run with:  python -m pytest tests.py -v
or simply: python tests.py
"""

import math
import torch
import pytest


# ---------------------------------------------------------------------------
# NoisyResidualBlock
# ---------------------------------------------------------------------------

def test_noisy_residual_block_shape():
    from ssmd.models.noisy_residual_block import NoisyResidualBlock
    nrb = NoisyResidualBlock(in_channels=64, gamma=0.9)
    x = torch.randn(2, 64, 32, 32)
    out = nrb(x)
    assert out.shape == x.shape, "NRB must preserve feature map shape"


def test_noisy_residual_block_different_from_input():
    from ssmd.models.noisy_residual_block import NoisyResidualBlock
    nrb = NoisyResidualBlock(in_channels=32, gamma=0.9)
    x = torch.randn(1, 32, 16, 16)
    out = nrb(x)
    # Very unlikely to be exactly equal (noise was added)
    assert not torch.allclose(out, x), "NRB should perturb the features"


# ---------------------------------------------------------------------------
# AdaptiveConsistencyCost
# ---------------------------------------------------------------------------

def test_adaptive_consistency_cost_scalar():
    from ssmd.models.adaptive_consistency_cost import AdaptiveConsistencyCost
    acc = AdaptiveConsistencyCost()
    N, K = 50, 3
    cls_s = torch.randn(N, K)
    cls_t = torch.randn(N, K)
    reg_s = torch.randn(N, 4)
    reg_t = torch.randn(N, 4)
    loss = acc(cls_s, cls_t, reg_s, reg_t)
    assert loss.shape == (), "Consistency cost must return a scalar"
    assert loss.item() >= 0, "Consistency cost must be non-negative"


def test_adaptive_weight_bg_suppression():
    """Background proposals should get weight ≈ 0."""
    from ssmd.models.adaptive_consistency_cost import AdaptiveConsistencyCost
    acc = AdaptiveConsistencyCost()
    # Both networks very confident about background (class 0)
    cls_bg = torch.zeros(10, 3)
    cls_bg[:, 0] = 10.0           # high logit for background
    w = AdaptiveConsistencyCost._instance_weight(
        torch.softmax(cls_bg, dim=-1),
        torch.softmax(cls_bg, dim=-1),
    )
    assert w.max().item() < 0.01, "Background proposals should have near-zero weight"


# ---------------------------------------------------------------------------
# ConsistencyScheduler (λ schedule)
# ---------------------------------------------------------------------------

def test_lambda_schedule_bounds():
    from ssmd.utils.lambda_schedule import consistency_lambda
    N = 1000
    for j in range(0, N + 1, 50):
        lam = consistency_lambda(j, N)
        assert 0.0 < lam <= 1.0, f"λ out of bounds at j={j}: {lam}"


def test_lambda_schedule_plateau():
    from ssmd.utils.lambda_schedule import consistency_lambda
    N = 1000
    # Middle quarter should be exactly 1.0
    assert consistency_lambda(N // 2, N) == 1.0


def test_lambda_schedule_ramp_up():
    from ssmd.utils.lambda_schedule import consistency_lambda
    N = 1000
    # λ should increase in the first quarter
    prev = consistency_lambda(0, N)
    for j in range(1, N // 4, 20):
        curr = consistency_lambda(j, N)
        assert curr >= prev, f"λ should be non-decreasing in ramp-up, failed at j={j}"
        prev = curr


# ---------------------------------------------------------------------------
# EMATeacher
# ---------------------------------------------------------------------------

def test_ema_teacher_update():
    from ssmd.utils.ema_teacher import EMATeacher

    class TinyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 4)

    student = TinyNet()
    teacher = EMATeacher(student, alpha=0.9)

    # Manually set student weights to all-ones
    with torch.no_grad():
        for p in student.parameters():
            p.fill_(1.0)
        for p in teacher.model.parameters():
            p.fill_(0.0)

    teacher.update(student)

    for p in teacher.model.parameters():
        # θ_t = 0.9*0 + 0.1*1 = 0.1
        assert torch.allclose(p.data,
                              torch.full_like(p.data, 0.1),
                              atol=1e-5), "EMA update incorrect"


def test_teacher_no_grad():
    from ssmd.utils.ema_teacher import EMATeacher

    class TinyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 2)

    student = TinyNet()
    teacher = EMATeacher(student, alpha=0.99)
    for p in teacher.model.parameters():
        assert not p.requires_grad, "Teacher params must not require gradients"


# ---------------------------------------------------------------------------
# Cutout augmentation
# ---------------------------------------------------------------------------

def test_cutout_zeros_regions():
    from ssmd.utils.augmentations import Cutout
    img = torch.ones(3, 100, 100)
    cutout = Cutout(n_masks=3, mask_size=20)
    out = cutout(img)
    # At least some pixels should be zeroed
    assert out.sum() < img.sum(), "Cutout should zero out some pixels"
    assert out.shape == img.shape, "Cutout must preserve shape"


def test_cutout_no_inplace():
    from ssmd.utils.augmentations import Cutout
    img = torch.ones(3, 50, 50)
    original = img.clone()
    cutout = Cutout(n_masks=2, mask_size=10)
    _ = cutout(img)
    assert torch.allclose(img, original), "Cutout must not modify original tensor"


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_noisy_residual_block_shape,
        test_noisy_residual_block_different_from_input,
        test_adaptive_consistency_cost_scalar,
        test_adaptive_weight_bg_suppression,
        test_lambda_schedule_bounds,
        test_lambda_schedule_plateau,
        test_lambda_schedule_ramp_up,
        test_ema_teacher_update,
        test_teacher_no_grad,
        test_cutout_zeros_regions,
        test_cutout_no_inplace,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed.")
