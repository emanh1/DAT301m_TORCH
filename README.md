# SSMD — PyTorch Implementation

Full PyTorch implementation of  
**"SSMD: Semi-Supervised Medical Image Detection with Adaptive Consistency and Heterogeneous Perturbation"**  
Zhou et al., arXiv 2106.01544.

---

## File Map

```
ssmd/
├── models/
│   ├── detector.py                  # RetinaNet + NRB injection  (Sec. 3)
│   ├── noisy_residual_block.py      # NRB — feature-space perturbation  (Sec. 3.2, Fig. 3, Eq. 7)
│   ├── adaptive_consistency_cost.py # ACC — weighted KL + MSE loss  (Sec. 3.1, Eq. 4-5)
│   └── adversarial_perturbation.py  # IAP — instance-level adversarial noise  (Sec. 3.3, Eq. 8-9)
├── utils/
│   ├── augmentations.py             # Cutout, student/teacher aug pipelines  (Algo. 1 lines 1-2)
│   ├── ema_teacher.py               # EMA weight update  (Eq. 2)
│   └── lambda_schedule.py           # λ consistency ramp schedule  (Eq. 10)
├── trainer.py                       # Full Algorithm 1 training loop
└── tests.py                         # Unit tests (no network needed)
```

---

## Requirements

```
torch >= 1.10
torchvision >= 0.11
```

Install:
```bash
pip install torch torchvision
```

---

## Quick Start

```python
import torch
from ssmd import SSMDTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainer = SSMDTrainer(
    num_classes=1,          # 1 for lesion/nuclei detection
    device=device,
    ema_alpha=0.99,         # Table 6: optimal EMA decay
    lr=1e-5,                # Paper Sec. 4.2
    total_epochs=100,
    steps_per_epoch=len(labeled_loader),
    xi=5e-7,                # Table 5: adversarial seed scale
    adv_eps=2.0,            # Table 5: adversarial magnitude
    tau=0.95,               # Fig. 5: foreground confidence threshold
    cutout_n=5,             # Table 8: number of cutout masks
    cutout_s=70,            # Table 8: cutout mask side length
    max_rot_deg=10.0,       # Sec. 4.2
    nrb_gamma=0.9,          # Sec. 4.2
)

# Training loop
for epoch in range(100):
    for (lab_imgs, lab_targets), (ulab_imgs, _) in zip(labeled_loader, unlabeled_loader):
        metrics = trainer.train_step(lab_imgs, lab_targets, ulab_imgs)
        print(f"loss_sup={metrics['loss_sup']:.4f}  "
              f"loss_cons={metrics['loss_cons']:.4f}  "
              f"λ={metrics['lambda']:.4f}")

    trainer.save_checkpoint(f"ckpt_epoch{epoch}.pt")
```

---

## Paper → Code Mapping

| Paper element | File | Symbol |
|---|---|---|
| Student/Teacher framework (Fig. 1) | `trainer.py` | `SSMDTrainer` |
| EMA teacher update (Eq. 2) | `utils/ema_teacher.py` | `EMATeacher.update()` |
| Noisy Residual Block (Fig. 3, Eq. 7) | `models/noisy_residual_block.py` | `NoisyResidualBlock` |
| Adaptive instance weight (Eq. 4) | `models/adaptive_consistency_cost.py` | `_instance_weight()` |
| KL + MSE consistency loss (Eq. 5, Algo. 1 L11-12) | `models/adaptive_consistency_cost.py` | `AdaptiveConsistencyCost.forward()` |
| Adversarial perturbation (Eq. 8-9) | `models/adversarial_perturbation.py` | `instance_adversarial_perturbation()` |
| Student augmentation (Algo. 1 L1) | `utils/augmentations.py` | `student_augment()` |
| Teacher augmentation (Algo. 1 L2) | `utils/augmentations.py` | `teacher_augment_base()` |
| Cutout (Sec. 4.5.7) | `utils/augmentations.py` | `Cutout` |
| λ schedule (Eq. 10) | `utils/lambda_schedule.py` | `consistency_lambda()` |
| Full training algorithm (Algo. 1) | `trainer.py` | `SSMDTrainer.train_step()` |

---

## Dataset Setup

### DSB 2018 Nuclei
- Download from https://www.kaggle.com/c/data-science-bowl-2018
- Split 80% train / 10% val / 10% test
- Input resolution: 448×448
- Evaluation: mAP

### DeepLesion
- Download from https://nihcc.app.box.com/v/DeepLesion
- Preprocessing: clip HU to [-1100, 1100], normalise to [-1, 1]
- Resize to 512×512 (mean voxel spacing 0.802 mm)
- Use official 15% test split
- Evaluation: sensitivity at 4 false positives per image

---

## Key Hyperparameters (Paper Defaults)

| Hyperparameter | Value | Source |
|---|---|---|
| EMA decay α | 0.99 | Table 6 |
| Learning rate | 1e-5 | Sec. 4.2 |
| LR step (÷10) | epoch 75/100 | Sec. 4.2 |
| Batch size | 8 | Sec. 4.2 |
| NRB γ | 0.9 | Sec. 4.2 |
| Rotation | ±10° | Sec. 4.2 |
| Cutout n, s | 5, 70 | Table 8 |
| Adversarial ξ | 5e-7 | Table 5 |
| Adversarial ε | 2.0 | Table 5 |
| Threshold τ | 0.95 | Fig. 5 |

---

## Running Tests

```bash
cd /path/to/project
python -m ssmd.tests
```
