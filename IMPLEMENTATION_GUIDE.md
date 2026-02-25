# 🧠 SymPanICH-Net v2 — Implementation Guide

**Text-Guided Symmetry-Aware Panoptic Segmentation with AI Clinical Analysis for ICH Detection**

---

## 📁 Project Structure

```
FYP26/
├── architecture/                    # Architecture visualization
│   └── index.html                   # Interactive HTML architecture page
├── configs/                         # Configuration files
│   ├── default.yaml                 # Default training config
│   ├── model.yaml                   # Model architecture config
│   └── augmentation.yaml            # Augmentation pipeline config
├── data/                            # Dataset (not tracked in git)
│   ├── raw/                         # Raw NIfTI files (.nii)
│   │   ├── patient_001/
│   │   │   ├── ct_scan.nii          # 3D CT volume
│   │   │   └── mask.nii             # Ground truth mask (6 classes)
│   │   └── ...
│   ├── processed/                   # Preprocessed slices (generated)
│   │   ├── images/                  # Multi-window 3-channel images
│   │   ├── masks/                   # Segmentation masks
│   │   └── metadata.csv             # Slice-level metadata
│   └── text_prompts/                # Clinical text descriptions
│       └── hemorrhage_descriptions.json
├── src/                             # Source code
│   ├── __init__.py
│   ├── models/                      # Model components
│   │   ├── __init__.py
│   │   ├── backbone.py              # Module 2: Dual-Stream Swin-T V2
│   │   ├── symmetry.py              # Module 3: SACA module
│   │   ├── fpn.py                   # Module 4: Feature Pyramid Network
│   │   ├── panoptic_head.py         # Module 5: Mask2Former-style decoder
│   │   ├── hover_branch.py          # Module 6: HoVer distance branch
│   │   ├── text_encoder.py          # Module 7: BiomedCLIP text encoder
│   │   ├── cross_modal_attention.py # Cross-modal text-vision attention
│   │   ├── report_generator.py      # Module 8: AI clinical report
│   │   └── sympanich_net.py         # Full model assembly
│   ├── data/                        # Data pipeline
│   │   ├── __init__.py
│   │   ├── preprocessing.py         # Module 1: Multi-window + 2.5D
│   │   ├── dataset.py               # PyTorch Dataset class
│   │   ├── augmentations.py         # Augmentation pipeline
│   │   └── datamodule.py            # PyTorch Lightning DataModule
│   ├── losses/                      # Loss functions
│   │   ├── __init__.py
│   │   ├── focal_loss.py            # Focal Cross-Entropy
│   │   ├── dice_loss.py             # Binary Dice Loss
│   │   ├── hover_loss.py            # MSE for distance maps
│   │   ├── contrastive_loss.py      # InfoNCE for text alignment
│   │   └── combined_loss.py         # Weighted loss combination
│   ├── utils/                       # Utilities
│   │   ├── __init__.py
│   │   ├── metrics.py               # PQ, SQ, RQ, Dice, IoU
│   │   ├── panoptic_fusion.py       # Post-processing + panoptic merge
│   │   ├── visualization.py         # Overlay, mask coloring
│   │   └── ct_windows.py            # HU windowing functions
│   └── training/                    # Training pipeline
│       ├── __init__.py
│       ├── trainer.py               # PyTorch Lightning module
│       └── callbacks.py             # Custom callbacks
├── scripts/                         # Executable scripts
│   ├── preprocess.py                # Run full preprocessing pipeline
│   ├── train.py                     # Launch training
│   ├── evaluate.py                  # Run evaluation
│   ├── predict.py                   # Run inference on new scans
│   └── generate_report.py           # Generate AI clinical report
├── notebooks/                       # Jupyter notebooks
│   ├── 01_data_exploration.ipynb    # EDA on ICH dataset
│   ├── 02_preprocessing_demo.ipynb  # Visualize preprocessing pipeline
│   ├── 03_training_analysis.ipynb   # Training curves + analysis
│   └── 04_results_comparison.ipynb  # v1 vs v2 comparison
├── tests/                           # Unit tests
│   ├── test_preprocessing.py
│   ├── test_model.py
│   ├── test_losses.py
│   └── test_metrics.py
├── requirements.txt                 # Python dependencies
├── README.md                        # Project overview
└── IMPLEMENTATION_GUIDE.md          # This file
```

---

## 🔧 Dependencies & Setup

### requirements.txt

```
# Core
torch>=2.1.0
torchvision>=0.16.0
pytorch-lightning>=2.1.0
timm>=0.9.12

# Medical imaging
nibabel>=5.2.0
SimpleITK>=2.3.1
monai>=1.3.0

# Augmentation
albumentations>=1.3.1

# Text encoder
open-clip-torch>=2.24.0
transformers>=4.36.0

# Report generation (Module 8)
peft>=0.7.0        # LoRA adapters
bitsandbytes>=0.41 # 4-bit quantization

# Metrics & evaluation
torchmetrics>=1.2.0
rouge-score>=0.1.2
bert-score>=0.3.13

# Experiment tracking
wandb>=0.16.0

# Visualization
matplotlib>=3.8.0
seaborn>=0.13.0
opencv-python>=4.8.0

# Utilities
pyyaml>=6.0.1
tqdm>=4.66.0
pandas>=2.1.0
scipy>=1.11.0
scikit-image>=0.22.0
```

### Setup Commands

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install detectron2 (for deformable attention reference)
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 4. Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

---

## 📋 Implementation Phases

### Phase 1: Data Pipeline (Module 1) — Week 1

#### Step 1.1: CT Windowing (`src/utils/ct_windows.py`)

```python
# What to implement:
# - apply_window(ct_slice, center, width) → normalized 0-1 image
# - get_multi_window(ct_slice) → 3-channel (brain, subdural, bone)
#
# Brain:    center=40, width=80
# Subdural: center=75, width=215
# Bone:     center=600, width=2800
#
# Formula: clamp HU to [center-width/2, center+width/2], then normalize to [0,1]
```

#### Step 1.2: 2.5D Context (`src/data/preprocessing.py`)

```python
# What to implement:
# - For each target slice t, get slices [t-2, t-1, t, t+1, t+2]
# - Apply multi-window to each → 5 slices × 3 channels = 15 channels
# - 1×1 Conv projection: 15ch → 3ch (learnable in model)
# - Handle edge cases: pad with zeros for first/last 2 slices
#
# Input:  NIfTI volume (.nii) → shape (H, W, num_slices)
# Output: per-slice 3-channel image (256×256×3) + mask (256×256)
```

#### Step 1.3: Dataset Class (`src/data/dataset.py`)

```python
# What to implement:
# - ICHDataset(torch.utils.data.Dataset)
#   - __init__(data_dir, split, transform, context_slices=2)
#   - __getitem__(idx) → returns:
#       - image: (3, 256, 256) float tensor (multi-window)
#       - image_flipped: (3, 256, 256) horizontally flipped
#       - mask: (256, 256) int tensor (0=bg, 1-6=hemorrhage types)
#       - instance_mask: (256, 256) int tensor (instance IDs)
#       - hv_maps: (2, 256, 256) horizontal/vertical distance maps
#       - metadata: dict with patient_id, slice_idx, etc.
#
# Pre-compute HV maps from instance masks at preprocessing time:
# - For each instance, compute centroid
# - H-map: normalized horizontal distance from each pixel to centroid
# - V-map: normalized vertical distance from each pixel to centroid
```

#### Step 1.4: Augmentations (`src/data/augmentations.py`)

```python
# What to implement using albumentations:
# - Random rotation ±15° (p=0.5)
# - Elastic deformation α=50, σ=5 (p=0.3)
# - Random scale 0.85-1.15× (p=0.3)
# - Intensity shift ±10 HU (p=0.5)
# - Gaussian noise σ=0.02 (p=0.3)
# - Random crop + resize 224→256 (p=0.5)
# - CoarseDropout 1-3 patches, 16-32px (p=0.2)
#
# IMPORTANT: Apply SAME transform to image, mask, instance_mask, and hv_maps
# Use albumentations' additional_targets for this
```

#### Step 1.5: DataModule (`src/data/datamodule.py`)

```python
# What to implement:
# - ICHDataModule(pl.LightningDataModule)
#   - 60/20/20 patient-level split (49/16/16 patients)
#   - CRITICAL: Split by PATIENT, not by slice (avoid data leakage)
#   - train_dataloader: batch_size=4, shuffle=True, num_workers=4
#   - val_dataloader: batch_size=4, shuffle=False
#   - test_dataloader: batch_size=4, shuffle=False
#   - Optional: 5-fold cross-validation setup
```

---

### Phase 2: Core Model (Modules 2-4) — Week 2

#### Step 2.1: Dual-Stream Backbone (`src/models/backbone.py`)

```python
# What to implement:
# - DualStreamSwinV2(nn.Module)
#   - Load pretrained Swin-T V2 from timm:
#       self.swin = timm.create_model('swinv2_tiny_window8_256',
#                                      pretrained=True, features_only=True)
#   - SINGLE backbone with shared weights (not two copies)
#   - forward(x_orig, x_flip):
#       f_orig = self.swin(x_orig)  # list of 4 feature maps
#       f_flip = self.swin(x_flip)  # same backbone, shared weights
#       return f_orig, f_flip
#
# Output shapes (for 256×256 input):
#   F1: (B, 96,  64, 64)   — 1/4 scale
#   F2: (B, 192, 32, 32)   — 1/8 scale
#   F3: (B, 384, 16, 16)   — 1/16 scale
#   F4: (B, 768, 8,  8)    — 1/32 scale
#
# Modify first conv if using 2.5D (15ch→3ch projection layer before backbone)
```

#### Step 2.2: Symmetry Cross-Attention (`src/models/symmetry.py`)

```python
# What to implement:
# - SymmetryCrossAttention(nn.Module) — single scale
#   - __init__(dim, num_heads=8)
#   - Multi-head cross-attention: Q=orig, K=V=flip
#   - Difference gating: g = sigmoid(W @ |F_orig - F_flip|)
#   - Gated fusion: out = (1-g) * F_orig + g * cross_attn_output
#
# - SymmetryModule(nn.Module) — all 4 scales
#   - Apply SymmetryCrossAttention at each scale independently
#   - dims = [96, 192, 384, 768] for Swin-T
#   - forward(f_orig_list, f_flip_list) → f_sym_list (4 tensors)
#
# Key implementation detail:
# - Reshape (B, C, H, W) → (B, H*W, C) for attention
# - Reshape back after attention
```

#### Step 2.3: Feature Pyramid Network (`src/models/fpn.py`)

```python
# What to implement:
# - FPN(nn.Module)
#   - Lateral connections: 1×1 conv for each scale → 256 channels
#     lateral_convs = [Conv2d(96→256), Conv2d(192→256), Conv2d(384→256), Conv2d(768→256)]
#   - Top-down pathway: upsample + add
#     P4 = lateral_4(F4)
#     P3 = lateral_3(F3) + upsample_2x(P4)
#     P2 = lateral_2(F2) + upsample_2x(P3)
#     P1 = lateral_1(F1) + upsample_2x(P2)
#   - Output convs: 3×3 conv on each P level
#   - Output: [P1, P2, P3, P4] all with 256 channels
```

---

### Phase 3: Panoptic Head + HoVer (Modules 5-6) — Week 3-4

#### Step 3.1: Pixel Decoder (`src/models/panoptic_head.py`)

```python
# What to implement:
# - PixelDecoder(nn.Module) — Multi-Scale Deformable Attention
#   - Takes FPN outputs [P1, P2, P3, P4]
#   - 3 layers of multi-scale deformable attention
#   - Outputs per-pixel embeddings: (B, 256, H/4, W/4)
#
# Option A: Use detectron2's MSDeformAttn implementation
# Option B: Custom implementation using torch deformable attention
#
# If detectron2 is problematic on Windows, use simplified version:
#   - Standard multi-scale attention with bilinear sampling
#   - Progressive upsampling with skip connections
```

#### Step 3.2: Transformer Decoder (`src/models/panoptic_head.py`)

```python
# What to implement:
# - TransformerDecoder(nn.Module)
#   - N = 50 learnable object queries: nn.Embedding(50, 256)
#   - 9 decoder layers, cycling through 3 resolution levels (P4→P3→P2) × 3
#   - Each layer:
#       1. Self-attention among queries
#       2. MASKED cross-attention (queries attend to feature map,
#          but restricted to predicted foreground from previous layer)
#       3. FFN (2-layer MLP with ReLU)
#
# - ClassificationHead: MLP → (N, 7) — 6 types + background
# - MaskEmbeddingHead: MLP → (N, 256) — per-query mask embeddings
# - Mask prediction: dot product of mask embeddings with pixel embeddings
#   masks = einsum('bqc, bchw -> bqhw', mask_emb, pixel_emb)
#
# Masked Cross-Attention implementation:
#   - Get predicted mask from previous layer
#   - Threshold at 0.5 to create binary attention mask
#   - In cross-attention, set attention weights to -inf where mask=0
```

#### Step 3.3: HoVer Distance Branch (`src/models/hover_branch.py`)

```python
# What to implement:
# - HoVerBranch(nn.Module)
#   - Input: per-pixel embeddings (B, 256, H/4, W/4)
#   - Conv 3×3 (256→128) + BN + ReLU
#   - Conv 3×3 (128→64) + BN + ReLU
#   - Two heads:
#     - Conv 1×1 (64→1) → H-map (horizontal distances)
#     - Conv 1×1 (64→1) → V-map (vertical distances)
#   - Output: (B, 2, H/4, W/4) — concatenated H and V maps
#
# Post-processing (at inference):
#   - Compute Sobel gradients of H/V maps
#   - High gradient = instance boundary
#   - Use watershed algorithm to refine Mask2Former masks
```

---

### Phase 4: Text Encoder + Cross-Modal Attention (Module 7) — Week 5

#### Step 4.1: Clinical Text Descriptions (`data/text_prompts/hemorrhage_descriptions.json`)

```json
{
  "intraventricular": "Hyperdense fluid collection within the ventricular system, often with fluid-fluid levels and ventricular enlargement. Appears bright white inside ventricles.",
  "intraparenchymal": "Well-defined hyperdense lesion within brain parenchyma, round or oval shape, surrounded by low-density perilesional edema ring creating a halo effect.",
  "subarachnoid": "Hyperdense material filling sulci and cisterns, following contour of brain surface. Star-shaped pattern in basal cisterns. Thin linear densities.",
  "epidural": "Biconvex lens-shaped hyperdense collection between skull inner table and dura mater. Does not cross suture lines. Smooth inner margin.",
  "subdural": "Crescent-shaped hyperdense collection along brain convexity, crosses suture lines, follows dural reflections. May cause midline shift.",
  "ambiguous": "Ill-defined hyperdense region with mixed density patterns, unclear boundaries between adjacent hemorrhage types. May represent evolving bleed."
}
```

#### Step 4.2: Text Encoder (`src/models/text_encoder.py`)

```python
# What to implement:
# - TextEncoder(nn.Module)
#   - Load BiomedCLIP from open_clip:
#       model, _, preprocess = open_clip.create_model_and_transforms(
#           'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
#       self.text_encoder = model.text  # just the text tower
#   - Tokenize 6 descriptions → 6 token sequences
#   - Encode → 6 × hidden_dim (typically 512 for BiomedCLIP)
#   - Linear projection → 6 × 256 (match visual dim)
#   - Output: T ∈ ℝ^(6×256) — fixed text embeddings
#
# Training strategy:
#   - Phase 1: text encoder frozen (use as fixed features)
#   - Phase 2: fine-tune text encoder with LR=5e-6 (10× smaller than heads)
#
# If BiomedCLIP not available, fallback:
#   - Use PubMedBERT: 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
#   - Tokenize + encode + [CLS] pooling + linear projection
```

#### Step 4.3: Cross-Modal Attention (`src/models/cross_modal_attention.py`)

```python
# What to implement:
# - CrossModalAttention(nn.Module)
#   - __init__(visual_dim=256, text_dim=256, num_heads=8)
#   - Q = Linear(visual_features)     # (B, H*W, 256)
#   - K = Linear(text_embeddings)      # (6, 256)
#   - V = Linear(text_embeddings)      # (6, 256)
#   - Attention: softmax(Q @ K.T / sqrt(d)) @ V  → (B, H*W, 256)
#   - Residual: output = visual + alpha * text_guided
#   - alpha: learnable scalar, initialized to 0.1
#   - LayerNorm at the end
#
# Key: text embeddings are shared across ALL spatial positions
# Each pixel independently attends to all 6 type descriptions
```

---

### Phase 5: Losses & Training (Weeks 5-6)

#### Step 5.1: Loss Functions (`src/losses/`)

```python
# focal_loss.py:
# - FocalLoss(gamma=2.0, alpha=0.25)
# - Applied to class predictions after Hungarian matching

# dice_loss.py:
# - BinaryDiceLoss — per-mask dice
# - Applied to predicted masks vs GT masks

# hover_loss.py:
# - MSE loss on H/V distance maps
# - Only computed on hemorrhage pixels (masked MSE)

# contrastive_loss.py (NEW):
# - InfoNCE loss for text-visual alignment
# - For each detected hemorrhage region:
#     - Pool visual features → visual embedding
#     - Compute similarity with all 6 text embeddings
#     - Positive pair = matching hemorrhage type text
#     - Negative pairs = other 5 type texts
#   - L = -log(exp(sim(v_i, t_pos)/τ) / Σ_j exp(sim(v_i, t_j)/τ))
#   - Temperature τ = 0.07 (learnable)

# combined_loss.py:
# - CombinedLoss — weighted sum
#   L_total = 2.0*L_cls + 5.0*L_dice + 5.0*L_mask_focal
#           + 1.0*L_hv + 1.0*L_deep
#           + 0.5*L_contrastive + 0.3*L_text_cls
```

#### Step 5.2: Hungarian Matching (`src/utils/hungarian.py`)

```python
# What to implement:
# - Match N=50 predictions to M ground truth instances
# - Cost matrix: C(i,j) = λ_cls * L_cls(pred_i, gt_j)
#                        + λ_mask * L_mask(pred_i, gt_j)
# - Use scipy.optimize.linear_sum_assignment
# - Unmatched predictions → "no object" (background class)
# - Apply losses only on matched pairs
```

#### Step 5.3: Training Module (`src/training/trainer.py`)

```python
# What to implement:
# - SymPanICHNetModule(pl.LightningModule)
#   - __init__: build full model + losses
#   - forward: full forward pass through all 8 modules
#   - training_step: compute all losses, log to wandb
#   - validation_step: compute metrics (PQ, Dice, IoU)
#   - configure_optimizers:
#       - AdamW with weight_decay=0.05
#       - Differential LR:
#           backbone params: 1e-5
#           text encoder params: 5e-6
#           head params: 1e-4
#       - Poly LR scheduler (power=0.9)
#       - Linear warmup for 500 steps
#   - on_train_epoch_start: handle training phase transitions
#       Phase 1 (ep 1-5):   freeze backbone + text encoder
#       Phase 2 (ep 6-60):  unfreeze all
#       Phase 3 (ep 61-90): lower LRs
#       Phase 4 (ep 91-100): freeze segmentation, train report gen
```

---

### Phase 6: Model Assembly — Week 6

#### Step 6.1: Full Model (`src/models/sympanich_net.py`)

```python
# What to implement:
# - SymPanICHNetV2(nn.Module) — assembles all modules
#   - self.context_proj = Conv2d(15, 3, kernel_size=1)  # 2.5D projection
#   - self.backbone = DualStreamSwinV2()                 # Module 2
#   - self.symmetry = SymmetryModule()                   # Module 3
#   - self.fpn = FPN()                                   # Module 4
#   - self.pixel_decoder = PixelDecoder()                # Module 5 part 1
#   - self.cross_modal = CrossModalAttention()           # Text fusion
#   - self.transformer_decoder = TransformerDecoder()    # Module 5 part 2
#   - self.hover_branch = HoVerBranch()                  # Module 6
#   - self.text_encoder = TextEncoder()                  # Module 7
#   - self.report_generator = ReportGenerator()          # Module 8
#
#   - forward(images, images_flipped, text_descriptions=None):
#       # Module 2: Extract features
#       f_orig, f_flip = self.backbone(images, images_flipped)
#       # Module 3: Symmetry fusion
#       f_sym = self.symmetry(f_orig, f_flip)
#       # Module 4: FPN
#       pyramid = self.fpn(f_sym)
#       # Module 7: Text encoding
#       text_emb = self.text_encoder(text_descriptions)
#       # Module 5: Panoptic head with text fusion
#       pixel_emb = self.pixel_decoder(pyramid)
#       pixel_emb = self.cross_modal(pixel_emb, text_emb)
#       class_pred, mask_pred = self.transformer_decoder(pixel_emb, pyramid)
#       # Module 6: HoVer maps
#       hv_maps = self.hover_branch(pixel_emb)
#       return class_pred, mask_pred, hv_maps, text_emb
```

---

### Phase 7: AI Report Generator (Module 8) — Week 7

#### Step 7.1: Report Generator (`src/models/report_generator.py`)

```python
# What to implement:
# - ReportGenerator(nn.Module)
#   - This module is a STRUCTURED report builder, not end-to-end generation
#
# Approach A: Template-based (simpler, more reliable)
#   - Extract structured info from predictions:
#       - hemorrhage types detected (class predictions)
#       - location (mask centroid + brain atlas mapping)
#       - size (mask area in mm²)
#       - severity (volume + midline shift + type count)
#   - Fill structured template with extracted values
#   - Use text similarity with text embeddings for appearance matching
#
# Approach B: LLM-based (more natural, but heavier)
#   - Load BioMistral-7B with 4-bit quantization + LoRA
#   - Create structured prompt from predictions
#   - Generate free-text report
#   - Only train in Phase 4 (epochs 91-100) with frozen segmentation
#
# RECOMMENDATION: Start with Approach A, add Approach B later
# Approach A works without GPU-intensive LLM and is more deterministic
```

#### Step 7.2: Severity Classifier

```python
# Rule-based severity from predictions:
# severity_score = 0
# severity_score += volume_score(total_hemorrhage_volume)  # 0/1/2
# severity_score += shift_score(midline_shift_mm)          # 0/1/2
# severity_score += type_count_score(num_types)            # 0/1/2
# severity_score += location_score(deepest_location)       # 0/1/2
#
# Mild: score 0-2 | Moderate: score 3-5 | Severe: score 6-8
```

---

### Phase 8: Post-Processing & Inference — Week 7-8

#### Step 8.1: Panoptic Fusion (`src/utils/panoptic_fusion.py`)

```python
# What to implement:
# 1. Filter predictions by confidence > 0.5
# 2. Threshold masks at 0.5 → binary
# 3. If HoVer branch enabled:
#    - Compute Sobel gradients of H/V maps
#    - Find instance boundaries (high gradient regions)
#    - Apply watershed to refine mask boundaries
# 4. Sort predictions by confidence (descending)
# 5. Assign pixels to highest-confidence prediction (greedy assignment)
# 6. Connected components to split disconnected regions
# 7. Remove instances with area < 50 px²
# 8. Output: panoptic map (H, W) where each pixel = class_id * 1000 + instance_id
```

#### Step 8.2: Inference Pipeline (`scripts/predict.py`)

```python
# Full inference pipeline:
# 1. Load NIfTI volume
# 2. For each slice:
#    a. Apply multi-window extraction
#    b. Create 2.5D context
#    c. Create flipped copy
#    d. Forward pass through model
#    e. Panoptic fusion
#    f. Generate AI report (if requested)
# 3. Stack results → 3D panoptic volume
# 4. Save results + reports
```

---

## 📊 Evaluation & Metrics

### Metrics Implementation (`src/utils/metrics.py`)

```python
# What to implement:
# 1. Panoptic Quality (PQ):
#    PQ = (Σ IoU(p,g)) / (|TP| + 0.5*|FP| + 0.5*|FN|)
#    where TP = matched pairs with IoU > 0.5
#
# 2. Segmentation Quality (SQ) = mean IoU of matched pairs
# 3. Recognition Quality (RQ) = TP / (TP + 0.5*FP + 0.5*FN)
# 4. Per-class Dice = 2*|P∩G| / (|P|+|G|)
# 5. Per-class IoU = |P∩G| / |P∪G|
# 6. Size-stratified metrics (small<100px, medium<1000px, large>1000px)
```

### Evaluation Commands

```bash
# Train the model
python scripts/train.py --config configs/default.yaml

# Evaluate on test set
python scripts/evaluate.py --checkpoint best_model.ckpt --split test

# Run inference on a new scan
python scripts/predict.py --input path/to/scan.nii --output results/

# Generate AI report
python scripts/generate_report.py --input path/to/scan.nii --output report.txt
```

---

## 🗓️ Implementation Timeline

| Week | Phase | What to Build | Deliverable |
|------|-------|---------------|-------------|
| **1** | Data Pipeline | CT windowing, 2.5D, dataset, augmentations | Working dataloader |
| **2** | Core Model | Backbone, SACA, FPN | Feature extraction pipeline |
| **3** | Panoptic Head | Pixel decoder, transformer decoder | Mask2Former head |
| **4** | Panoptic Head | Hungarian matching, mask prediction, HoVer | End-to-end segmentation |
| **5** | Text + Losses | Text encoder, cross-modal attention, all losses | Text-guided model |
| **6** | Assembly | Full model, training loop, first training run | Training pipeline |
| **7** | Report + Post | AI report generator, panoptic fusion, inference | Full v2 system |
| **8** | Eval + Polish | Metrics, ablations, comparison, documentation | Final results |

---

## ⚡ Quick Start (After Setup)

```bash
# Step 1: Preprocess the dataset
python scripts/preprocess.py --data_dir data/raw --output_dir data/processed

# Step 2: Train the model
python scripts/train.py --config configs/default.yaml --gpus 1 --max_epochs 100

# Step 3: Evaluate
python scripts/evaluate.py --checkpoint checkpoints/best.ckpt

# Step 4: Run inference + AI report
python scripts/predict.py --input new_scan.nii --report
```

---

## 🔬 Ablation Study Plan

Run each ablation for 50 epochs on fold-0, compare against full model:

| # | Experiment | What to Change | Expected Impact |
|---|-----------|----------------|-----------------|
| 1 | − Symmetry Module | Remove SACA, use only original stream | ↓ recall for subtle bleeds |
| 2 | − Multi-Window | Use single brain window (1ch×3) | ↓ subdural/epidural detection |
| 3 | − HoVer Branch | Remove distance maps | Adjacent bleeds merge |
| 4 | − 2.5D Context | Single slice (no neighbors) | ↓ volumetric consistency |
| 5 | − Masked Cross-Attn | Standard cross-attention | ↓ small bleed detection |
| 6 | − Text Encoder | Remove Module 7 + cross-modal | ↓ type-specific Dice |
| 7 | − AI Report | Remove Module 8 | No impact on segmentation |
| 8 | ResNet-50 vs Swin | Swap backbone | Quantify transformer benefit |
| 9 | CLIP vs BiomedCLIP | Swap text encoder | Medical vs general knowledge |
| 10 | Queries: 25/50/100 | Vary N | Find optimal query count |

---

## 💡 Key Implementation Notes

1. **Start simple, add complexity**: Get Module 1-5 working first, then add text (Module 7) and report (Module 8)
2. **Test each module independently**: Write unit tests before integrating
3. **Memory management**: Use gradient accumulation (4 steps) to simulate batch_size=16
4. **Mixed precision**: Always use `torch.cuda.amp` for 2× speedup
5. **Patient-level splits**: NEVER split by slice — always by patient to avoid leakage
6. **Wandb logging**: Log losses, metrics, and sample predictions every epoch
7. **Save checkpoints**: Save best model by PQ on validation set
8. **Windows compatibility**: Use `pathlib.Path` for all file paths, avoid Unix-specific commands
