# GCAN: Generative Counterfactual Attention-guided Network  
**Explainable Cognitive Decline Diagnostics from fMRI Functional Connectivity**  

This repository contains the official implementation of **GCAN**, accepted at **MICCAI 2024**. GCAN leverages a generative counterfactual attention mechanism to enhance interpretability and diagnostic performance in Mild Cognitive Impairment (MCI) detection using fMRI-derived functional connectivity (FC) matrices.

---

## 🌐 Environment Setup

Install the required dependencies with the exact versions below:

```diff
torch == 1.13.1
numpy == 1.22.3
nibabel == 1.10.2
torchcam == 0.3.2
torchvision == 0.14.1
einops == 0.6.0
python == 3.9.0
imageio == 2.31.1
```
💡 We strongly recommend using a virtual environment (e.g., conda or venv) to ensure version compatibility.


### 🧠  Workflow Overview
GCAN operates in three sequential stages:

#### * Pretrain a baseline FC classifier (mode_net = 'pretrain')
#### * Generate counterfactual functional connectivity maps (mode_net = 'image_generator')
  * This step synthesizes FC matrices corresponding to a target diagnostic label.
  * The counterfactual map is obtained by:
     ```
        ΔFC = FC_target − FC_source
     ```
#### * Train a region-specific classifier (mode_net = 'region-specific')
Uses the counterfactual map as an attention prior to validate its contribution to MCI diagnosis.

⚠️ Note:
Functional connectivity (FC) matrices must be extracted offline using MATLAB (e.g., via SPM12 batch processing).
If your input FC matrix size differs from the default, you may need to adjust the kernel_size of the average pooling layer in ResNet to avoid shape mismatches.
### ▶️ Running the Model
#### 1. Generate k-Fold Cross-Validation Splits
Create stratified train/validation/test CSV files:

```
python generate_csv.py
```
This script outputs **fold-wise .csv** files used for reproducible evaluation.

#### 2. Pretrain the Baseline Classifier
In opt.py, set:
```
opt.mode_net = 'pretrain'
```
Run training:
```
python main.py
```
The model learns to classify cognitive status (e.g., NC vs. MCI) from raw FC inputs.

#### 3. Generate Counterfactual Attention Maps
In opt.py, set:
```
opt.mode_net = 'image_generator'
```
Run:
```
python main.py
```
This stage uses the pretrained classifier to guide the generation of target-label FC matrices. The difference between generated and original FC yields the counterfactual attention map, highlighting regions most influential for diagnostic reclassification.

#### 4. Train the Final Region-Specific Classifier
In **opt.py**, set:
```
opt.mode_net = 'region-specific'
```
Run:
```
python main.py
```
This classifier integrates the counterfactual map as an attention prior, enabling interpretable and performance-validated MCI diagnosis.

📁 Project Highlights
```
✅ Explainable: Counterfactual maps reveal brain regions driving diagnostic decisions.
✅ Modular: Each stage is independently configurable via opt.py.
✅ Reproducible: k-fold splits and version-locked dependencies ensure consistent results.
```
