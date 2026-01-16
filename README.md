# DualWave-Unet: Reimagining 3D U-Net with Wavelet-Preserved Encoding and Wave-MLP-Driven Adaptive Fusion for Brain MRI

## 🔍 Overview
**DualWave-Unet** is a lightweight 3D segmentation framework integrating frequency-domain processing (Wavelets) with phase-aware attention (Wave-MLP), the model preserves critical anatomical details while drastically reducing parameter count.

---

## ✨ Key Contributions
1. **Information-Preserving Attention:** Uses 3D Haar wavelet decomposition to separate volumes into eight frequency subbands (LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH), and combine them using attention fusion in Encoder blocks, ensuring no critical data is discarded.
2. **Phase-Aware Token Mixing (PATM3D):** Models spatial locations as waves with learnable amplitude and phase, enabling efficient global context understanding across depth, height, and width.
3. **Three-Way Attention Fusion:** A unique decoder fusion that balances learned upsampling, high-resolution skip connections, and wavelet reconstruction based on local anatomical characteristics.
---

## 📊 Results
DualWave-Unet achieves clinical-grade performance with a fraction of the parameters of state-of-the-art (SOTA) models.

**BraTS 2020 Dataset Results**:
| Model | ET (Dice) | TC (Dice) | WT (Dice) | AVG Dice | Params (M) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| CKD-TransBTS | 78.72 | 83.77 | 87.48 | 83.32 | 82.28 |
| Swin-UNETR | 78.53 | 83.51 | 89.80 | 83.95 | 62.19 |
| DynUNET | 78.17 | 85.44 | 90.05 | 84.55 | 24.93 |
| **DualWave-Unet (Ours)** | **79.52** | **85.01** | **90.73** | **85.08** | **3.65** |

---
## Dataset
We use the **BraTS2020** and **BraTS2021** datasets, which provide multimodal MRI volumes of brain tumors with expert-annotated ground truth.

| Dataset | #Cases | Modalities | Resolution | Labels |
|----------|---------|-------------|-------------|---------|
| BraTS2021 | 1251 | T1, T1Gd, T2, FLAIR | 240×240×155 | 4 |
| BraTS2020 | 369 | T1, T1Gd, T2, FLAIR | 240×240×155 | 4 |
| BraTS2020 | 285 | T1, T1Gd, T2, FLAIR | 240×240×155 | 4 |

Each case includes four co-registered MRI modalities and voxel-wise annotations for:
- **ET (Enhancing Tumor)**
- **TC (Tumor Core)**
- **WT (Whole Tumor)**

Datasets are available at:  
- [BraTS 2021 - CBICA](https://www.med.upenn.edu/cbica/brats2021/data.html)
- [BraTS 2020 - CBICA](https://www.med.upenn.edu/cbica/brats2020/data.html)  
- [BraTS 2018 - CBICA](https://www.med.upenn.edu/sbia/brats2018/data.html)
---

## Preprocessing
All MRI volumes are:
1. **Skull-stripped**, **co-registered**, and **resampled** to 1 mm³ isotropic resolution.  
2. **Cropped** to 128×128×128 using bounding boxes around tumor regions.  
3. **Intensity normalized** per modality via MONAI transforms.  
4. **Augmented** during training:
   - Random flipping (axial/sagittal/coronal)
   - Random cropping and scaling
   - Random intensity perturbations

---

## Training Process
1. Launch training from `main.py` or `main.ipynb`.  
2. Configure paths to datasets, checkpoints, and output directories.  
3. Modify model parameters (e.g., number of blocks or window size).  
4. Training setup:
   - Optimizer: **AdamW**
   - Learning rate: `1e-4`
   - Weight decay: `1e-5`
   - Epochs: `200`
   - Loss: **Hybrid Dice + Cross-Entropy + Focal Tversky**
   - Hardware: NVIDIA Tesla P100 GPU

---

## Installation
- You should use notebook file `DualWave-Unet.ipynb` for easy setup and training.

## Pretrained Weights
You can download our pretrained model checkpoints from Google Drive:  
🔗 **[Download Weights (DualWave-UNet, BraTS2021)](https://gofile.io/d/qP0mSN)**  