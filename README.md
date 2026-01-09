[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/vboussot/KonfAI/blob/main/LICENSE)
[![Paper](https://img.shields.io/badge/📌%20Paper-KonfAI-blue)](https://arxiv.org/abs/2503.24121)

# 🔄 Slicer IMPACT-Reg

<img src="ImpactReg.png" alt="IMPACT reg Logo" width="300" align="right">

**Slicer IMPACT-Reg** is an open-source 3D Slicer extension dedicated to **multimodal medical image registration**.  
It integrates the **IMPACT similarity metric [1]** within the **Elastix** registration engine, bringing state-of-the-art deep semantic alignment directly into Slicer.

Powered by **KonfAI [2]**, the module provides the following features:

- Fully automated registration pipelines  
- GPU-accelerated feature extraction  
- Built-in quality assessment and visualization  
- Ensemble-based uncertainty quantification  

All within a clinically-friendly environment.

---

## 🖼️ User Interface

| IMPACT-Reg registration workflow | Registration evaluation panel |
|---------------------------------|-------------------------------|
|  <img src="docs/Registration.png" alt="Registration interface" width="100%"> | <img src="docs/Evaluation.png" alt="Registration interface" width="100%">|
| *Figure 1 – Multimodal registration interface.* | *Figure 2 – Evaluation with reference labels.* |


---

## ⚙️ Key Features

### 🧠 Deep semantic registration
- IMPACT: feature-space similarity from pretrained segmentation networks  
- Multi-preset execution enabling sequential refinement  
- GPU or CPU execution  
- Optional mask-constrained registration  

### 📊 Built-in evaluation and QA
- Landmark, segmentation, and intensity-based metrics  
- Automatic warped volume generation  
- 2D/3D synchronized visualization inside Slicer  

### 🔁 Ensemble-based robustness
- Multiple registration presets executed sequentially  
- Composite deformation field estimation  
- Average transform computation  

### 📉 Uncertainty quantification
- Analysis of the statistical variability of transforms  
- Automatic visualization of uncertainty volumes  
- JSON metrics export for downstream analysis  

---

## 🚀 Installation

Requires **3D Slicer ≥ 5.6**

### 1️⃣ Clone the KonfAI module
```bash
git clone https://github.com/vboussot/SlicerKonfai.git
```

### 2️⃣ Clone this repository
```bash
git clone https://github.com/vboussot/SlicerImpactReg.git
```

### 3️⃣ In Slicer  
Go to:
> **Edit → Application Settings → Modules → Additional Module Paths**

Add:
- `SlicerKonfai/KonfAI`  
- `SlicerImpactReg/ImpactReg`

### 4️⃣ Restart Slicer → open **IMPACT-Reg** 🎯

---

## 🧩 Presets & Models

Parameter maps and pretrained models are automatically downloaded from:  
📦 **VBoussot/ImpactReg** on Hugging Face Hub  

Each preset includes:
- Parameter maps for Elastix  
- Feature extractor models for IMPACT  
- A volume-dependent preprocessing function  

---

## 📚 References

1. **Boussot, V. et al.**  
   *IMPACT: A Generic Semantic Loss for Multimodal Medical Image Registration.*  
   arXiv:2503.24121 — 2025  

2. **Boussot, V. & Dillenseger, J-L.**  
   *KonfAI: A Modular and Fully Configurable Framework for Deep Learning in Medical Imaging.*  
   arXiv:2508.09823 — 2025  




