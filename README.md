# CWOD

Official code for the paper **"A Framework for Unsupervised Cross-Weather Object Detection of Pears in Complex Environments"**.

CWOD is an unsupervised cross-weather pear detection framework built on top of YOLO11. It targets the practical problem that a detector trained on clear-weather orchard images degrades substantially when deployed in rainy, foggy, or dark environments.

## ✨ Highlights

- A **C3k2-BSD** backbone module improves local structure modeling under occlusion, scale variation, and cluttered orchard backgrounds.
- **FSC** performs channel-wise feature space correction to reduce domain shift between the clear-weather source domain and the adverse-weather target domain.
- **CMM** modulates spatial responses according to target-domain degradation cues to improve robustness under rain, fog, and low-light conditions.
- The method improves **AP by 7.1, 4.7, and 7.1 points** over the YOLO11 baseline on the rainy, foggy, and dark target domains, respectively.

## 📦 Repository Contents

- [BSD.yaml](BSD.yaml): YOLO11 model configuration with the proposed `C3k2_BSD` blocks.
- [CWOD.py](CWOD.py): fused FSC + CMM training script.
- [ultralytics](ultralytics): local research codebase derived from Ultralytics YOLO and extended for CWOD.

## ⚙️ Method Overview

CWOD combines one detector-side architectural change and two training-time domain adaptation modules:

1. **C3k2-BSD** replaces the original C3k blocks in YOLO11 with a dual-branch structure that mixes local channel interaction and neighborhood spatial modeling.
2. **FSC** aligns source and target features by correcting channel-wise statistical shifts with an exponential moving average of target-domain responses.
3. **CMM** builds a degradation-aware modulation mask from target-domain responses and injects it into intermediate feature maps during training.

FSC and CMM are only used during training for cross-domain adaptation. Inference keeps the original YOLO-style forward path.

## 📊 Results

| Target domain | Baseline (YOLO11) AP@50 | CWOD AP@50 | Gain |
| --- | ---: | ---: | ---: |
| Rainy | 71.5 | 78.6 | +7.1 |
| Foggy | 79.3 | 84.0 | +4.7 |
| Dark | 71.0 | 78.1 | +7.1 |

## 🚀 Installation

Create a Python environment and install the repository in editable mode:

```bash
git clone https://github.com/taro407/CWOD.git
cd CWOD
pip install -e .
```

The codebase follows the local `ultralytics` package in this repository, so running scripts from the repository root will use the research version bundled here.

## 🗂️ Data Preparation

The training script expects YOLO-format dataset YAML files for:

- the labeled **source** domain
- the unlabeled **target** domain

At minimum, prepare:

```text
path/to/source.yaml
path/to/target.yaml
```

Each YAML should follow the standard Ultralytics detection dataset format. The source-domain YAML is used for supervised detection training, while the target-domain YAML provides target-domain images for feature adaptation. If you use your own dataset, organize it according to the standard YOLO detection directory structure and prepare the corresponding YAML files.

## 🛠️ Training

### Train CWOD

```bash
python CWOD.py \
  --model-cfg BSD.yaml \
  --source-data path/to/source.yaml \
  --target-data path/to/target.yaml
```

## 🙏 Acknowledgment

This work is built on top of [Ultralytics YOLO](https://github.com/ultralytics/ultralytics). We thank the original authors for open-sourcing their codebase.
