# CWOD

本仓库是论文 **A Framework for Unsupervised Cross-Weather Object Detection of Pears in Complex Environments** 的代码发布版本。

CWOD 是一个基于 YOLO11 的跨天气无监督目标检测框架，面向梨果在雨天、雾天和低照度环境下的鲁棒检测问题。

## 当前仓库包含

- `BSD.yaml`：包含 `C3k2_BSD` 模块的模型配置
- `CWOD.py`：融合版 FSC + CMM 训练脚本
- `ultralytics/`：本地研究代码实现

## 主要结果

- Rainy: AP@50 从 `71.5` 提升到 `78.6`
- Foggy: AP@50 从 `79.3` 提升到 `84.0`
- Dark: AP@50 从 `71.0` 提升到 `78.1`

## 安装

```bash
git clone https://github.com/taro407/CWOD.git
cd CWOD
pip install -e .
```

## 训练入口

```bash
python CWOD.py --model-cfg BSD.yaml --source-data path/to/source.yaml --target-data path/to/target.yaml
```

完整英文说明请参考 [README.md](README.md)。
