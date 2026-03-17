# Transformer From Scratch (PyTorch)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

本项目严格遵循 Google 2017 年论文 [《Attention Is All You Need》](https://arxiv.org/abs/1706.03762)，**从零实现**完整的 Transformer 模型（编码器-解码器架构）。**无任何高层封装**（如 HuggingFace Transformers），所有模块均手动构建，并与论文公式一一对应。旨在深入理解大语言模型的核心基础。

## ✨ 核心亮点

- **全链路手写**：从词嵌入、位置编码到多头注意力、掩码机制、前馈网络、层归一化、残差连接，全部手动实现，无黑盒。
- **严格对齐论文**：
  - ✅ Scaled Dot-Product Attention & Multi-Head Attention（论文 3.2 节）
  - ✅ 6 层 Encoder + 6 层 Decoder 堆叠（论文 3.3 节）
  - ✅ Position-wise Feed-Forward Networks（论文 3.4 节）
  - ✅ 正弦余弦位置编码（论文 3.5 节）
  - ✅ Padding Mask 与 Sequence Mask（因果掩码）双机制
- **模块化设计**：每个组件独立封装（`EncoderLayer`、`DecoderLayer`、`MultiHeadAttention` 等），易于复用和二次开发。
- **开箱可测**：每个模块都附有测试函数（如 `test_encoder()`、`test_decoder()`），可直接运行验证。

## 🚀 快速开始

### 环境配置
推荐使用 conda 创建虚拟环境：
```bash
conda create -n transformer python=3.8
conda activate transformer
pip install torch numpy matplotlib
