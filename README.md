# GP2F: Cross-Domain Graph Prompting with Adaptive Fusion of Pre-trained Graph Neural Networks

<p align="center">
    <a href="https://arxiv.org/abs/2602.11629" alt="arXiv">
        <img src="https://img.shields.io/badge/arXiv-2602.11629-Blue" /></a>
    <a href="https://icml.cc/" alt="Conference">
        <img src="https://img.shields.io/badge/ICML'26-purple" /></a>
    <a href="https://pytorch.org/" alt="PyTorch">
        <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>
</p>

Official implementation of [**GP2F: Cross-Domain Graph Prompting with Adaptive Fusion of Pre-trained Graph Neural Networks**](https://arxiv.org/abs/2602.11629), accepted by ICML 2026.

## Environment Setup

Create a conda environment:

```bash
conda create -n gp2f python=3.10 -y
conda activate gp2f
```

Install PyTorch and PyTorch Geometric according to your CUDA version. For example, with CUDA 12.8:

```bash
pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128

pip install torch-geometric==2.7.0
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

Install the other required packages:

```bash
pip install numpy==2.1.2 tqdm==4.67.1 torchmetrics==1.8.2
```

## How to Run

Run all commands under the project root:

```bash
cd GP2F
```

For node classification:

```bash
bash scripts/run.sh
```

For graph classification:

```bash
bash scripts/graph_run.sh
```

Experimental logs are saved to `result/`, pre-trained models are saved to `pretrained_model/`, and sampled few-shot splits are saved to `sample_data/`.

## Citation

```
@article{he2026gp2f,
  title={GP2F: Cross-Domain Graph Prompting with Adaptive Fusion of Pre-trained Graph Neural Networks},
  author={He, Dongxiao and Sun, Wenxuan and Huang, Yongqi and Zhao, Jitao and Jin, Di},
  journal={arXiv preprint arXiv:2602.11629},
  year={2026}
}
```
