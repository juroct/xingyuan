# GNNGLS: Graph Neural Network Guided Local Search

This repository contains the implementation of Graph Neural Network Guided Local Search (GNNGLS) for solving the Traveling Salesperson Problem (TSP) and its variant with Draft Limits (TSPDL).

## Overview

GNNGLS combines the power of Graph Neural Networks (GNNs) with traditional Guided Local Search (GLS) to efficiently solve combinatorial optimization problems. The method uses a GNN to learn edge importance, which is then used to guide the local search process.

The repository includes:
- Implementation of traditional algorithms (Nearest Neighbor, Insertion, Local Search, Guided Local Search)
- GNN models for edge property prediction
- Training and evaluation scripts
- Visualization tools
- Support for external solvers (Concorde, LKH)
- Extension to the TSP with Draft Limits (TSPDL) problem

## Installation

### Requirements
- Python ≥ 3.10
- PyTorch ≥ 2.2
- DGL ≥ 1.2
- NetworkX ≥ 3.2

### Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/proroklab/gnngls.git
cd gnngls

# Create and activate conda environment
conda env create -f environment.yml
conda activate gnngls

# Install the package
pip install -e .
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/proroklab/gnngls.git
cd gnngls

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Usage

### Basic Usage (TSP)

```python
import torch
from gnngls import (
    TSPInstance,
    nearest_neighbor,
    local_search,
    guided_local_search
)

# Create a TSP instance
instance = TSPInstance.generate_random(problem_size=20)

# Solve using nearest neighbor
nn_tour = nearest_neighbor(instance)

# Apply local search
ls_tour, ls_cost, _ = local_search(instance, nn_tour)

# Apply guided local search
t_limit = time.time() + 5.0  # 5 seconds time limit
gls_tour, gls_cost, _ = guided_local_search(instance, nn_tour, t_limit)
```

### TSPDL Usage

```python
import torch
from gnngls import (
    TSPDLInstance,
    nearest_neighbor_tspdl,
    local_search_tspdl,
    guided_local_search_tspdl
)

# Create a TSPDL instance
instance = TSPDLInstance.generate_random(
    problem_size=20,
    hardness='medium'
)

# Solve using nearest neighbor
nn_tour = nearest_neighbor_tspdl(instance)

# Apply local search
ls_tour, ls_cost, _ = local_search_tspdl(instance, nn_tour)

# Apply guided local search
t_limit = time.time() + 5.0  # 5 seconds time limit
gls_tour, gls_cost, _ = guided_local_search_tspdl(instance, nn_tour, t_limit)
```

### Using GNN-Guided Local Search

```python
import torch
from gnngls import (
    TSPDLInstance,
    TSPDLEdgeModel,
    nearest_neighbor_tspdl,
    gnn_guided_local_search_tspdl
)

# Create a TSPDL instance
instance = TSPDLInstance.generate_random(
    problem_size=20,
    hardness='medium'
)

# Load a trained model
model = TSPDLEdgeModel(
    in_dim=3,
    embed_dim=128,
    out_dim=1,
    n_layers=3,
    n_heads=8
)
model.load_state_dict(torch.load("models/tspdl/best_model.pt"))
model.eval()

# Solve using nearest neighbor
nn_tour = nearest_neighbor_tspdl(instance)

# Apply GNN-guided local search
t_limit = time.time() + 5.0  # 5 seconds time limit
gnn_gls_tour, gnn_gls_cost, _ = gnn_guided_local_search_tspdl(
    instance, model, nn_tour, t_limit
)
```

## 详细运行流程

本项目的完整运行流程包括数据准备、模型训练和评估应用三个阶段。以下是详细的步骤说明：

### 1. 数据准备

首先，我们需要生成TSPDL问题实例用于训练和测试：

```bash
# 创建数据目录
mkdir -p data/tspdl/train data/tspdl/val data/tspdl/test

# 生成训练集
python scripts/generate_tspdl_instances.py data/tspdl/train --problem_size 20 --n_instances 100 --hardness medium

# 生成验证集
python scripts/generate_tspdl_instances.py data/tspdl/val --problem_size 20 --n_instances 20 --hardness medium

# 生成测试集
python scripts/generate_tspdl_instances.py data/tspdl/test --problem_size 20 --n_instances 20 --hardness medium

# 创建实例列表文件
cp data/tspdl/train/instances.txt data/tspdl/train/train.txt
cp data/tspdl/val/instances.txt data/tspdl/val/val.txt
cp data/tspdl/test/instances.txt data/tspdl/test/test.txt
```

参数说明：
- `--problem_size`: 问题规模（节点数量）
- `--n_instances`: 生成的实例数量
- `--hardness`: 难度级别（easy, medium, hard）

### 2. 模型训练

接下来，我们使用生成的数据训练GNN模型：

```bash
# 创建模型目录
mkdir -p models/tspdl

# 训练GNN模型
python scripts/train_tspdl_gnn.py data/tspdl models/tspdl --embed_dim 128 --n_layers 3 --n_heads 8 --batch_size 32 --n_epochs 100
```

参数说明：
- `--embed_dim`: 嵌入维度
- `--n_layers`: GNN层数
- `--n_heads`: 注意力头数
- `--batch_size`: 批量大小
- `--n_epochs`: 训练轮数
- `--lr`: 学习率（默认0.001）

训练过程中，模型会定期在验证集上评估，并保存最佳模型到`models/tspdl/best_model.pt`。

### 3. 评估和应用

训练完成后，我们可以评估模型性能并应用于实际问题：

```bash
# 创建结果目录
mkdir -p results/tspdl

# 评估算法性能
python scripts/evaluate_tspdl.py data/tspdl/test results/tspdl --model_path models/tspdl/best_model.pt --time_limit 10.0
```

参数说明：
- `--model_path`: 模型路径
- `--time_limit`: 搜索算法的时间限制（秒）

评估结果会保存到`results/tspdl`目录，包括性能指标和可视化图表。

### 4. 在自己的代码中使用

您可以按照以下方式在自己的代码中使用训练好的模型：

```python
import torch
from gnngls import TSPDLInstance, TSPDLEdgeModel, gnn_guided_local_search_tspdl

# 创建实例
instance = TSPDLInstance.generate_random(problem_size=20, hardness='medium')

# 加载模型
model = TSPDLEdgeModel(
    in_dim=3,
    embed_dim=128,
    out_dim=1,
    n_layers=3,
    n_heads=8
)
model.load_state_dict(torch.load("models/tspdl/best_model.pt"))
model.eval()

# 生成初始解
from gnngls import nearest_neighbor_tspdl
initial_tour = nearest_neighbor_tspdl(instance)

# 应用GNN引导的局部搜索
import time
t_limit = time.time() + 5.0  # 5秒时间限制
tour, cost, _ = gnn_guided_local_search_tspdl(
    instance, model, initial_tour, t_limit
)

# 检查解决方案
from gnngls import TSPDLSolution
solution = TSPDLSolution(instance, tour)
print(f"成本: {solution.cost:.4f}, 可行: {solution.feasible}")
```

## Project Structure

- `gnngls/`: Main package
  - `models.py`: GNN model architecture for TSP
  - `algorithms.py`: TSP algorithms (nearest neighbor, local search, guided local search)
  - `operators.py`: Local search operators (2-opt, relocate)
  - `datasets.py`: Dataset handling
  - `tspdl.py`: TSPDL problem definition
  - `tspdl_algorithms.py`: TSPDL algorithms
  - `tspdl_models.py`: GNN models for TSPDL
  - `tspdl_gnn_gls.py`: GNN-guided local search for TSPDL
  - `tspdl_visualization.py`: Visualization tools for TSPDL
- `scripts/`: Scripts for training, evaluation, and demonstration
- `data/`: Directory for datasets
- `models/`: Directory for saved models
- `docs/`: Documentation

## External Solvers

The repository supports external solvers like Concorde and LKH for finding optimal solutions. See [docs/external_solvers.md](docs/external_solvers.md) for installation and usage instructions.

## Citation

If you use this code in your research, please cite the following paper:

```
@inproceedings{
    hudson2022graph,
    title={Graph Neural Network Guided Local Search for the Traveling Salesperson Problem},
    author={Hudson, Benjamin and Malencia, Matteo and Prorok, Amanda},
    booktitle={International Conference on Learning Representations},
    year={2022}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
