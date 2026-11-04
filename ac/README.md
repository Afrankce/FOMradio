# A 基线与 Beam–Matérn 源代码

本目录提供四组实验的核心实现、冻结查询清单及部分训练 / 验证入口。`ac/` 是沿用的历史目录名。输入只包含目标 Tx、建筑高度和 Beam；监督路径与推理输入分开。

- `dataset.py`、`manifests/`：查询与训练监督接口、固定训练/验证/测试配置。
- `anchor_source.py`：有限锚点采样与方差归一化。
- `beam_matern_source.py`、`beam_scale_source.py`：Beam 概率与三尺度权重。
- `anchor_training.py`：四组共享的速度网络、源和优化器。
- `train_anchor.py`、`evaluate_anchor.py`：训练与验证入口，分别需要 CUDA、数据集和匹配的清单/输出路径；对外发布后尚未做完整 GPU 重训或测试集复现。
- `vendor/`：所需的神经算子和 Matérn 源实现；`tests/`：合成输入上的实现检查。
- `beam_statistics.json`：训练 Beam 梯度的 95% 分位尺度；`PLAN.md`：原始实验计划。

原实验在 PyTorch 2.5.1+cu121、NumPy 1.26.4、SciPy 1.13.1 环境下运行；结果图另需 Matplotlib。设置模块搜索路径后可从仓库根目录执行合成测试：

```bash
export PYTHONPATH="$PWD/ac:$PWD/ac/vendor"
python -m pytest ac/tests/test_anchor_source.py ac/tests/test_beam_scale_source.py ac/tests/test_dataset.py
```

PowerShell 可使用 `$env:PYTHONPATH="$PWD/ac;$PWD/ac/vendor"` 设置路径。

**复现边界：**模型检查点、数据、四组逐查询测试结果和测试集完整评估入口未随仓库提供。公开版的输入约束与源码身份哈希已调整，不能用于恢复原始服务器断点；完整端到端训练和评估尚需独立验证。原始 A 的历史逐查询记录保留在 `results/ac_two_seeds/raw/A_seed42/` 和 `A_seed43/`，其中 identity 保存生成时的源码哈希，不代表当前源码哈希。
