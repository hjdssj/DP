# Adaptive Gradient Clipping Implementation Notes

## 概述

本项目实现了MNIST分类任务的差分隐私训练，重点是**自适应梯度裁剪**——根据梯度分布动态调整裁剪阈值C，以最小化梯度估计的MSE（均方误差）。

## 文件说明

| 文件 | 说明 |
|------|------|
| `minst_baseline.py` | 标准DP-SGD实现（使用Opacus），固定C |
| `minst_adaptive_histogram.py` | 基于Opacus的自适应裁剪，尝试用hook捕获梯度 |
| `minst_adaptive_dp_manual.py` | 手动实现DP-SGD，完全控制梯度处理流程 |

## 核心挑战：获取真实梯度范数

自适应裁剪的关键是知道**裁剪前**的真实梯度分布。但存在以下问题：

### Opacus版本的Hook顺序问题

```
backward pass 执行顺序:
1. 真实梯度生成
2. Opacus裁剪hook（修改grad_sample）
3. 我们的hook（只能看到裁剪后的值）
```

**结果**：在`minst_adaptive_histogram.py`中，即使注册了hook，也只能观测到被Opacus裁剪后的梯度范数（约等于C），无法用于准确的MSE计算。

### 手动DP-SGD的挑战

`minst_adaptive_dp_manual.py`尝试手动实现DP-SGD以获得完全控制：

**优点**：
- 可以观测真实（未裁剪）的梯度范数
- 完全控制裁剪和噪声添加

**问题**：
- Per-sample梯度裁剪：梯度范数呈重尾分布（大多数~1，少数~100000+），导致裁剪引入巨大偏差
- Batch-level裁剪：训练变得不稳定（即使禁用DP）

## MSE公式

自适应裁剪的核心是最小化：
```
MSE = Bias² + Variance
```

- **Bias²**：裁剪引入的误差，bias ≈ E[(||g|| - C)²] for ||g|| > C
- **Variance**：DP噪声方差，var ≈ (C × σ)²

## 当前最佳配置

```bash
# 无DP的baseline
python minst_adaptive_dp_manual.py -n 10 -b 64 --disable-dp
# 准确率: ~98%

# 使用Opacus（固定C=1.0效果最好）
python minst_baseline.py -n 10 -b 64 --sigma 1.0 -c 1.0
# 准确率: ~90%，ε≈0.5

# 手动DP-SGD（小sigma）
python minst_adaptive_dp_manual.py -n 5 -b 64 --sigma 0.01 -c 10.0 --lr 0.01
# 准确率: ~98%但隐私保护弱
```

## 关键发现

1. **C=0.4 是固定裁剪的最优值**（非自适应），在Opacus实现下达到93.60%准确率
2. **Per-sample梯度重尾分布**使自适应裁剪困难
3. **手动DP-SGD**在数学上正确但数值不稳定
4. **σ和C的权衡**：大σ小C vs 小σ大C

## 未来改进方向

1. **修改Opacus源码**：在裁剪hook之前插入观测点
2. **使用Vectorized PGD**：批量计算per-sample梯度
3. **改进MSE估计**：考虑重尾分布特性
4. **混合方案**：先用小batch获取分布，再用大batch训练

## 运行示例

```bash
# 查看帮助
python minst_adaptive_histogram.py --help
python minst_adaptive_dp_manual.py --help

# 运行测试
python minst_baseline.py -n 10 -b 64 --sigma 1.0 -c 0.4
```
