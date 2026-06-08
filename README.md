# TargetMolGraph：面向电解液分子的目标导向图生成算法

## 1. 项目简介

本项目面向图论课程大作业，提出一个结合图论算法与人工智能的电解液分子生成方案：将电解液候选溶剂/添加剂分子表示为带标签图，学习“分子结构–图–电化学性质”之间的映射关系，并进一步从目标性质范围出发，反向生成满足要求的候选电解液分子结构。

项目核心思想是：

> 用图论刻画电解液分子结构，用图神经网络学习结构–电化学性质关系，用图搜索算法从目标性质反向生成候选电解液分子。

该方案不是简单地把分子当作字符串处理，而是将电解液分子筛选问题转化为图空间中的结构建模、性质学习、约束搜索和结果验证问题。

### 1.1 电解液应用背景

锂电池、钠电池等二次电池的电解液需要同时满足多种约束：较宽的电化学稳定窗口、合适的前线轨道能级、较强的离子溶剂化能力、适中的分子量和结构复杂度。本项目使用 `data/Electrolytes.csv` 作为原始数据，其中包含电解液分子的 SMILES 与以下性质：

- `Es-Ea (eV)`：与电化学稳定/激发能相关的能量差指标；
- `LUMO_sol (eV)`：溶液环境下 LUMO 能级，用于刻画还原稳定性趋势；
- `HOMO_sol (eV)`：溶液环境下 HOMO 能级，用于刻画氧化稳定性趋势；
- `Dielectric constant of solvents`：溶剂介电常数，影响盐解离和离子传输。

代码会先为每个 SMILES 计算 RDKit 分子描述符，并生成 `data/processed/electrolytes_with_rdkit_descriptors.csv`。训练 GNN 时，原始 `Electrolytes.csv` 中的电解液性质会作为图级辅助特征，与 RDKit 描述符一起输入模型；为了避免预测某一性质时直接把答案作为输入，默认的 per-task GNN 会屏蔽当前预测目标，只保留其他电解液性质和 RDKit 描述符。若存在 `data/cluster_*.csv`，程序还会合并 4 个电解液簇的 cluster ID、PCA 坐标和二进制结构特征，并用它们进行更均衡的训练/验证/测试划分。

---

## 2. 选题名称

TargetMolGraph：面向电解液分子的目标导向图生成算法

---

## 3. 研究目标

本项目希望回答以下问题：

1. 如何将分子结构转化为图论中的带标签图？
2. 如何利用 RDKit 和公开数据集获得分子性质标签？
3. 如何利用图神经网络学习“分子图 → 分子性质”的映射关系？
4. 如何从目标性质区间出发，在图空间中搜索并生成候选分子？
5. 如何利用图论约束保证生成分子的合法性、连通性和非重复性？
6. 如何通过 RDKit 回算性质，验证生成结果是否满足目标要求？

最终形成一个完整闭环：

```text
公开分子数据集
      ↓
RDKit 计算性质标签
      ↓
SMILES 转换为分子图
      ↓
GNN 学习结构–性质关系
      ↓
给定目标性质区间
      ↓
图搜索生成候选分子
      ↓
图论约束检查与去重
      ↓
GNN 快速预测与筛选
      ↓
RDKit 回算性质验证
      ↓
输出 Top-K 分子结构与解释
```

---

## 4. 核心思路

本项目包括两个方向：

### 4.1 正向学习：结构 → 图 → 性质

首先将分子结构转换为分子图：

- 原子对应节点；
- 化学键对应边；
- 原子类型、价态、芳香性等作为节点特征；
- 键类型、是否共轭、是否在环中等作为边特征。

随后使用 RDKit 计算分子性质，例如：

- `MolLogP`
- `TPSA`
- `QED`
- `MolWt`
- `RingCount`
- `NumHAcceptors`
- `NumHDonors`
- `NumRotatableBonds`

由此构建监督学习数据集：

\[
\mathcal{D}=\{(G_i, \mathbf{y}_i)\}_{i=1}^{N}
\]

其中 \(G_i\) 是分子图，\(\mathbf{y}_i\) 是分子性质向量。

然后使用图神经网络学习映射：

\[
F_\theta:G\rightarrow \mathbf{y}
\]

即根据分子图预测分子性质。

### 4.2 反向生成：性质 → 图 → 结构

给定目标性质区间，例如：

```text
1.0 ≤ logP ≤ 2.5
40 ≤ TPSA ≤ 90
QED ≥ 0.60
MolWt ≤ 350
RingCount ≤ 2
```

项目将分子生成转化为图空间中的约束搜索问题：

\[
\text{Find } G^* \in \mathcal{G}, \quad \text{s.t. } \mathbf{y}(G^*) \in \Omega
\]

其中：

- \(\mathcal{G}\)：所有合法分子图构成的搜索空间；
- \(\Omega\)：目标性质区间；
- \(G^*\)：满足目标性质要求的候选分子图。

生成阶段从简单片段出发，通过添加节点、连接边、拼接子图、闭合环等图操作生成候选分子，并使用 GNN 快速预测性质，筛选最接近目标区间的结构。

---

## 5. 图论知识点关联

本项目的图论主线贯穿全流程。

### 5.1 图表示

分子被建模为带标签图：

\[
G=(V,E,\phi_V,\phi_E)
\]

其中：

- \(V\)：原子节点集合；
- \(E\)：化学键边集合；
- \(\phi_V\)：节点标签，如 C、O、N、F 等原子类型；
- \(\phi_E\)：边标签，如单键、双键、芳香键等。

这一过程将分子设计从字符串问题转化为图结构问题。

### 5.2 节点度约束

原子的价键规则可转化为节点度约束：

\[
deg(v)\leq Valence(v)
\]

例如：

```text
C ≤ 4
N ≤ 3
O ≤ 2
F ≤ 1
Cl ≤ 1
```

因此，化学合法性首先表现为图结构中的节点度合法性。

### 5.3 连通性检查

合法分子应为一个完整结构，即分子图应为连通图：

\[
c(G)=1
\]

可以利用 BFS 或 DFS 检查候选分子图是否只有一个连通分量。

### 5.4 回路与环秩

分子中的环结构对应图中的回路。对于连通图，独立环数量可由环秩表示：

\[
r(G)=|E|-|V|+1
\]

该指标可用于控制分子环结构数量和结构复杂度。

### 5.5 子图匹配

羰基、醚键、羟基、芳香环等官能团可以视为特定子图。通过子图匹配，可以判断候选分子是否包含目标官能团结构。

### 5.6 图同构与去重

不同生成路径可能得到同一个分子，这本质上是带标签图同构问题。实际实现中可以使用 canonical SMILES 进行规范化表示和重复结构去除。

### 5.7 子结构匹配与数据集驱动结构先验

在目标导向生成中，GNN 负责学习 `RE&WSE.csv` 中分子结构到目标性质的连续映射，但 GNN 本身并不显式说明哪些局部子结构支持这些性质。为增强生成方向，本项目引入子结构匹配问题：

\[
\text{Match}(G, \mathcal{M})=\max \sum_{(m,s)\in A} w_m
\]

其中 \(\mathcal{M}\) 是从 RE&WSE 目标成功分子中学习得到的 C/O/F 子结构集合，\(s\) 是候选分子中的实际子图匹配，\(A\) 是目标 motif slot 与实际子结构之间的匹配关系，\(w_m\) 是该 motif 在高性能分子中相对低性能分子的富集权重。

实现上，程序先根据目标区间对 RE&WSE 分子排序，将最接近目标的分子视为 target-success group，将偏离目标较大的分子视为 contrast group；随后统计 ether、carbonyl、ester、carbonate、acetal、fluoroalkyl、trifluoromethyl 等 C/O/F motif 在两组中的出现频率：

\[
\Delta_m=P(m\mid good)-P(m\mid bad)
\]

富集度越高，motif 权重越大。生成阶段对每个候选分子执行 RDKit SMARTS 子结构匹配，并通过小规模最大权匹配计算 `motif_match_score`。该分数作为软奖励加入 Beam Search：

```text
Score(G)
= GNN target-property score
+ RE&WSE-derived motif matching bonus
- complexity / ring penalties
```

因此，匹配并不替代 GNN，也不把 RDKit 当作目标性质计算器；它的作用是将 RE&WSE 中学到的结构先验注入搜索过程，使 Beam Search 更倾向于扩展包含目标相关子结构的候选分子。

### 5.8 图搜索与搜索树

分子生成过程可以看作图空间中的状态转移：

\[
G_{t+1}=T(G_t,a_t)
\]

其中 \(a_t\) 表示一次图操作，例如添加原子节点、连接化学键、拼接官能团子图或闭合环。

整个生成过程可以展开为搜索树，并通过 Beam Search 保留高分候选，避免搜索空间爆炸。

### 5.9 GNN 消息传递

GNN 通过邻域聚合学习图结构信息：

\[
h_v^{(k)}=\sigma\left(W_1h_v^{(k-1)}+W_2\sum_{u\in N(v)}h_u^{(k-1)}\right)
\]

一层 GNN 聚合一跳邻居信息，多层 GNN 捕捉多跳子图结构。最终通过 READOUT 得到图级表示，用于预测分子性质。

---

## 6. 方法流程

### Step 1：数据收集

可选数据集：

- ZINC 子集
- QM9
- ESOL
- FreeSolv
- 自建小分子 SMILES 集合

输入格式主要为 SMILES。

### Step 2：性质计算

使用 RDKit 计算分子性质标签：

```python
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

mol = Chem.MolFromSmiles(smiles)
logp = Descriptors.MolLogP(mol)
tpsa = Descriptors.TPSA(mol)
qed = QED.qed(mol)
molwt = Descriptors.MolWt(mol)
ring_count = Descriptors.RingCount(mol)
```

### Step 3：分子图构建

将 SMILES 转换为图结构：

- 节点：原子；
- 边：化学键；
- 节点特征：原子类型、度、形式电荷、芳香性、杂化方式；
- 边特征：键类型、是否在环中、是否共轭。

可使用 PyTorch Geometric 构建 `Data` 对象：

```python
Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, y=properties)
```

### Step 4：GNN 训练

可选模型：

- GCN
- GIN
- MPNN

推荐使用 GIN 或 GCN 作为课程设计版本。

模型输入为分子图，输出为性质预测值：

```text
分子图 G → GNN → 分子表示 h_G → MLP → 性质预测 y_hat
```

评价指标：

\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y_i}|
\]

\[
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y_i})^2}
\]

### Step 5：目标性质设定

设定目标性质区间，例如：

```text
1.0 ≤ logP ≤ 2.5
40 ≤ TPSA ≤ 90
QED ≥ 0.60
MolWt ≤ 350
RingCount ≤ 2
```

定义候选分子得分函数：

\[
Score(G)= -\sum_j w_j d(\hat{y}_j,[L_j,U_j])-\lambda P(G)
\]

其中 \(d\) 表示预测性质与目标区间之间的距离，\(P(G)\) 表示结构惩罚项。

### Step 6：目标导向图搜索生成

从初始片段出发：

```text
C, CC, CO, CN, benzene, carbonyl, ether, ester
```

每一步执行图操作：

```text
添加原子节点
连接化学键
拼接官能团子图
闭合小环
停止生成
```

采用 Beam Search：

```text
每一轮扩展候选图
→ 检查合法性
→ GNN 预测性质
→ RE&WSE 子结构匹配评分
→ 根据目标性质与 motif 匹配联合目标函数排序
→ 保留 Top-K
→ 继续扩展
```

### Step 7：合法性检查与去重

检查内容包括：

- 节点度是否满足价键约束；
- 分子图是否连通；
- 环数量是否过多；
- 是否包含必要官能团；
- 是否与已有分子重复；
- RDKit 是否能够 sanitize。

### Step 8：RDKit 回算验证

对最终 Top-K 候选分子重新计算性质，验证是否满足目标范围：

```text
GNN 预测性质 → RDKit 回算性质 → 判断是否满足目标区间
```

评价指标：

\[
SuccessRate=\frac{\#\{G_i:\mathbf{y}_i\in\Omega\}}{\#\{G_i\}}
\]

---

## 7. 预期结果

最终输出包括：

1. 分子图数据集；
2. RDKit 计算得到的性质标签；
3. 训练好的 GNN 性质预测模型；
4. 目标导向分子图生成算法；
5. Top-K 候选分子结构；
6. GNN 预测性质与 RDKit 回算性质对比；
7. 结构–性质关系解释；
8. 可视化图表与汇报材料。

示例输出表：

| Rank | SMILES | GNN logP | RDKit logP | GNN TPSA | RDKit TPSA | QED | 是否满足目标 |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | CCOC(=O)N | 1.82 | 1.76 | 68.4 | 70.1 | 0.72 | 是 |
| 2 | CCN(CC)C=O | 2.13 | 2.28 | 54.7 | 51.9 | 0.69 | 是 |
| 3 | CCOC1COC1 | 0.91 | 1.05 | 88.2 | 84.6 | 0.66 | 是 |

---

## 8. 实验设计

### 实验一：结构–性质预测实验

目的：验证 GNN 是否能够学习分子图与性质之间的映射关系。

方法：

- 使用公开分子数据集；
- RDKit 计算性质标签；
- 训练 GNN 预测 logP、TPSA、QED 等性质；
- 使用 MAE、RMSE 评价预测效果。

可对比方法：

- Morgan Fingerprint + Random Forest；
- 传统图特征 + MLP；
- GNN 模型。

### 实验二：目标导向生成实验

目的：验证图搜索生成算法能否生成满足目标性质范围的分子。

方法：

- 给定目标性质区间；
- 使用 Beam Search 生成候选图；
- 使用 GNN 快速筛选；
- 使用 RDKit 回算验证；
- 统计成功率。

### 实验三：消融实验

目的：证明图论约束和 GNN 筛选的作用。

对比设置：

| 方法 | GNN 筛选 | 图论约束 | 预期表现 |
|---|---|---|---|
| 随机生成 | 否 | 弱 | 非法率高，成功率低 |
| 仅图论约束 | 否 | 强 | 合法性高，但目标命中率一般 |
| GNN + 弱约束 | 是 | 弱 | 目标性增强，但可能生成非法结构 |
| GNN + 完整图论约束 | 是 | 强 | 合法性和目标命中率均较好 |

---

## 9. 可视化内容

建议展示以下图表：

1. 总体流程图；
2. 分子结构到分子图的转换示意图；
3. GNN 消息传递示意图；
4. Beam Search 分子生成树；
5. 目标性质区间雷达图；
6. Top-K 分子结构图；
7. GNN 预测值与 RDKit 回算值对比图；
8. 生成分子的性质分布图；
9. 图结构变化与性质变化的解释图。

---

## 10. 项目特色

本项目的特色可以概括为：

1. **图论驱动**：分子表示、合法性约束、结构搜索、去重和解释全部建立在图论方法上。
2. **AI 赋能**：使用 GNN 学习复杂的结构–性质关系，为生成过程提供快速性质评估。
3. **目标导向**：不是随机生成分子，而是从目标性质区间出发反向搜索候选结构。
4. **验证闭环**：最终使用 RDKit 回算性质，避免只依赖模型预测。
5. **可解释性强**：能够从节点、边、子图、路径、回路等角度解释结构变化对性质的影响。

一句话总结：

> 图论刻画分子结构，GNN 学习结构–性质关系，图搜索反向生成满足目标性质的新分子。

---

## 11. 半页 PPT 精简版内容

### 选题

**基于图神经网络与目标导向图搜索的分子生成算法**

### 研究内容

将分子表示为带标签图，利用 RDKit 计算分子性质标签，训练 GNN 学习“分子图 → 性质”的映射关系；随后给定目标性质区间，通过 Beam Search 在分子图空间中生成候选结构，并结合图论约束、GNN 筛选和 RDKit 回算验证，得到满足目标性质要求的新分子。

### 关联图论知识点

**图表示：** 将分子建模为带标签图 \(G=(V,E,\phi_V,\phi_E)\)，原子为节点、化学键为边，将分子设计转化为图空间中的结构生成问题。

**图约束：** 用节点度约束对应原子价键规则，用连通性保证完整分子，用回路/环秩刻画环结构，用子图匹配识别官能团，用图同构去除重复结构。

**图搜索：** 分子生成可视为图状态转移，通过添加节点、连接边、拼接子图形成搜索树，并结合 Beam Search 在庞大离散空间中筛选候选结构。

**图学习：** GNN 通过邻域聚合学习局部子图、路径关系与全局拓扑对性质的影响，建立“分子图 → 性质”的映射。

**核心意义：** 图论不仅是分子的表示方法，更是合法性判断、目标生成、结构去重和性质解释的基础。

---

## 12. 后续可扩展方向

1. 将目标性质从 RDKit 描述符扩展到量子化学性质，如 HOMO、LUMO、偶极矩等；
2. 引入更复杂的图生成模型，如 GraphVAE、GraphRNN、GraphAF 或扩散模型；
3. 将分子生成目标迁移到电解液、药物分子、聚合物单体等具体应用场景；
4. 引入多目标优化，平衡性能、合成可及性和结构复杂度；
5. 结合可视化界面，实现交互式分子图搜索与性质筛选。

---

## 13. 当前项目文件结构

```text
TargetMolGraph/
├── README.md
├── data/
│   ├── Electrolytes.csv
│   ├── cluster_*.csv
│   ├── processed/
│   │   ├── electrolytes_with_rdkit_descriptors.csv
│   │   ├── electrolyte_graph_dataset.pt
│   │   └── electrolyte_graph_dataset_*.pt
│   └── generated/
│       └── topk_electrolytes.csv
├── src/
│   ├── electrolyte_data.py
│   ├── train_electrolyte_gnns.py
│   ├── generate_electrolytes.py
│   ├── run_electrolyte_pipeline.py
│   └── shared graph / RDKit / GNN utilities
├── models/
│   ├── electrolyte_generation_ready/  # active generation-compatible best checkpoints
│   └── trial/                         # non-selected preserved checkpoints
├── GNN_Model/                         # copied active GNN checkpoints for handoff/use
├── results/
│   ├── electrolyte/
│   ├── electrolyte_generation/        # active metrics and best-model CSVs
│   ├── trial/                         # non-selected metrics and manifests
│   └── figures/electrolyte/
└── demo/
    └── original non-electrolyte molecule demo assets
```

`demo/` preserves the original generic molecule data, code, checkpoint, and results. It is not needed for the active electrolyte workflow.

---

## 14. 关键词

```text
图论
分子图
带标签图
图神经网络
GNN
结构–性质关系
目标导向生成
图搜索
Beam Search
节点度约束
连通性
回路
环秩
子图匹配
图同构
RDKit
分子性质预测
AI for Science
```

---

## 15. 当前代码实现与运行方式

本仓库已经按上述结构实现完整流程，并已切换到电解液分子场景。默认原始数据为 `data/Electrolytes.csv`，程序会先计算 RDKit 描述符，再训练电解液性质预测 GNN，最后进行目标导向电解液候选生成。

### 15.1 一键运行

```bash
python -m src.run_electrolyte_pipeline --epochs 120 --batch-size 32
```

### 15.2 分步运行

```bash
python -m src.electrolyte_data
python -m src.train_electrolyte_gnns --mode both --epochs 120
python -m src.generate_electrolytes
```

默认生成命令会创建一个带时间戳的运行目录，避免覆盖上一次结果：

```text
results/electrolyte_generation_runs/YYYYMMDD_HHMMSS_electrolyte_generation/
```

常用生成参数：

```bash
python -m src.generate_electrolytes --beam-width 128 --max-steps 7 --top-k 100
python -m src.generate_electrolytes --output-dir results/manual_generation_check
python -m src.generate_electrolytes --output data/generated/topk_electrolytes.csv
```

### 15.3 重要输出文件

```text
data/processed/electrolytes_with_rdkit_descriptors.csv   电解液原始性质 + RDKit 描述符
data/processed/electrolyte_graph_dataset_*.pt            电解液图数据集
models/electrolyte_generation_ready/*_gnn.pt             当前保留的最佳生成兼容 GNN 模型
GNN_Model/dielectric_constant_of_solvents_gnn.pt         介电常数生成兼容 GNN checkpoint 副本
GNN_Model/es_ea_ev_gnn.pt                                Es-Ea 生成兼容 GNN checkpoint 副本
GNN_Model/homo_sol_ev_gnn.pt                             HOMO_sol 生成兼容 GNN checkpoint 副本
GNN_Model/lumo_sol_ev_gnn.pt                             LUMO_sol 生成兼容 GNN checkpoint 副本
results/electrolyte_generation/*_metrics.csv             当前最佳模型的单任务预测误差
results/electrolyte_generation/per_task_model_summary.csv 分任务模型汇总
results/electrolyte_generation/best_model_summary.csv     按 R2/MAE 平衡指标保留的最佳模型汇总
results/electrolyte_generation/ordered_best_checkpoints.csv 最佳 checkpoint 顺序索引
models/trial/                                            未选中 checkpoint 归档
results/trial/                                           未选中指标、调参记录与 trial_manifest.csv
demo/                                                     原始非电解液 demo 归档
results/electrolyte_generation_runs/<run>/run.log         终端同款运行日志
results/electrolyte_generation_runs/<run>/final_candidates.csv  本次生成的最终候选
results/electrolyte_generation_runs/<run>/step_summary.csv      每步扩展、筛选、命中统计
results/electrolyte_generation_runs/<run>/resolved_config.json  本次运行的完整配置
results/electrolyte_generation_runs/<run>/run_summary.json      本次运行摘要与关键路径
data/generated/topk_electrolytes.csv                     兼容旧脚本的候选 CSV 副本
results/electrolyte/generated_electrolytes.csv           兼容旧脚本的结果副本
```

### 15.4 修改分子特征与目标性质

所有可配置项集中在 `src/config.py`：

- `FeatureConfig`：修改原子类型、杂化方式、节点特征、边特征；
- `ELECTROLYTE_TARGETS`：修改需要预测的电解液性质；
- `RDKIT_DESCRIPTOR_NAMES` / `EXPANDED_RDKIT_DESCRIPTOR_NAMES`：修改输入 GNN 的 RDKit 描述符；当前 HOMO 生成兼容模型使用 expanded descriptor set，其余 active 模型使用 base descriptor set；
- `ElectrolyteGenerationConfig.targets`：修改电解液候选生成的目标性质区间；当前默认目标为 `Es-Ea > 0.25 eV`、`LUMO > 7.5 eV`、`HOMO < -7.5 eV`、`dielectric < 10`；
- `ElectrolyteGenerationConfig.atom_choices`：修改生成阶段可添加的原子类型；
- `ElectrolyteGenerationConfig.seed_smiles`：修改搜索初始电解液片段。
- `ElectrolyteGenerationConfig.fragment_smiles`：修改生成阶段可拼接的电解液官能团片段，例如醚、碳酸酯、砜、磷酸酯和含氟片段。

### 15.5 模型策略

电解液数据中不同性质的量纲和噪声水平差异较大，例如介电常数跨度远大于 HOMO/LUMO 能级。当前实现默认使用 `multitarget_stratified` 划分策略：每个样本都会携带 4 个电解液目标性质作为只用于划分的数据，划分器同时按 4 个性质的分位桶和电解液簇做贪心平衡，并固定训练、验证、测试比例为 8:1:1。分任务模型不再按当前预测性质单独切分数据，因此 Es-Ea、LUMO、HOMO 和介电常数模型使用完全相同的 train/validation/test 样本 ID。

当前实现会训练分任务 GNN：

```text
Es-Ea       → 3 层 edge-gated GNN + graph feature encoder
LUMO_sol    → 4 层 edge-gated GNN + graph feature encoder
HOMO_sol    → 4 层 edge-gated GNN
Dielectric  → log1p 目标变换 + 4 层 GNN + graph feature encoder
```

同时也可以训练多任务 GNN 作为辅助监督模型。介电常数对簇分布和高值尾部更敏感，当前结果显示它通常更受益于多任务训练。推荐运行：

```bash
python -m src.train_electrolyte_gnns --mode both --epochs 120
```

运行结束后，`per_task_model_summary.csv` 保留每个单任务 GNN 的验证集和测试集 MAE、RMSE、R2；`multitask_metrics.csv` 保留多任务 GNN 的对应指标；`recommended_model_summary.csv` 会按验证集 R2 为每个性质选择当前最合适的 GNN。
