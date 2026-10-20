# PhysicsRelationCTF（PRCTF）

[English](README.md) | **简体中文**

## 面向不完备物理知识的多变量时序预测

PhysicsRelationCTF 将**稀疏物理关系**与**物理方程计算得到的消息**融入 ChannelTokenFormer（CTF）。模型保留原生局部 token、通道 token、掩码注意力和**通道 token 解码器**：用已知关系引导信息交互，用可计算的方程特征提供数值内容，再由数据驱动注意力与轻量残差学习物理知识尚未覆盖的动态效应。

**核心问题：当观测变量、设备参数与动态方程不足以建立完整机理模型时，如何利用仍然可用的部分物理信息改善预测？**

<p align="center">
  <img src="figures/architecture/model_architecture.png" alt="PRCTF 总体结构：保留 CTF 骨干，通过物理拓扑和方程消息进行增强" width="100%">
</p>

*总体结构示意。物理知识通过拓扑引导与方程消息两个接口进入通道注意力。图为概念性展示，实际方程消息在注意力内部融合，并非在最终预测曲线上另加一条补偿曲线。*

> 本仓库公开研究代码、已有实验记录与获准公开的成品图，不包含论文正文、工业数值数据或训练权重。相对 Original CTF，电解槽、LBNL、TEP 三个任务的三种子平均 MSE 分别降低 **0.393%、0.778%、0.712%**。这一结果不代表对所有方法、所有场景均占优。

### 目录

- [1. 问题发掘与研究定位](#problem)
- [2. 模型结构与物理信息嵌入](#architecture)
- [3. 三个物理系统的建模](#physics)
- [4. 实验设置与结果](#results)
- [5. 安装与快速核验](#quick-start)
- [6. 公开数据准备与训练](#reproduce)
- [7. 代码与实验位置](#code-map)
- [8. 论文配图与工业预测示例](#figures)
- [9. 结论边界与复现限制](#limits)
- [10. 致谢、许可与引用](#citation)

<a id="problem"></a>
## 1. 问题发掘与研究定位

### 1.1 从工业应用出发：预测不只是拟合曲线

工业监控系统持续采集电气、温度、压力、流量、液位和成分等多类测量。预测这些物理量的未来轨迹，有助于提前判断工况变化，为过程监测与运行决策提供参考。不同测量共同受到物料输运、能量交换和闭环控制影响，但量纲与响应时间并不一致。

本研究从**工业碱性电解水制氢系统**出发。模型需要从耦合测量中提取有效信息，同时避免将所有统计相关都解释为已知物理依赖。

<p align="center">
  <img src="figures/industrial/raw_signal_overview.png" alt="获准公开的工业电解槽原始监测信号概览" width="100%">
</p>

*工业监测信号示例。该成品图用于定性介绍数据，不等同于公开完整数值数据集。*

### 1.2 问题所在：物理知识并非完全未知，也不足以完整建模

仅依赖既有监测记录开展研究时，设备参数、测点边界、工况变化与动态关系往往不齐全，难以构造完整装置模型。然而，一部分局部规律仍然明确：电功率与电流、电压有关；产气参考速率遵循法拉第规律；热输运与流量和温差有关。

利用这些知识，需要分别回答两个问题：

| 问题 | 建模困难 | PRCTF 的处理方式 |
|---|---|---|
| **哪些通道之间具有物理关系依据？** | 已知图往往稀疏，不能把所有未知连接都判定为无效。 | 在通道注意力中加入软物理偏置，保留原生自由交互。 |
| **这些关系能够提供什么数值信息？** | 仅给出图连接，并不等于网络已经利用了方程；异量纲信息也不能直接混合。 | 构造方程消息，经目标尺度转换和轻量关系适配后参与聚合。 |

模型遵循：**可由已知物理计算的部分直接计算，未知动态由数据学习，未建模效应由受限残差补充。** 这里不假设少量局部方程足以闭合整个系统。

### 1.3 为什么以 CTF 为骨干？

CTF 用局部 token 表示通道内时序信息，用通道 token 组织跨通道交互，为物理关系和方程内容提供了明确的嵌入位置。因此，PRCTF 保留原生预测骨干，在其中增加两个物理接口，而不是使用额外 GCN 的输出替换通道 token。

本研究的重点是**不完备物理信息如何表达、如何被预测模型利用**，不是首次将图与 Transformer 结合，也不声称注意力与经典 GCN 严格等价。

<a id="architecture"></a>
## 2. 模型结构与物理信息嵌入

### 2.1 保持不变的 CTF 骨干

1. 对各通道历史窗口归一化，划分完整 patch。
2. 将 patch 映射为**局部 token**，为各通道引入原生可学习的**通道 token**。
3. 保留 CTF 掩码：局部 token 仅读取本通道局部 token；通道 token 读取本通道历史，以及其他通道相同槽位的通道 token。
4. 经过多层注意力后，**仅使用最终通道 token 解码预测**。

PRCTF 不改变上述 token 构造、掩码和读出方式。当前任务均使用真实观测组成的完整 patch，不通过复制观测补齐长度。不使用局部 token 解码，也不使用旧版 `GCN → 替换通道表征` 路径。

### 2.2 接口一：以物理拓扑引导注意力

对于来源通道 `j` 指向目标通道 `i` 的关系，邻接矩阵约定为：

$$
A^{\mathrm{phys}}_{ij}=1 \quad\Longleftrightarrow\quad j\rightarrow i.
$$

**行对应接收信息的 Query，列对应提供信息的 Key/Value。** 固定图表达已有物理依据，不是相关系数图，也不是完整因果图。轻量网络根据当前历史调整已有物理边的强度，不新增物理连接。

在相应通道 token 位置，注意力为：

$$
\alpha^{(l)}=\operatorname{Softmax}_{\mathrm{key}}\!\left(
\frac{Q^{(l)}K^{(l)\top}}{\sqrt{d_h}}+\beta_l B(X_{\mathrm{hist}})+M
\right).
$$

其中，`B` 是映射到 token 布局的物理偏置，`M` 是原生 CTF 掩码，`beta_l` 控制当前层物理引导强度。偏置**只加入相同槽位的 Channel–Channel 位置**，不改变 Local–Local、Local–Channel 和 Channel–Local 的掩码结构。没有物理先验覆盖、但原生掩码允许的跨通道交互仍由数据学习。

<p align="center">
  <img src="figures/architecture/topology_attention.png" alt="将稀疏有向物理拓扑映射为同槽位通道 token 注意力偏置" width="100%">
</p>

*物理拓扑与注意力矩阵的对应关系。两层为便于说明而突出不同路径，实际共享同一物理先验。彩色单元表示存在偏置的位置，不表示实测注意力强弱。多层注意力逐层传播已融合的通道信息，无需人为增加物理图的高阶连接。*

### 2.3 接口二：由物理方程构造预测消息

方程信息分支**只使用历史观测**，主要包含四步：

1. **计算方程特征。** 例如，由电流、电压得到电功率 `UI`，由电流得到单电解单元的氢气产率参考量 `I/(2F)`。
2. **转换到目标表示。** 用训练数据标定的观测增益处理未知测量尺度，并在关系允许的历史范围内进行对齐。
3. **投影与关系适配。** 复用 patch 投影，将方程特征转换为 CTF 可处理的向量，再用小规模关系适配器补充公式与实测过程之间的偏差。
4. **在注意力内融合。** 将方程消息与轻量残差消息沿相应物理关系注入，补充原生 value 的内容。

电解槽实例中的消息构造可概括为：

$$
p^{\mathrm{phys}}_{ij}
=E_P(\phi^{\mathrm{eq}}_{ij})
+\operatorname{Adapt}_{r_{ij}}\!\left(
E_P(\phi^{\mathrm{eq}}_{ij})\Vert E_P(z^{\mathrm{lag}}_{j\to i})
\right).
$$

`phi_eq` 表示经过尺度转换与历史对齐的方程特征，`z_lag` 表示相应源通道的历史，`r_ij` 表示已知关系类型。当前电解槽配置使用中心化的方程变化特征，不将局部公式当作传感器绝对读数的精确方程。

<p align="center">
  <img src="figures/architecture/physics_message.png" alt="历史观测经方程计算、尺度转换和关系适配形成物理消息，进入 CTF 注意力聚合" width="100%">
</p>

*方程消息流程。已知函数提供关系内容，轻量适配与基于历史的残差补充观测映射偏差和未建模效应。*

聚合过程可示意为：

$$
O_i^{(l)}=O_{\mathrm{data},i}^{(l)}
+\sum_{j:\,j\to i\in\mathcal E}\omega_{ij}^{(l)}
\left(\gamma_l\overline p_{ij}^{(l)}+\rho_l\overline r_{ij}^{(l)}\right).
$$

原生数据聚合始终保留。电解槽中，`omega` 使用相应原生注意力权重，训练时包含注意力 dropout。公开数据实例另外采用已记录的物理邻居条件权重、有效性和历史一致性处理。因此，三个实例遵循同一种双接口思路，**不代表每项计算细节完全相同**。

### 2.4 已知物理、未知动态与强度控制

| 组成 | 作用 | 不作出的假设 |
|---|---|---|
| 固定物理图与方程函数 | 记录已知关系，计算可获得的物理特征。 | 已获得完整系统拓扑或完整动态求解器。 |
| 数据注意力与关系适配器 | 学习其余依赖，适应公式与观测之间的偏差。 | 已辨识全部设备参数或真实传播时滞。 |
| 小规模残差消息 | 补充所选方程没有描述的效应。 | 独立输出一条未来补偿曲线。 |
| 可学习 `beta`、`gamma`、`rho` | 控制拓扑引导、方程信息和残差补偿的强度。 | 初始化越大就必然越好。 |

电解槽中采用非负或有界参数化：`beta = softplus(theta_beta)`、`gamma = 2 sigmoid(theta_gamma)`、`rho = 0.02 sigmoid(theta_rho)`，并参照原生 value 控制消息范数。**显式关闭三项增强接口后**，PRCTF 恢复配对的 Original CTF 前向路径；仅设 `beta = 0` 不会同时关闭方程与残差消息。

<a id="physics"></a>
## 3. 三个物理系统的建模

将框架用于其他物理系统时，需要重新定义相应图结构、方程特征和观测映射，不能直接把电解槽物理参数套到公开数据上。

| 系统 | 观测内容 | 使用的物理知识示例 | 物理边界与未知部分 |
|---|---|---|---|
| 工业碱性电解槽 | 电流、电压、温度、压力、流量、液位和气体成分。 | 电功率 `P = UI`；法拉第参考量 `I/(2F)`、`I/(4F)`；流量与温差构成的热输运代理量；冷却温差；分离器压差。 | 电解单元数量、效率、完整库存及测点边界不完全已知，不能据此精确闭合系统。 |
| LBNL 冷水机组仿真 | 水温、流量、机组功率、压力、泵/风机状态与控制参考量。 | 水侧换热、机组能量平衡特征、混合、相似定律以及历史控制参考量。 | 名义方程不一定适用于所有实测运行状态，使用有效标定和历史一致性处理。 |
| Tennessee Eastman 过程 | 过程流量、压力、温度、组分与操纵变量。 | 仿真器支持的阀门–流量、压差–流量、压缩机、换热、冷却平衡与液体密度关系。 | 测量含噪声，组分分析存在延迟，控制指令并非真实阀位；不补造不可观测的隐状态动力学。 |

当前冻结图分别包含：电解槽 **56 个通道、73 条有向边、9 类关系**；LBNL **73 个通道、112 条边、6 类关系**；TEP **52 个通道、52 条边、6 类关系**。图记录同时保留方程来源、关系类型、候选历史偏移和适用限制。

<details>
<summary>查看电解槽物理拓扑及对应注意力表示</summary>

<p align="center">
  <img src="figures/topology/electrolyzer_topology_zh.png" alt="以传感器物理量名称标注的电解槽有向物理关系图" width="100%">
</p>

<p align="center">
  <img src="figures/topology/electrolyzer_channel_attention_zh.png" alt="与电解槽物理拓扑对应的通道级关系矩阵" width="100%">
</p>

节点图与通道级矩阵表示同一个先验，不是训练后学习到的注意力热图。映射至 token 层时，每条关系重复作用于相匹配的通道 token 槽位。英文图和完整掩码注意力映射见 [figures/topology](figures/topology)。

</details>

精确定义见 [电解槽图](configs/industrial/graph.json)、[LBNL 图](configs/lbnl/graph.json) 和 [TEP 图](configs/tep/graph.json)。实际实验以这些机器可读配置为准，而非以示意图代替模型定义。

<a id="results"></a>
## 4. 实验设置与结果

### 4.1 评价设置

以下为 **100% 训练数据、固定种子 42/43/44** 的结果。同一 CTF–PRCTF 配对使用相同数据划分、历史输入、预测长度、目标、预处理、公共骨干初始化和训练策略。

| 数据集 | 采样间隔 | 输入通道 / 评价目标 | 输入 → 输出 | patch 长度 |
|---|---|---:|---:|---:|
| 工业电解槽 | 1 秒 | 56 / 56 | 96 → 96 点 | 24 |
| LBNL 冷水机组 | 1 分钟 | 73 / 58 | 96 → 96 点 | 24 |
| TEP | 3 分钟 | 52 / 52 | 20 → 20 点 | 5 |

LBNL 并非所有控制量和设定值都作为评价目标。TEP 窗口不跨仿真运行边界。归一化及经验观测标定使用训练数据，检查点由验证集选择。不同数据集的归一化误差**不直接横向比较，也不合并为一个总误差**。

### 4.2 与 Original CTF 的配对结果

下表为三种子**均值 ± 样本标准差**（`ddof = 1`），数值越低越好。

| 数据集 | 模型 | 归一化 MSE | 归一化 MAE |
|---|---|---:|---:|
| 电解槽 | Original CTF | 0.10901009 ± 0.00038041 | 0.10809476 ± 0.00016571 |
| 电解槽 | **PRCTF** | **0.10858198 ± 0.00075427** | **0.10775839 ± 0.00079343** |
| LBNL | Original CTF | 0.00257001 ± 0.00003264 | 0.00462957 ± 0.00018936 |
| LBNL | **PRCTF** | **0.00255002 ± 0.00003837** | **0.00456146 ± 0.00007519** |
| TEP | Original CTF | 0.82707916 ± 0.00123123 | 0.70286754 ± 0.00060949 |
| TEP | **PRCTF** | **0.82118871 ± 0.00135276** | **0.70023528 ± 0.00075871** |

降幅定义为 `100 ×（CTF 误差 − PRCTF 误差）/ CTF 误差`，**正值表示 PRCTF 误差更低**。平均误差降幅由两个模型的三种子均值计算，不是分别挑选最优结果后比较。

| 数据集 | 平均 MSE 降幅 | 平均 MAE 降幅 | 配对单种子最高 MSE 降幅 | MSE 胜出组数 |
|---|---:|---:|---:|---:|
| 电解槽 | **0.393%** | 0.311% | 1.275% | 2 / 3 |
| LBNL | **0.778%** | 1.471% | 2.588% | 2 / 3 |
| TEP | **0.712%** | 0.375% | 0.760% | 3 / 3 |

结果支持：**经过系统对应的物理建模，同一种双接口思路在三个物理场景中均获得了相对 CTF 的小幅平均收益**，共 7/9 组 MSE 改善。最高降幅不等于平均性能，三个随机种子也不构成统计显著性或独立装置重复验证。

<details>
<summary>查看全部九组配对 MSE，含退化结果</summary>

| 数据集 | 种子 | CTF MSE | PRCTF MSE | 降幅 |
|---|---:|---:|---:|---:|
| 电解槽 | 42 | 0.10912231 | 0.10773149 | +1.275% |
| 电解槽 | 43 | 0.10858620 | 0.10884467 | −0.238% |
| 电解槽 | 44 | 0.10932176 | 0.10916977 | +0.139% |
| LBNL | 42 | 0.00258835 | 0.00253391 | +2.103% |
| LBNL | 43 | 0.00253232 | 0.00259382 | −2.428% |
| LBNL | 44 | 0.00258936 | 0.00252234 | +2.588% |
| TEP | 42 | 0.82692956 | 0.82170499 | +0.632% |
| TEP | 43 | 0.82592958 | 0.81965383 | +0.760% |
| TEP | 44 | 0.82837836 | 0.82220730 | +0.745% |

来源：[ctf_prctf_paired.csv](results/ctf_prctf_paired.csv)，同时包含每组 MAE。

</details>

### 4.3 组件对照：收益来自哪里？

这些对照经过分别训练，不是仅在同一个训练完成的模型上临时关闭模块。评价对象为**完整模型 vs 关闭某个接口的模型**：

| 数据集 | 完整 PRCTF 平均 MSE | `no_bias` 平均 MSE | `no_equation` 平均 MSE |
|---|---:|---:|---:|
| 电解槽 | 0.10858198 | 0.10874063 | 0.10924427 |
| LBNL | 0.00255002 | 0.00255095 | 0.00257068 |
| TEP | 0.82118871 | 0.82118710 | 0.82708094 |

- `no_bias` **只关闭 beta**，不删除物理图，也不关闭方程和残差消息。完整模型在电解槽和 LBNL 上平均 MSE 略低；TEP 几乎不变，且 `no_bias` 的数值略优。
- `no_equation` **只关闭 gamma**，其他分支仍可能使用方程相关上下文。相较该对照，完整模型在电解槽、LBNL、TEP 上平均 MSE 分别降低 **0.606%、0.804%、0.712%**。
- 显式方程消息的收益较一致，不能据此声称每个接口在所有系统上都独立有效。

来源：[ablation_and_input_per_seed.csv](results/ablation_and_input_per_seed.csv)。保留的直接方程输入对照还用于检验：仅将相同方程特征交给 CTF，是否能达到相同效果。

### 4.4 其他预测方法与物理融合对照

仓库保留 **7 种方法 × 3 个数据集 × 3 个种子，共 63 条记录**：PRCTF、Original CTF、MTGNN、PatchTST、PhyGNN 预测适配版、EquationInput-CTF 和 PG-Reg-CTF。后两者为自定义对照，不是独立发表架构，也不称作完整系统的标准 PINN。

**PRCTF 并非在所有数据集上优于全部方法。** LBNL 上 MTGNN 和 PatchTST 平均 MSE 更低；TEP 上直接方程输入及物理正则对照更低。电解槽的对比方法扩展经过种子 42 的预实验筛选，应作为探索性对比，而非无偏架构排行榜。

<details>
<summary>查看全部方法的平均误差与历史训练耗时</summary>

耗时为历史记录的平均秒数，不是统一硬件条件下的效率基准。各行使用种子 42/43/44；完整标准差与逐指标最优值见 CSV。

| 数据集 | 模型 | 平均 MSE | 平均 MAE | 历史训练时间（秒） |
|---|---|---:|---:|---:|
| 电解槽 | Original CTF | 0.10901009 | 0.10809476 | 514.08 |
| 电解槽 | **PRCTF** | **0.10858198** | **0.10775839** | 615.17 |
| 电解槽 | MTGNN | 0.11012124 | 0.10799903 | 182.18 |
| 电解槽 | PatchTST | 0.11144723 | 0.11123995 | 89.18 |
| 电解槽 | PhyGNN 适配版 | 0.14648583 | 0.18029671 | 69.83 |
| 电解槽 | EquationInput-CTF | 0.11826381 | 0.11990936 | 472.50 |
| 电解槽 | PG-Reg-CTF | 0.12150680 | 0.12295444 | 511.56 |
| LBNL | Original CTF | 0.00257001 | 0.00462957 | 449.79 |
| LBNL | **PRCTF** | 0.00255002 | 0.00456146 | 521.15 |
| LBNL | MTGNN | **0.00240191** | **0.00422672** | 162.06 |
| LBNL | PatchTST | 0.00253362 | 0.00438390 | 71.18 |
| LBNL | PhyGNN 适配版 | 0.00607557 | 0.03631560 | 30.75 |
| LBNL | EquationInput-CTF | 0.00255610 | 0.00463364 | 850.74 |
| LBNL | PG-Reg-CTF | 0.00442930 | 0.00710169 | 543.65 |
| TEP | Original CTF | 0.82707916 | 0.70286754 | 634.95 |
| TEP | **PRCTF** | 0.82118871 | 0.70023528 | 737.27 |
| TEP | MTGNN | 0.86585570 | 0.72509488 | 272.31 |
| TEP | PatchTST | 1.01060121 | 0.78922623 | 125.17 |
| TEP | PhyGNN 适配版 | 0.86653941 | 0.73170794 | 114.27 |
| TEP | EquationInput-CTF | **0.79712365** | **0.68775879** | 992.96 |
| TEP | PG-Reg-CTF | 0.79830351 | 0.68861369 | 1059.45 |

</details>

数据来源：[逐种子结果](results/method_comparison_per_seed.csv)、[mean/std/best/耗时汇总](results/method_comparison_summary.csv)、[模型适配说明](workspace/PhysicsRelationCTF/runs/physics_ai_comparison_v1/UPSTREAM.md)。

<a id="quick-start"></a>
## 5. 安装与快速核验

### 5.1 配置环境并获取固定版本 CTF

建议使用 Python 3.10 或兼容环境。请先根据 CPU/CUDA 环境安装合适的 PyTorch，研究代码使用的接口要求 **PyTorch ≥ 2.7**。依赖快照记录的是打包时的软件环境，不代表已经完整还原原训练环境。

```bash
git clone https://github.com/field-mx/PhysicsRelationCTF.git
cd PhysicsRelationCTF

python -m venv .venv
source .venv/bin/activate

# 先安装适合本机的 PyTorch，再安装其余模型依赖。
pip install -r requirements-model.txt

# 获取第三方源码前，先阅读 THIRD_PARTY_NOTICES.md。
python scripts/fetch_upstreams.py --ctf
```

脚本获取 CTF 固定版本 `b1c100effb9c27cd25de07e91de7f9005d18d5e7`，仓库不直接捆绑该源码。若目标上游目录已经存在，脚本拒绝覆盖。需要对照方法源码时，可在确认相应许可后单独运行 `python scripts/fetch_upstreams.py --comparators`。

### 5.2 检查文件、表格与模型前向过程

```bash
# 仅需标准库，不下载数据，也不启动训练。
python scripts/verify_package.py
python scripts/summarize_results.py

# 安装依赖并获取 CTF 后，分别运行各实例的合成输入测试。
python scripts/smoke_test.py --dataset industrial
python scripts/smoke_test.py --dataset lbnl
python scripts/smoke_test.py --dataset tep
```

测试使用相同公共骨干权重构造配对模型，检查输出有限，并在显式关闭全部物理接口后与 CTF 比较。已记录的三组 CPU 打包测试最大绝对差均为 **0.0**。这只验证合成输入下的执行与退化等价性，不验证预测精度。工业测试中的合成单位统计量有明确标注，不会替代真实工业训练数据。

可添加 `--device cuda:0` 在 GPU 上执行；也可通过 `--ctf-root /path/to/ChannelTokenFormer` 指定已存在的固定版本 CTF 目录。

<a id="reproduce"></a>
## 6. 公开数据准备与训练

### 6.1 获取对应公开数据子集

| 数据集 | 公开来源 | 本研究使用范围 |
|---|---|---|
| LBNL | [冷水机组官方页面](https://faultdetection.lbl.gov/dataset/simulated-chiller-plant/) · [数据集合 DOI](https://doi.org/10.25984/1881324) | 仅使用无故障 `ChillerPlant.csv`，不混用全部故障场景。 |
| TEP | [Harvard Dataverse 数据记录](https://doi.org/10.7910/DVN/6C3JR1) · [获取镜像](https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset) | 无故障训练、测试 RData 文件中的固定运行编号。 |

LBNL 按时间划分为 **367878 / 78831 / 78831 点**。TEP 使用训练文件的 **1–80** 号运行训练、**81–100** 号运行验证，测试文件的 **401–500** 号运行测试。公开原始数据仅附来源，不在本仓库镜像分发。完整引用、来源、访问条件与校验记录见 [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md)。

```bash
pip install -r requirements-data.txt
python scripts/prepare_public_data.py --dataset lbnl
python scripts/prepare_public_data.py --dataset tep
```

准备脚本检查原文件哈希与任务成员，不启动训练。下载和认证条件可能变化；打包过程中没有重新执行完整大文件获取，因此不将这些入口描述为已经全面重测的下载服务。

### 6.2 配对训练 CTF 与 PRCTF

```bash
# LBNL：96 点历史预测未来 96 点。
python scripts/train_public.py --dataset lbnl --kind CTF --seed 42 --device cuda:0 --out outputs/lbnl_seed42
python scripts/train_public.py --dataset lbnl --kind Physics --seed 42 --device cuda:0 --out outputs/lbnl_seed42

# TEP：20 点历史预测未来 20 点。
python scripts/train_public.py --dataset tep --kind CTF --seed 42 --device cuda:0 --out outputs/tep_seed42
python scripts/train_public.py --dataset tep --kind Physics --seed 42 --device cuda:0 --out outputs/tep_seed42
```

`Physics` 对应 PRCTF。其余实验使用固定种子 **43、44**，不通过搜索种子寻找有利结果。入口**仅使用 Train/Validation 训练与选择检查点，不自动运行 Test**。新实验使用新的输出位置，脚本禁止将发布的证据目录指定为训练输出。

| 实例 | beta / gamma / rho 初始值 | 学习率 | 最大轮数 | batch size |
|---|---|---:|---:|---:|
| 电解槽 | 0.05 / 0.75 / 0.001 | 0.0001 | 16 | 32 |
| LBNL | 0.05 / 0.10 / 0.001 | 0.0003 | 32 | 32 |
| TEP | 0.05 / 0.75 / 0.001 | 0.0003 | 16 | 64 |

三个实例的 CTF 骨干均采用隐藏维度 256、FFN 维度 1024、2 层编码器、8 个注意力头、每通道 2 个通道 token、dropout 0.1。上述文件是初始化与结构配置，**不是训练后权重**；注入强度在训练中继续学习。

**最终测试单独进行：** 先锁定两个模型由验证集选择的检查点，再使用保留的评价函数开展测试。可移植入口不承诺“一条命令逐位复现论文误差”。评价约定与历史运行依赖见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

### 6.3 工业数据的复现边界

工业数值观测、拟合得到的尺度与增益统计、模型权重和作图源序列不随仓库分发。公开的图、数据结构说明、模型代码、指标与成品图可供检查方法，但没有原始记录授权访问就不能重新生成工业精度结果。仓库不作未经确认的“可申请获取数据”等承诺。

<a id="code-map"></a>
## 7. 代码与实验位置

```text
.
├── README.md / README_zh-CN.md     # 中英文说明及切换
├── DATA_AVAILABILITY.md           # 数据来源、子集与访问边界
├── REPRODUCIBILITY.md             # 训练评价约定与限制
├── configs/{industrial,lbnl,tep}/  # 冻结的模型配置与物理图
├── results/                      # 完整对比与组件对照表格
├── figures/                      # 获准公开的 PNG/PDF 成品图
├── scripts/                      # 模型构造、数据准备和核验入口
├── manifests/                    # 文件哈希与来源映射
└── workspace/
    ├── PhysicsRelationCTF/        # 精选模型、实验源码与数值证据
    ├── PhysicsGuidedCTF/          # CTF 通道解码封装和数据工具
    └── GNN_Transformer/           # 所需的历史物理拓扑定义
```

`workspace/` 保留相邻目录结构是为了兼容原有相对导入，并非上传整台服务器工作区。存在 `GNN_Transformer/` 不表示当前模型仍使用旧版 GCN 编码器。

| 查看内容 | 代码入口 |
|---|---|
| 可移植的配对模型构造 | [scripts/model_factory.py](scripts/model_factory.py) |
| 电解槽拓扑偏置、消息投影、聚合与残差 | [relation_model.py](workspace/PhysicsRelationCTF/relation_model.py) |
| 电解槽方程函数、关系类型和历史范围 | [physics.py](workspace/PhysicsRelationCTF/physics.py) |
| 原生 CTF 通道 token 解码封装 | [PhysicsGuidedCTF/model.py](workspace/PhysicsGuidedCTF/model.py) |
| 当前 LBNL 物理消息实现 | [model_v7.py](workspace/PhysicsRelationCTF/runs/publicdata_exp/optimization_v2/model_v7.py) 及其保留的导入模块 |
| LBNL 方程与控制参考量定义 | [public_physics_v4.py](workspace/PhysicsRelationCTF/runs/publicdata_exp/public_physics_v4.py) 及其导入模块 |
| TEP 方程库与标定 | [tep_physics.py](workspace/PhysicsRelationCTF/runs/TEP/tep_physics.py) |
| 当前 TEP 20→20 适配器 | [length_model.py](workspace/PhysicsRelationCTF/runs/TEP/window_search_v1/length_model.py) |
| 组件对照与直接方程输入实验 | [paper_minimal_v1](workspace/PhysicsRelationCTF/runs/paper_minimal_v1) |
| 电解槽方法对比适配与记录 | [physics_ai_comparison_v1](workspace/PhysicsRelationCTF/runs/physics_ai_comparison_v1) |
| 公开数据的方法对比记录 | [physics_ai_comparison_public_v1](workspace/PhysicsRelationCTF/runs/physics_ai_comparison_public_v1) |
| 原文件与公开副本的对应关系 | [manifests/source_map.json](manifests/source_map.json) |

`paper_minimal_v1` 只是组件对照实验的历史目录名，不包含论文正文。部分历史总控脚本依赖原缓存、权重锁或工业标定信息；公开的免权重模型构造和 Train/Validation 训练应使用 `scripts/` 入口，不应把每个历史总控都理解为可独立直接运行的脚本。

<a id="figures"></a>
## 8. 论文配图与工业预测示例

以下均为已获准公开的成品图，仓库不包含论文正文或参考文献全文 PDF。

| 图片 | PNG | PDF |
|---|---|---|
| 总体模型结构 | [查看](figures/architecture/model_architecture.png) | [下载](figures/architecture/model_architecture.pdf) |
| 拓扑到注意力的映射 | [查看](figures/architecture/topology_attention.png) | [下载](figures/architecture/topology_attention.pdf) |
| 方程消息流程 | [查看](figures/architecture/physics_message.png) | [下载](figures/architecture/physics_message.pdf) |
| 英文电解槽拓扑 | [查看](figures/topology/electrolyzer_topology_en.png) | [下载](figures/topology/electrolyzer_topology_en.pdf) |
| 中文电解槽拓扑 | [查看](figures/topology/electrolyzer_topology_zh.png) | [下载](figures/topology/electrolyzer_topology_zh.pdf) |
| 工业滤波预测示例 | [查看](figures/industrial/lowpass_test_predictions.png) | [下载](figures/industrial/lowpass_test_predictions.pdf) |

<p align="center">
  <img src="figures/industrial/lowpass_test_predictions.png" alt="独立低通滤波实验中仪表空气总管压力和氢气冷却器出口温度的预测对比" width="100%">
</p>

*独立滤波实验：仪表空气总管压力与氢气冷却器出口温度。训练、验证、测试的输入及目标均经过二阶 Butterworth 因果低通滤波，截止频率为 0.05 Hz。曲线由连续预测窗口拼接，不是一次性开环预测未来 4847 点，也不是前述原始数据三种子结果表的来源。*

图片使用前请查阅 [图片说明](figures/README.md) 和 [图片来源映射](manifests/figure_sources.json)。架构图采用简化布局，精确计算以图配置和源码为准。

<a id="limits"></a>
## 9. 结论边界与复现限制

- **已支持的结论：** 不完备物理知识能够转化为注意力引导与方程消息；经系统对应的适配后，当前三个全数据任务均取得了相对相同 CTF 骨干的小幅平均改善。
- **尚未证明：** 对所有方法均占优、向新装置零样本迁移、仅凭三种子就具有统计显著性、严格守恒、因果发现，或数据越少优势必然越大。
- **可解释性的含义：** 可以检查已知关系来源、方程计算和物理信息进入预测的位置；注意力权重和拟合观测增益不等于因果作用或已辨识的物理常数。
- **关系失准处理：** 公开实例包含已记录的方程有效性与历史一致性检查，但不能保证识别和抑制所有错误先验。
- **开发阶段暴露：** 种子 42 和固定测试集曾用于早期开发；增加种子有助于观察重复性，但不能将已经观察过的测试集重新称为盲测。
- **比较条件：** 适用的物理方法使用同一可用物理信息预算；不同架构并非参数量或运行时间完全匹配。适配方式及预实验筛选限制均有记录。
- **核验范围：** 已完成内容、文件校验和合成前向测试；仓库打包没有重新执行完整数据下载及全量训练，不保证跨软硬件逐位相同的精度。
- **少数据结论：** 之前的 few-shot 与 SDWPF 纯偏置实验未确立“数据越少，物理优势越大”的单调规律；本说明不将其改写为成功验证。

详细记录：[VALIDATION.md](VALIDATION.md)、[REPRODUCIBILITY.md](REPRODUCIBILITY.md)、[results/README.md](results/README.md)。

<a id="citation"></a>
## 10. 致谢、许可与引用

PRCTF 基于 **ChannelTokenFormer**，不将基础预测骨干作为本工作的原创。使用相应组件时，也请引用上游方法与公开数据：

- [ChannelTokenFormer](https://github.com/jinkwan1115/ChannelTokenFormer)：CTF 骨干及掩码注意力设计。
- [MTGNN](https://github.com/nnzhan/MTGNN)、[PatchTST](https://github.com/yuqinie98/PatchTST)、[PhyGNN](https://github.com/DiYi1999/PhyGNN)：对比方法源码，各自遵循原许可；参见 [适配说明](workspace/PhysicsRelationCTF/runs/physics_ai_comparison_v1/UPSTREAM.md)。
- [LBNL FDD 数据集合](https://doi.org/10.25984/1881324)：请同时说明使用的冷水机组仿真子集。
- [Rieth 等，TEP 仿真数据，Harvard Dataverse V1](https://doi.org/10.7910/DVN/6C3JR1)：公开过程数据来源。

本仓库尚未为原创代码选定开源许可证，公开可见不等于授予无限制复用权；第三方源码和数据仍适用各自条款。请查阅 [LICENSE_STATUS.md](LICENSE_STATUS.md) 与 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

当前请按照 [CITATION.md](CITATION.md) 以**固定提交号**引用本仓库，不虚构论文录用信息、文章 DOI 或软件归档 DOI。旧版源码保留在 [legacy-v1 分支](https://github.com/field-mx/PhysicsRelationCTF/tree/legacy-v1)，`main` 为面向论文公开的目录结构。
