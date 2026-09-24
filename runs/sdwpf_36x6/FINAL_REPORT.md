# SDWPF Physics-CTF 36→6实验报告

## 结论摘要

本次单随机种子（seed 42）实验**不支持**“训练数据越少，Jensen Wake Physics相对Original CTF的优势越大”。Physics-CTF在四个训练比例上的测试MSE都没有低于Original CTF；10%数据时反而退化0.522%。50%与20%的MAE有约0.03%的微弱改善，但MSE没有改善，且按连续时间块进行的配对bootstrap区间均跨过0。

这不是“物理图没有被模型使用”：最终模型每个样本平均激活约184.9条有向边，物理活跃边logit标准差约0.75，beta保持约1.96–2.00。结果说明：当前Jensen尾流图确实进入注意力，但仅靠`Channel Attention += beta * Jensen bias`没有转化为更好的功率预测。

## 1. 数据、坐标与时间定义

- 数据：官方SDWPF full Parquet，MD5 `16e40b93890d43a18946bafbd6aa3577`。
- 原文件含2020年15分钟和2021年10分钟两种采样间隔。为避免“一步”物理时间不一致，仅使用2021年统一10分钟段。
- 任务：过去36步（6小时）预测未来6步（1小时）；预测目标为134台风机的`Patv`。
- 坐标：`sdwpf_turb_location_elevation.csv`中的`TurbID,x,y`按TurbID 1–134严格对齐；SCADA与坐标ID完全覆盖，无聚合或模糊匹配。
- 固定时间划分：Train `[0,36791)`、Validation `[36833,44675)`、Test `[44717,52559)`；相邻集合之间各留42点隔离区。
- 测试集固定为883个窗口，两种模型完全一致。
- 功率按1500 kW额定功率缩放；同时报告归一化误差和MAE(kW)。
- 按公开SDWPF规则屏蔽无效目标，包括缺失、`Patv<=0 & Wspd>2.5`、桨距角大于89度以及方向范围异常。输入只做因果前向填充，不从未来回填。

## 2. 风向换算与sanity check

- 使用公开字段定义：绝对来流方向 `wind_from = (Ndir + Wdir) mod 360°`。
- 尾流传播方向：`wind_to = wind_from + 180°`。
- `Wdir_w`与`Ndir±Wdir`在训练段的圆周MAE均约92°，不能可靠判定正负号，因此没有用它覆盖公开定义。
- 坐标轴相对真北的方向未由数据文档明确给出。仅在训练段比较四个全局旋转角的上游—下游风速差，选择270°；对应得分0.151，高于0°/90°/180°的0.083/0.059/0.019。验证集和测试集未参与该选择。
- 最终使用各风机方向的场级圆均值构图，以避免局部尾流和机舱测量噪声使图结构剧烈抖动。

## 3. Jensen Wake Graph

对源风机`j`和目标风机`i`：

1. 由坐标差和当前下风向计算`d_parallel`与`d_perp`；
2. 仅当`d_parallel>0`且`|d_perp| <= D/2 + k_w*d_parallel`时建立`j→i`；
3. 转子直径`D=82 m`；
4. 使用SL1500/82公开推力系数曲线`C_T(U)`；3 m/s处大于1的原始值按Jensen平方根定义裁剪到0.999；3–25 m/s之外关闭尾流；
5. 原始Jensen速度亏损：

   `delta_ji = (1 - sqrt(1-C_T)) / (1 + 2*k_w*d_parallel/D)^2`；

6. `k_w=0.075`。验证阶段比较过0.04/0.075/0.10，0.075整体最好；
7. 最终对每个目标节点的非零`delta`按行最大值归一化，保留拓扑和相对强度，再由可学习beta控制总幅度。

图为有向、随风向动态变化、无自环；没有使用相关系数建图，也没有人工加入二阶/三阶边。

## 4. 模型修改位置

两种模型都使用真正的Channel Token解码。Original CTF的Local Token、Patch、Channel Token、原始Mask、两层Masked Attention和Decoder保持一致。

Physics-CTF只修改Channel Token到Channel Token的注意力logit：

`score_ij = Q_i K_j^T/sqrt(d) + beta_l * B_phys[i,j,t] + M_ij`

Local→Local、Local→Channel、Channel→Local均未修改。每层只有一个非负可学习`beta_l=softplus(theta_l)`，总参数从1,620,486增加到1,620,488，仅增加2个标量。`beta=0`等价性测试逐元素完全一致。

## 5. Few-shot构造

- 先固定Train Pool/Validation/Test，再只减少Train Pool。
- 将Train Pool切成连续288点（2天）块；剔除几乎完全不可评价的停采块。
- 按`Wind Speed Bin × 45° Wind Direction Sector × Operating State`分层。
- seed 42在每个工况内打乱，并以最大配额缺口算法交错各工况，形成嵌套的100%/50%/20%/10%块集合。
- 每个块内部以6点步长构造36→6窗口，窗口不跨块。
- 训练窗口数：100%=4138、50%=2066、20%=835、10%=438；验证992、测试883。
- 各比例风速与风向直方图保存在`data/sdwpf/cache/protocol.json`。

该方案使用完整Train Pool的工况标签来保证覆盖，但没有使用Validation/Test选择训练块。

## 6. 最终测试指标

`Gain=(CTF-Physics)/CTF`，正值才表示Physics-CTF更好。

|训练比例|CTF MSE|Physics MSE|MSE Gain|CTF MAE|Physics MAE|MAE Gain|CTF MAE (kW)|Physics MAE (kW)|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|100%|0.00917422|0.00918456|-0.1127%|0.0609907|0.0610291|-0.0630%|91.486|91.544|
|50%|0.00937627|0.00937975|-0.0371%|0.0614169|0.0613988|+0.0294%|92.125|92.098|
|20%|0.01064175|0.01064277|-0.0096%|0.0659623|0.0659435|+0.0285%|98.943|98.915|
|10%|0.01257719|0.01264288|-0.5223%|0.0731903|0.0733487|-0.2164%|109.785|110.023|

相对各自100%训练数据的MSE退化：

|训练比例|CTF退化|Physics退化|
|---:|---:|---:|
|100%|0%|0%|
|50%|+2.202%|+2.125%|
|20%|+15.996%|+15.877%|
|10%|+37.093%|+37.654%|

Physics-CTF在50%和20%的相对退化略小，但10%更差，因此不存在一致的“数据越少，物理优势越大”趋势。

## 7. 配对时间块不确定性

使用24个测试窗口为一块、2000次配对block bootstrap。MSE Gain的95%区间：

- 100%：`[-0.172%, -0.038%]`，Physics-CTF稳定更差；
- 50%：`[-0.106%, +0.039%]`，无法区分；
- 20%：`[-0.048%, +0.041%]`，无法区分；
- 10%：`[-0.635%, -0.378%]`，Physics-CTF稳定更差。

这些区间只反映测试时间块变化，不包含随机种子不确定性；本阶段按要求只运行一个seed，不能报告跨seed的mean±std。

## 8. Physics Bias数值诊断

以100%模型为例：

- 学习后beta：第1/2层为1.990/1.961；
- 每样本平均激活有向边：184.9；
- 行归一化边权均值/标准差：0.538/0.385；
- 第1层活跃边QK标准差3.544，Physics Bias标准差0.765；
- 第2层活跃边QK标准差2.477，Physics Bias标准差0.754；
- 全矩阵Physics Bias标准差约0.031，低值来自绝大多数非物理边为0，不能据此认定物理分支未生效。

因此失败原因不是bias太小或beta归零。物理先验确实显著改变了活跃边logit，但这些改动没有稳定改善预测。

## 9. 调参中最重要的观察

1. 原始Jensen deficit配`beta=0.1`几乎与CTF数值一致，偏置被QK淹没。
2. 增大beta到2/5/20后物理分支生效，但验证误差总体不降。
3. 场级圆均风向比逐风机局部风向更稳定，但单独不能带来可靠提升。
4. `k_w=0.04/0.075/0.10`中没有出现可重复的大幅优势。
5. 按目标节点做row-max归一化后，20%验证集出现0.025%的微小改善，因此选择为最终配置；该改善没有迁移到测试MSE。
6. 最终beta没有自动衰减到0，说明优化器没有主动退回Original CTF；这也解释了10%时物理偏置可能成为额外误差源。

## 10. 典型动态物理边

固定测试段中按“激活频率×平均原始Jensen deficit”选出的代表边包括T132→T111、T2→T27和T24→T25。它们只在约15%的测试窗口激活，原始最大速度亏损约0.12–0.13，说明动态图随风向显著切换，并非固定邻接矩阵。

可视化见`dynamic_wake_edges.png`；训练比例对比见`few_shot_comparison.png`。

## 11. 最终判断

**核心假设被本次实验否定。** 在SDWPF、36→6、seed 42和当前CTF实现下，`Fixed Jensen topology + dynamic wake strength + additive channel-attention bias`没有降低CTF学习空间依赖所需的数据量，10%训练数据时还产生了明确退化。

更合理的解释是：

- Jensen描述的是瞬时、稳态、理想化速度亏损，而目标是未来多步有功功率；控制、停机、地形和湍流使二者并非简单一一对应；
- 仅改变“注意谁”不能把风速、风向和传播状态直接送入Value或Decoder，物理上下文的信息带宽有限；
- 36步历史已使Original CTF从功率轨迹中学到强自回归信息，少量正确空间边不足以取代该信息；
- 数据降到10%后，固定强度约2的物理偏置没有随证据不足自动减弱，错误或不适用的边更容易产生负迁移。

这组负结果适合用来约束下一版研究问题：如果继续，不应只进一步放大attention bias，而应验证“带传播时间的wake transport message”或“物理上下文进入Value/状态更新”是否必要；那将是新结构，不属于本阶段。

## 12. 复现文件

- `sdwpf/model.py`：Original CTF与Physics-CTF；
- `sdwpf/wake.py`：方向、SL1500/82 Ct曲线和Jensen动态图；
- `sdwpf/prepare_data.py`、`build_protocol.py`：质量审计、固定切分、few-shot采样；
- `sdwpf/train.py`、`evaluate_run.py`：公平训练与唯一测试；
- `final_results.json/csv`：完整结果与逐时距指标；
- 各run目录：配置、history、最佳检查点、测试预测和测试指标；
- 单元测试：17项全部通过，包括beta=0严格等价、仅Channel Token子矩阵受影响、有向边方向、无未来回填和嵌套连续块。

## 参考资料

- SDWPF数据论文与字段说明：https://pmc.ncbi.nlm.nih.gov/articles/PMC11187227/
- SDWPF官方数据发布：https://figshare.com/articles/dataset/SDWPF_dataset/24798654
- SL1500/82技术资料与Ct曲线：https://nepra.org.pk/licensing/Licences/Licence%20Application/2014/July%202014/Application%20of%20Titan.pdf
- Jensen, *A Note on Wind Generator Interaction*, Risø-M-2411：https://backend.orbit.dtu.dk/ws/portalfiles/portal/55857682/ris_m_2411.pdf
