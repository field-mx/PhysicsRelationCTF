# PhysicsRelationCTF (PRCTF)

## Publication companion

The maintained research-code and experiment companion is in **[publication/](publication/README.md)**.
It contains the selected PRCTF implementations, fixed three-seed results,
public-dataset access links and approved finished figures. Manuscripts, review
materials, industrial numerical observations, trained weights and local caches
are not included in that directory.

- [Installation and model checks](publication/README.md#installation-and-quick-check)
- [Data availability and public links](publication/DATA_AVAILABILITY.md)
- [Complete comparison results](publication/results/method_comparison_summary.csv)
- [Reproducibility](publication/REPRODUCIBILITY.md)
- [Licence status and third-party attribution](publication/LICENSE_STATUS.md)

The content below is preserved **historical development documentation**, not the
current publication protocol or a claim of universal superiority.

---

# Physics-Relation-Guided CTF — historical notes

本目录独立于旧实验。数据为原始 `CIST-MAE/data/SensorData.xlsx`，不是自适应滤波版本。
基线与实验组均为 **Channel Token 解码**，不使用 Local Token 解码或 GCN。

> **当前状态（2026-09-26）**：最新实验为 `runs/random_gaps_v1` —— 训练区间固定嵌套
> 随机时间缺失下的 CTF/Physics 配对比较（单模型种子42、单mask种子，无超参搜索）。
> 100% 观测下 Physics MSE 降低 1.27%，50% 降低 0.25%，30% 反而差 0.36%；
> 详见文末「2026-09-26 随机时间缺失实验」一节。入口 `run_random_gap_experiment.py`。

## 不变的主干

96个历史点 → 每通道4个24点Local Token + 2个原生Channel Token →
2层CTF masked attention（d=256，8头，FFN=1024）→ 原生Channel Token head → 未来96点。
所有原生参数按同一种子初始化，实验组复制基线主干初值。

## 新分支

1. 在原始单位下，从输入历史计算公式特征。
2. 每条物理边在所属关系允许的历史时滞集合中做形状匹配，softmax得到分布式时滞权重。
   v2按训练校准的映射方向区分正向/反向变化；不把反相关排气—库存关系强制当正相关。
3. 对齐源信号和公式特征。用训练前缀估计的标量增益将公式变化量换算到目标通道单位，
   再除以当前历史窗口目标通道标准差，避免直接混合A、V、MPa和摄氏度。
4. 利用原生patch线性嵌入构造非零物理消息；每种关系增加共享的rank8小适配器。
5. 仅在Channel→Channel的既有允许位置增加关系bias和物理value消息。
6. 保留原生输出投影、残差、LayerNorm、FFN和预测头。

公式特征不作为额外预测目标，也不插入原生CTF token序列；因此原生mask尺寸不变。
这不是显式新增虚拟节点的多跳GNN。多层Channel attention传播通道状态；
虚拟物理量由公式预先算出，再沿候选物理边进入消息。

### 方程与适用边界

| 特征 | 计算 | 边界/限制 |
|---|---|---|
| 电功率 | SCRDLFK × SCRDYFK | A×V=W；不是直接把全部电功率当产热功率 |
| 理论单池产氢/产氧 | I/(2F)、I/(4F) | Nc未知，明确为每池理论mol/s；不是整堆实测体积流量 |
| 冷却水温差 | TT010002 − TT010001 | 回水−供水 |
| 碱液温差 | TT010401/TT010403 − TT010301 | 相应侧电解堆出口与冷却器碱液出口的代理温差，未知完整循环路径 |
| 热输运代理 | FT010401 × 碱液温差 | FT010401表头缺时间基准，不标成W；不假设已知cp |
| 压差 | PT010302 − PT010301 | 两侧分离器压差，用于气体杂质边 |
| 阀前后压差 | PT010303 − PT010501 | 已实现特征；当前稀疏图不含该下游边，因此未参与本轮forward消息 |

F=96485.3321233100184 C/mol。训练标量增益只是观测映射校准，**不是**辨识出的
法拉第效率、串联单池数、热容、气室容积或传递函数。未知完整状态方程不强行闭合。
注意：自由标量校准会抵消Faraday特征中的常数倍，所以这一部分实质上是电流驱动的
校准先验，不能仅凭总模型提升声称网络验证了化学计量系数。该限制需要在论文中披露。

依据：
- [NIST CODATA2022](https://www.physics.nist.gov/cuu/pdf/wall_2022.pdf)
- [碱性电解槽热动态与控制](https://arxiv.org/html/2202.13422)
- [电解槽动态建模](https://arxiv.org/html/2311.09882)
- [压力动态相关文献](https://arxiv.org/html/2505.16935)

## 物理图与时间尺度

继承 `GNN_Transformer/equation_topology_v2.py` 的稀疏物理边界；
剔除产氢/产氧测量流量之间的共同原因伪直接边，以及未知单池数下平均/最大电压→总电压边。
73条有向边；`A[i,j]=1`表示目标i读取源j；不添加A²/A³。
每次运行的 `graph.json` 包含完整逐边来源、方程、物理边界、关系类型、时滞和限制。

| 关系 | 边数 | 候选时滞/秒 |
|---|---:|---|
| electrical | 1 | 0 |
| electrochemical | 10 | 0,2,4,8 |
| gas_generation | 2 | 8,16,24,32,48,60 |
| gas_transport | 4 | 0,4,8,16,24,32,48 |
| thermal | 12 | 24,32,48,60,72 |
| hydraulic | 12 | 0,4,8,16,24,32,48 |
| gas_inventory | 10 | 4,8,16,24,32,48,60 |
| impurity | 10 | 24,32,48,60,72 |
| state_consistency | 12 | 0 |

gas_generation的非零时滞指电流生成气体到**下游测量流量**的过程，不指法拉第反应本身延迟。
时滞集合是待验证的工程先验，不是已辨识的装置真实常数。热/杂质动态可超过96秒，
当前输入无法辨识该范围之外的动态。时滞匹配也不能证明因果关系。

同步一致性只连接同侧成对出口温度、同侧成对出口压力、同侧液位与液位差压测点，
双向、零时滞；不跨氢氧侧连接所有同量纲通道。

## 融合与诊断

`score = QK/√d + beta × relation_bias + 原生mask`

`message_ij = attention_ij × (V_j + gamma × physics_ij + bounded_residual_ij)`

- beta=softplus(theta)，初值0.05；gamma=2sigmoid(theta)，初值0.5。
- rho=0.02sigmoid(theta)，初值0.001。小残差网络约6k参数，输出经过tanh后按原生V范数约束。
- 未乘gamma的物理消息最多为2倍对应V范数；不会把接近零的物理消息强行放大。
- 每层记录beta/gamma/rho、物理/残差/V范数，以及真实边位置上的bias/QK标准差。
- 完整模型参数1,712,321；基线1,650,528，增加61,793（约3.74%）。

## 实验口径

- 24234个1Hz点；沿时间固定60%/20%/20%边界：14540、19387、24234。
- 61通道中排除由完整原训练段定义的5个全平稳通道，56个预测通道。
- 100%/50%/30%分别使用原训练段的前14540/7270/4362点（前缀采样，v1/centered等历史实验）；
  random_gaps_v1改为嵌套随机时间缺失，观测点预算相同，见文末该实验一节。
- 训练步长1；验证/测试步长96，每次完整保留未来96点。验证50窗口，测试50窗口。
- 每个训练比例单独拟合输入scaler和物理校准；val/test不参与拟合。
- 跨比例的MSE/MAE统一换回原完整训练段标准化尺度，不是各比例各自的方差尺度。
- 前缀减数据同时减少了工况覆盖，因此这不是只改变独立样本数量的纯样本效率实验。
- Adam1e-4，batch32，最大16轮，前6轮恒定学习率后余弦衰减，最少8轮后patience5。
- 首轮seed42；通过验证诊断后按资源补充43/44，不挑最优seed汇报。
- 验证集选checkpoint/候选方案；`test()`需要冻结checkpoint哈希清单，禁止训练时自动查看test。
- 此测试区间曾用于早期模型研究，不是全新盲测数据；结果不能单独证明跨设备泛化。

## 运行与恢复

最新实验入口（随机时间缺失协议，双卡并行：CTF@cuda:0、Physics@cuda:1）：

```bash
cd /home/maxon/workspace/PhysicsRelationCTF
/home/maxon/workspace/CTF/.venv/bin/python run_random_gap_experiment.py
/home/maxon/workspace/CTF/.venv/bin/python -m unittest test_relation test_random_time_gaps \
    test_random_time_gap_integration test_random_time_decile_coverage test_random_gap_summary -v
```

- 首次运行冻结 `runs/random_gaps_v1/protocol.json`（源码与mask的sha256、嵌套校验、
  十分位保留率、分位数箱计数）；重复运行若协议不一致立即拒绝，防止覆盖已有结果。
- 训练全部成功后自动写入 `selection_lock.json`（6个验证选定checkpoint的sha256），
  锁定后才一次性评估测试集并生成 `summary.json`。
- 逐轮保存 `resume.pth`，同配置重新启动可恢复；日志在 `ctf_train.log` / `physics_train.log`。
- 单元测试覆盖：mask嵌套性与精确点预算、窗口过滤口径、十分位覆盖、汇总计算。

结果位于 `runs/random_gaps_v1/train{100,050,030}/{CTF,Physics}_seed42` 子目录：
`config.json`参数与数据选择、`status.json`实时进度、`history.json`逐轮诊断、
`training.json`验证汇总、`best.pth`验证最优权重、`validation_predictions.npz`验证预测，
测试评估后生成 `metrics.json` / `predictions.npz`。
历史实验（v1/v2/centered/sdwpf_36x6 等）保留在本地 `runs/` 下作对照；
`.gitignore` 规定仅 `runs/random_gaps_v1` 同步到 GitHub。
早期的 user-systemd 训练服务（`physics-relation-ctf-*`）已退役，仅存于历史记录。

## 2026-09-22 最终状态

本轮训练和一次性测试均已结束，GPU已释放。最终保留 `runs/centered` 中的模型：
去除公式patch的末值锚定，使用中心化公式变化；保持两层物理注入。
第一层/第二层均有可学习beta/gamma/rho。不是下面被否定的late-only版本。

固定测试集、3个配对种子（42/43/44），MSE为均值±样本标准差：

| 训练比例 | Original CTF MSE | Physics-CTF MSE | 平均MSE降低 |
|---|---:|---:|---:|
|100%|0.109010 ± 0.000380|0.108418 ± 0.001000|0.543%|
|50%|0.111800 ± 0.000817|0.111384 ± 0.000743|0.372%|
|30%|0.112170 ± 0.000260|0.112060 ± 0.000382|0.098%|

|训练比例|CTF MAE|Physics MAE|
|---|---:|---:|
|100%|0.108095|0.107712|
|50%|0.111336|0.111010|
|30%|0.111965|0.111982|

每组均只有2/3个种子MSE改善；第43个种子在三个比例下都未超过CTF。
逐通道平均MSE改善数为32/56、25/56、34/56。**平均MSE略优，但不能宣称稳定优势或统计显著；
也未支持“数据越少，物理先验优势越大”。**30% MAE还略差。

测试前冻结了18个checkpoint及源码/数据哈希，见 `runs/final/selection.json`。
完整数值、逐通道/逐时距误差和诊断见 `runs/final/summary.json`。
预测位于各模型目录的 `predictions.npz`，每个形状为[50,96,56]；
所有18份预测独立复算MSE，与汇总最大差8.2e-10，配对真实值与窗口索引一致。

### 本次新增小迭代：只用验证集筛选

下表均是诊断种子43，不是最终三种子测试表。选择它是因为50%失效明显；未按结果挑选最终种子。

|方案|50%验证MSE|30%验证MSE|判断|
|---|---:|---:|---|
|CTF控制组（复用）|0.096052|0.098348|不重训、不改划分|
|原中心化Physics|0.096517|0.098107|已完成3种子复核|
|隔离物理分支对共享patch投影的额外梯度|0.096525|0.098032|未修复50%主要退化，不采用|
|公式特征去除相关系数幅度衰减|0.096653|0.098072|50%更差，不采用|
|仅末层物理注入|0.096180|0.098666|改善50%但损害30%，不采用|

这些实验对应独立假设，未叠加未经验证的修改。被否定结果均保留；未对其追加100%训练或测试。
新增实现分别是 `isolated_model.py`、`equation_normalized_model.py`、`late_model.py`，
各有对应 `run_*.py`。最终模型仍使用 `relation_model.py`，不是从候选中按通道拼接预测。

### 原因判断与边界

- 物理分支并非完全不起作用。测试末批的跨层/跨种子平均物理消息范数约为2.69/1.42/2.06，
  小残差范数约0.0178/0.0152/0.0158；残差没有压过物理项。它们不是全测试集的逐样本平均。
- 减数据时，训练相关系数会削弱或翻转部分公式观测映射；但直接消除幅度衰减没有改善50%。
  因而不能把收益有限只归因于“物理项太弱”。
- 前层物理消息会经后一层数据注意力传到无物理入边通道；固定权重验证确认了这一路径。
  但关闭前层注入后30%更差，故不能把这一路径直接等同于主要失败原因。
- 独立投影梯度检查未支持“共享patch投影的额外梯度是主要问题”。
- 真实测量关系尚未完全闭合：Nc、流量标况、容积、控制指令等未知。历史公式特征不等于
  已知未来驱动或完整状态转移方程；滞后权重近均匀也不支持精确时延辨识。
- 原CTF已经接收这些历史通道；公式/拓扑先验的额外信息收益可能有限。这是与结果相容的解释，
  不是已证明的普遍结论。

验证与测试的判断分开保留：3种子验证时50%平均略差0.0098%，最终测试平均略好0.372%。
不得用这个测试结果倒过来挑选候选或继续调参。后续论文验证需要新的独立证据。

## 2026-09-26 随机时间缺失实验（random_gaps_v1，当前最新）

前缀截断在减少样本的同时也裁掉了后半段工况，无法区分"样本少"与"覆盖窄"两种效应。
本轮改为**只对训练区间施加固定嵌套的随机时间缺失**：验证/测试窗口与100%运行
逐字节一致（origins sha256相同），从而单独检验真实观测稀缺的影响。
单模型种子42、mask种子20260922，全程无超参搜索；CTF与Physics共用数据与训练日程。

### 缺失协议（time_gap_sampling.py / random_gap_data.py）

- 仅训练段14540点施加缺失；缺失段长96~720秒随机抽取，保留段最短288秒；
  每个时间十分位固定保护一段≥288秒的连续区间，防止整段工况被删空。
- 三个观测集严格嵌套：30%⊆50%⊆100%；点预算精确为14540/7270/4362。
- 只保留完整落在观测时间戳内的96→96窗口（14349/5551/2261个），
  并校验每个保留点都参与至少一个完整窗口，预算不虚报。
- scaler只用保留时间戳拟合；val/test不参与任何mask或统计。
- `protocol.json`记录十分位保留率与代表通道（SCRDLFK/TT010002/PT010002/FT010002）
  的训练段分位数箱计数，证实在减数据时工况覆盖比前缀截断均匀得多；
  但50%时十分位保留率仍在0.20~1.00间波动，且随机缺失不保证覆盖所有未标注工况
  （协议自带limitation字段）。
- 协议首次运行即冻结（源码与mask的sha256、全部统计）；之后任何不一致的
  重复运行都会被拒绝。

### 模型与训练设置

主干不变（56通道、96→96、2层CTF masked attention、d=256、8头、FFN=1024）。
物理选项：beta=0.05、gamma=0.75（初值由前缀实验的0.5上调，一次性设置）、rho=0.001、
centered表征、physical时滞模式。Adam 1e-4、batch32、16轮，前6轮恒定lr后余弦衰减。
6个最优checkpoint（3比例×2模型）全部按验证集选定并冻结sha256（`selection_lock.json`）后，
才一次性评估测试集。

### 结果（固定测试集50窗口，原完整训练段标准化尺度）

| 观测比例 | 训练窗口 | CTF MSE | Physics MSE | MSE增益 | CTF MAE | Physics MAE | MAE增益 |
|---|---:|---:|---:|---:|---:|---:|---:|
|100%|14349|0.109122|0.107731|+1.27%|0.107962|0.106950|+0.94%|
|50%|5551|0.110613|0.110340|+0.25%|0.109933|0.110254|−0.29%|
|30%|2261|0.113082|0.113486|−0.36%|0.114302|0.115350|−0.92%|

增益=(CTF−Physics)/CTF，正为物理更优。相对各自100%基线的MSE退化：
CTF 50% +1.37%、30% +3.63%；Physics 50% +2.42%、30% +5.34%。

### 判断与边界

- 100%时物理优势（MSE −1.27%）大于前缀实验三种子平均（−0.54%），50%缩水到−0.25%
  （MAE已略差），30%转为+0.36%。**工况覆盖拉均匀后，"数据越少物理先验越有利"
  仍未被支持；物理分支对观测缺失反而更敏感（30%时退化比CTF快约1.7个百分点）。**
- 单种子、单mask，无置信区间；结论是方向性的，不能宣称统计显著。
- gamma初值0.75不同于前缀实验的0.5，属一次性工程设置，未做搜索；
  两轮实验的绝对数值不宜直接相减。
- 与前缀实验互补而非替代：随机缺失回答"覆盖均匀时缺失本身的影响"，前缀实验回答
  "连续尾部数据被裁掉的影响"。两者一致的结论是：物理注入的净收益有限，
  且不随数据稀缺放大。
