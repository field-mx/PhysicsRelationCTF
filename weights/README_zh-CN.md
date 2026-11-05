# PRCTF预训练模型

[English](README.md) | **简体中文**

本目录发布与仓库论文结果对应的三个**固定seed=42、由验证集选择训练轮次**的
PRCTF参考检查点，可直接加载推理，无需重新训练。不进行新的种子筛选或精度实验。

| 数据集 | 权重文件 | 输入→预测步数 | 输入／默认输出通道 | 采样间隔 |
|---|---|---|---|---|
| 工业碱性电解槽 | [下载](industrial/prctf_seed42.pt) | 96→96 | 56／56 | 1 s |
| LBNL冷水机组仿真 | [下载](lbnl/prctf_seed42.pt) | 96→96 | 73／58 | 60 s |
| Tennessee Eastman过程 | [下载](tep/prctf_seed42.pt) | 20→20 | 52／52 | 180 s |

电解槽权重对应README中的**原始完整训练池实验**，不是后来仅使用12:00之后记录的
重训版本。三个模型分别使用适合各自系统的物理图和方程。

## 发布内容

每个`.pt`包含完整训练后参数、结构配置、物理图、输入通道顺序、预测目标、采样间隔，
以及训练时拟合的归一化与物理关系标定参数。为了支持推理，本次同时发布电解槽的这些
**聚合统计参数**，但不发布原始观测序列、作图源序列、优化器状态或论文文件。

[manifest.json](manifest.json)记录文件大小、SHA-256、原检查点哈希、训练轮次与验证误差。
加载器先核对哈希，再以`weights_only=True`读取，并以`strict=True`恢复全部模型状态。
本次不修改模型数学结构或归档结果，也不代替作者选择新的开源许可证；权利状态与
第三方条款见[LICENSE_STATUS.md](../LICENSE_STATUS.md)和
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。

## 安装与命令行推理

在仓库根目录运行，使用Python 3.10以上及合适的PyTorch版本（原代码接口要求≥2.7）：

```bash
pip install -r requirements-model.txt
# 先阅读THIRD_PARTY_NOTICES.md；已存在固定版本上游时无需重复获取。
python scripts/fetch_upstreams.py --ctf

python scripts/predict.py --dataset industrial \
  --input /path/to/history.csv --output outputs/forecast.csv --device cpu
```

GPU推理使用`--device cuda:0`。也可通过`--ctf-root /path/to/ChannelTokenFormer`
指定已有的固定版本上游源码。加载与推理不依赖原实验目录或私有数据缓存。

### 输入与输出

- CSV首行为通道ID，需包含`manifest.json`中该模型要求的全部输入通道。
  脚本按名称恢复顺序，忽略额外列。
- 取CSV最后96行（电解槽、LBNL）或20行（TEP）作为历史窗口。这些行必须时间连续、
  顺序正确且来自同一运行记录，采样间隔由调用方保证。不会填补缺失点或补造数据；
  历史长度不足、NaN及无穷值会报错。
- **电解槽：** 使用原始工作簿中的测量单位，不要预先标准化，也不要先低通滤波再传入
  这个原始数据检查点。通道ID见[数据结构](../configs/industrial/data_schema.json)，
  物理关系见[图配置](../configs/industrial/graph.json)。换装置部署前需核对测量定义。
- **LBNL：** 输入预处理后的SI量：温度°C、流量m³/s、功率W、压力Pa，以及无量纲控制量。
  原`ChillerPlant.csv`的单位不同，需先按[元数据](../configs/lbnl/data_metadata.json)
  中的`source_units`和[data.py](../workspace/PhysicsRelationCTF/runs/publicdata_exp/data.py)
  完成相同的单位换算，再调用推理。
- **TEP：** 使用原数据的测量单位及[52通道定义](../configs/tep/data_metadata.json)，
  不要把不同仿真运行拼接成一个历史窗口。

加载器自动使用训练时的统计量标准化，并在推理后逆标准化。输出CSV包含预测步编号、
提前秒数及各目标通道预测值，数值恢复为上述输入物理单位。已有输出文件不会被覆盖。

### Python接口

```python
import numpy as np
from scripts.pretrained import load_pretrained

predictor = load_pretrained("industrial", device="cpu")
history = np.load("/path/to/history.npy")  # 真实历史数据，形状[96, 56]
prediction = predictor.predict(history, channels=predictor.channels)  # [96, 56]
print(predictor.output_channels)
```

只有数组实际采用模型顺序时，才能填写`channels=predictor.channels`；否则应提供数组
真实列名，加载器会据此重排。也支持`[批次, 历史长度, 通道数]`输入。
LBNL默认只返回参与评价的58个物理目标；`targets_only=False`返回全部输出，但其余通道
并非全部作为监督预测目标训练。不同数据集按原代码的模块约定，在独立进程中加载。

## 核验命令

```bash
python scripts/verify_package.py
PRCTF_TEST_DATASET=industrial python -m unittest discover -s tests -p test_pretrained.py
PRCTF_TEST_DATASET=lbnl python -m unittest discover -s tests -p test_pretrained.py
PRCTF_TEST_DATASET=tep python -m unittest discover -s tests -p test_pretrained.py
```

测试使用合成输入核对完整状态加载、标准化、通道顺序、输出目标、CSV与Python接口一致性，
以及异常输入处理。不构成新增预测精度或故障预警实验。模型输出未来测量轨迹；下游预警
阈值与报警决策不属于当前推理接口。
