# DayNight-DiffVPR: Stage 1（可运行）

适用：RTX 4090（24GB）、已有 VPR 代码库。当前版本 **已实现 Stage 1 基线评测 + 可选 VPR triplet 微调**，
**尚未实现 Diffusion 生成和联合反传**（见 `docs/stage2_stage3_plan.md`）。

## 目录

```text
DayNight-DiffVPR/
├── daynight_diffvpr/
│   ├── manifest.py        # 数据 + GT 校验
│   ├── images.py          # 图像预处理、triplet dataset
│   ├── models.py          # 旧 VPR / SALAD / demo 模型适配
│   ├── retrieval.py       # memmap 特征 + 分块 exact top-k + Recall@K
│   ├── evaluate.py        # Stage 1 评测主入口
│   └── train_vpr.py       # Stage 1 可选 triplet 微调
├── scripts/
│   ├── create_demo.py     # 合成数据检验流程
│   └── prepare_tokyo247.py  # 已格式化 Tokyo24/7 -> CSV
├── configs/
│   ├── stage1_demo.yaml
│   ├── stage1_salad.yaml
│   ├── stage1_existing.yaml
│   └── stage1_triplet_train.yaml
├── tests/
├── docs/stage2_stage3_plan.md
├── pyproject.toml
└── environment.yml
```

## 0. 安装（Linux / NVIDIA / Python 3.10）

```bash
conda env create -f environment.yml
conda activate daynight-diffvpr
# 如果已有可用 PyTorch CUDA 环境，避免重新安装破坏现有依赖。
# 全新环境的示例（版本须与你的 CUDA 驱动兼容）：
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e '.[test]'
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

如果要使用**官方 SALAD torch.hub**（而不是你的现有 VPR），再安装：

```bash
pip install -e '.[salad]'
```

SALAD 首次 `torch.hub` 调用需要联网下载代码/权重，受上游依赖和 GitHub 可用性影响。上游 README 使用
`torch.hub.load("serizba/salad", "dinov2_salad")`，评测图像预处理为 ImageNet 标准化、322×322。
官方代码和其依赖若版本冲突，优先使用你已有的 SALAD 环境，通过 `existing` 适配器接入。

## 1. 运行不依赖任何真实模型的 smoke test

```bash
python scripts/create_demo.py --output data/demo
python -m daynight_diffvpr.evaluate --config configs/stage1_demo.yaml --device cpu --output outputs/demo
pytest -q
```

在正常运行下，合成颜色 demo 的 day/night R@1 是 100%，**不能视作 VPR 科学结果**。
该测试只验证图片加载、标注映射、Top-K、Recall 和保存格式。

## 2. 如何接入已有的 VPR 模型（推荐）

旧仓库里添加 `your_adapter.py`（或者在任意可 import 的目录创建），内容形如：

```python
from my_vpr.network import MyVPR  # 修改为已有模型的真实 import


def build_model():
    # 必须返回 torch.nn.Module
    return MyVPR()
```

将 `configs/stage1_existing.yaml` 中的 `model.pythonpath`、`model.factory`、`model.checkpoint`
及 `data` 路径改成自己的路径。如果模型已自行加载权重，可不填写 `model.checkpoint`。
如果 forward 返回 `dict`，在 YAML 里提供 `model.output_key`；tuple 使用第一个元素。
最终描述子必须是形状 `[B, D]` 的张量，系统自动 L2 归一化。
**输入 Resize、归一化必须与你已有模型的训练预处理一致**（可在 `images.build_transform()` 中修改）。

```bash
python -m daynight_diffvpr.evaluate \
  --config configs/stage1_existing.yaml \
  --device cuda --output outputs/E0_existing
```

## 3. Tokyo 24/7 数据集

### 获得数据

按数据集作者的许可获取原始 Tokyo24/7。该数据集主要用于 **测试**，不要参与训练/调参。
可先参照 [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader) 将数据整理为：

```text
/path/to/tokyo247/images/test/
  database/@easting@northing@zone@letter@....png
  queries/@easting@northing@zone@letter@....png
```

请确认这是该工具产出的**UTM 命名格式**，否则需自己构建 CSV 与 ground-truth 正样本。

### 构造 CSV 标注（25m 半径）

```bash
python scripts/prepare_tokyo247.py --root /path/to/tokyo247 \
  --output data/tokyo247_manifests --radius-m 25
```

输出三份 CSV：

```text
database.csv: id,path,domain
queries.csv: id,path,domain
positives.csv: query_id,db_id
```

`positives.csv` 用查询点与数据库点的 UTM 欧氏距离 ≤ 25 米构建；若查询无合法正样本则明确报错。
如果原始 benchmark 提供官方 ground-truth，复现实验时**优先使用原始官方定义**并对齐数量和有效查询集合。

**特别注意**：脚本无法只凭 UTM 文件名判断 day/night，默认 `domain=unknown`，绝不伪造“夜间 Recall”。
如有可信的时间标注，准备 `query_domains.csv`：

```csv
filename,domain
@300000@3900000@54@S@sample_1.png,night
@300050@3900050@54@S@sample_2.png,day
```

再生成：

```bash
python scripts/prepare_tokyo247.py --root /path/to/tokyo247 \
  --output data/tokyo247_manifests --query-domains query_domains.csv
```

记得修改 YAML 的 `data.image_root` 为数据实际根目录。

### E0 Baseline

```bash
python -m daynight_diffvpr.evaluate --config configs/stage1_salad.yaml \
  --device cuda --batch-size 16 --output outputs/E0_salad
```

结果：`metrics.json`（包含 all / night / day 分组，前提是时间标签存在）、`retrieval_topk.csv`，
`database.npy` 与 `queries.npy`。分块矩阵乘法为**精确余弦 Top-K**，不依赖 FAISS ANN。
上游 SALAD 的官方模型加载与 ImageNet 预处理参见其仓库：
https://github.com/serizba/salad

## 4. 可选：Stage 1 VPR triplet 微调

训练数据来自其他训练集（例如 RobotCar 训练分区），CSV：

```csv
anchor,positive,negative
route1/night_001.jpg,route1/day_001.jpg,route1/day_205.jpg
```

其中 positive 必须是**同地点**的可靠配对，negative 应是**不同地点**。
禁止使用 Tokyo24/7 测试图片、同一地理地点泄漏到测试的图像训练。
编辑 `configs/stage1_triplet_train.yaml` 中数据路径，运行：

```bash
python -m daynight_diffvpr.train_vpr --config configs/stage1_triplet_train.yaml \
  --device cuda --output outputs/stage1_train
```

默认冻结 `backbone`，仅更新 aggregator，以控制显存；`batch_size=2`、`accumulate=4`、
自动 BF16（GPU 支持时）、梯度裁剪。保存 `vpr_last.pth` 和 `training.jsonl`。
这是简单的**监督 triplet 对比目标**，不保证 SOTA 性能；用于与你的现有训练方法做接口验证。

评测微调权重：在你已有的 `model` 配置中添加：

```yaml
checkpoint: outputs/stage1_train/vpr_last.pth
```

该 checkpoint 由 `model_state_dict` 包装，Stage 1 加载器会自动解包。

## 5. 可复现设置与常见问题

- 测试集 ground-truth 必须真实可信；只修改夜间输入不改正样本定义。
- `image_size=[322,322]` 是 SALAD 默认；已有其他模型必须使用其自身预处理。
- 4090 OOM：先将评测 `batch_size` 从 16 降至 4，再将训练微批从 2 降至 1。
- PyTorch CUDA 与 NVIDIA 驱动要匹配；`torch.cuda.is_available()` 必须为 `True`。
- 本工程未自动下载任何 Tokyo 24/7 图像，未捆绑 SALAD 权重与外部数据。
- 外部 checkpoint 只加载可信文件；大模型权重和第三方模型可能受许可约束。

## 6. 下一步（Stage 2 / Stage 3）

详见 `docs/stage2_stage3_plan.md`。先固化 **E0 原始 Recall@K**，才开始 Diffusion 生成。
Stage 2 增加离线 daylight→night 生成与 E2 对照；Stage 3 建立可微采样链路、VPR place loss
并对 Diffusion LoRA 与 VPR 交替优化。在 Stage 1 中未假装生成结果或训练好的联合模型已存在。
