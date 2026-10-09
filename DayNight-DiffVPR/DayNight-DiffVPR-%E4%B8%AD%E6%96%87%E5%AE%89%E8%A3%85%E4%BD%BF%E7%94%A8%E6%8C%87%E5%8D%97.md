# DayNight-DiffVPR：RTX 4090 一步一步运行（中文）

> 状态说明：Stage 1、Stage 2、Stage 3 的 Python 入口与训练逻辑均在此仓库实现；
> CI/CPU 测试检验脚本、CSV 与自动微分。**尚未在这次交付环境中下载 6GB+ 的模型或在 RTX 4090 上实际训练。**
> 因此不提供虚构的 R@1、显存峰值或结果图片。请先跑三张图的真实 GPU smoke test。

## 0. 你需要准备什么？

| 类型 | 必需？ | 获取方式 | 用途 |
|---|---|---|---|
| 本工程 | 必需 | 下载本次 ZIP 并解压 | Stage 1–3 程序/配置 |
| Python 3.10 / Conda | 必需 | https://docs.conda.io/ | 环境 |
| NVIDIA GPU 驱动 | 必需 | https://www.nvidia.com/Download/ | CUDA，4090 24 GB |
| CUDA 版 PyTorch | 必需 | https://pytorch.org/get-started/previous-versions/ | GPU 计算 |
| SD 1.5（Diffusers 版） | Stage 2、3 | https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 | 生成器；首次运行会自动下载 |
| Canny ControlNet | Stage 2、3 | https://huggingface.co/lllyasviel/sd-controlnet-canny | 控制道路和建筑边缘；首次运行自动下载 |
| 已有 VPR 模型 + 权重 | 必需 | 你现有的 VPR 项目或 SALAD | 检索 Backbone/aggregator |
| 带 place ID 的白天训练图 | Stage 2 | 自己的 GSV/RobotCar 等训练划分 | `train_day.csv` |
| 同地点昼夜真实图像 + 异地点负图 | Stage 3 | 如 RobotCar 的训练划分；用户自行匹配 | `train_pairs.csv` |
| 带 GT 的 VPR 测试集 | 定量评测 | Tokyo 24/7 等；按数据集许可获取 | 比较 E0/E2/E3 |

模型权重**没有打包在 ZIP**，首次 `.from_pretrained()` 将从 Hugging Face 下载。
磁盘建议至少留 **30 GB** 供模型、缓存、图像和中间结果；数据集本身另计。
不同模型仓库需要遵守对应的使用许可，访问受限时先登录 Hugging Face 账号并接受条款。

## 1. 安装环境（推荐 Ubuntu 22.04 / Python 3.10）

```bash
unzip DayNight-DiffVPR-all-stages.zip
cd DayNight-DiffVPR
conda env create -f environment.yml
conda activate daynight-diffvpr

# 适合支持 CUDA 12.1 wheels 的驱动；按你电脑的驱动选择 PyTorch 官方正确指令。
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[diffusion,test]'
python scripts/doctor.py
pytest -q
```

如果使用官方 `salad_hub`，而不是你原来的模型，还需要：

```bash
python -m pip install -e '.[salad]'
```

**环境要点**：`torch.cuda.is_available()` 应为 `True`，GPU 应显示 RTX 4090。
如果已有良好的 CUDA/PyTorch 环境，不要贸然覆盖原环境；新建一个隔离环境最好。

`diffusers==0.32.2` 是本项目的目标 API 版本。不建议使用 `pip install -U` 将
所有库无上限升级，也不要用没做兼容验证的最新 diffusers API 混跑旧工程。

## 2. 接入已有 VPR（只改 1 个适配文件）

在原有 VPR 仓库创建 `your_adapter.py`：

```python
from my_existing_vpr.my_model import MyVPR  # 换成你的真实 import

def build_model():
    model = MyVPR()  # 换成你的真实构造参数
    return model
```

修改 `configs/stage1_existing.yaml`、`configs/stage2_vpr_train.yaml` 与
`configs/stage3_joint.yaml` 的 `model` 相关字段：

```yaml
model:
  type: existing
  pythonpath: /home/me/my-vpr-repo
  factory: your_adapter:build_model
  checkpoint: /home/me/weights/pretrained.pth
  # state_dict_key: model_state_dict   # 视自己 checkpoint 结构而定
  # output_key: global_descriptor     # 若 forward 返回字典
```

要求：`forward(image)` 返回 `[B,D]` 描述子；本工程统一 L2 normalize。
三份配置的 `data.image_size/normalize` 必须与**你的 VPR 原始预训练代码一致**。
若 VPR 封装自行完成预处理，可以改 `daynight_diffvpr/images.py` 和
`diffusion_common.differentiable_vpr_input` 两处，保证 Stage 1、2、3 完全一致。

如果 Stage3 使用 SALAD：把 `model.type` 改成 `salad_hub`，去掉 `factory`、`pythonpath`；
但首先必须在 Stage1 证明该构造器正常加载，SALAD 的仓库依赖可能随版本变化。

## 3. Stage 1：先取得 E0 基线

```bash
python scripts/create_demo.py
python -m daynight_diffvpr.evaluate \
  --config configs/stage1_demo.yaml --device cpu --output outputs/demo
```

这个 demo 用合成颜色图，只检查代码，不代表真实 VPR 性能。

为了评估真实昼夜检索，需要按 [`STAGE1_ORIGINAL.md`](STAGE1_ORIGINAL.md) 准备 `database.csv`、`queries.csv`、`positives.csv`，
并修改 `configs/stage1_existing.yaml` 的 `data` 路径。这里 `queries.csv` 的 `domain` 是
`night` / `day` / `unknown`；未知标签不可伪称夜晚测试。

```bash
python -m daynight_diffvpr.evaluate \
  --config configs/stage1_existing.yaml --device cuda \
  --output outputs/E0
```

检查 `outputs/E0/metrics.json`、`retrieval_topk.csv`。**固定这份测试集，Stage 2/3 严禁用它训练。**

## 4. Stage 2：生成夜景 + 离线 VPR 增强（E2）

### 4.1 准备白天训练清单

格式参考 `examples/train_day.csv`：

```csv
id,path,place_id,split
A_day,city/stationA/day1.jpg,stationA,train
B_day,city/stationB/day1.jpg,stationB,train
```

`path` 是相对 `train_data.image_root` 的路径，也可以是绝对路径。
`place_id` 必须是**不同地点的唯一地点标签**，不是“白天/夜晚”的域标签。
**至少两个不同地点**才能自动生成异地点负样本。
`split` 必须为 `train`，程序拒绝 `test` / `val`。

修改 `configs/stage2_generate.yaml`：

```yaml
train_data:
  image_root: /data/robotcar_train
  day_csv: data/train_day.csv
generation:
  resolution: 512
  max_images: 3      # 第一次只跑三张做 GPU smoke test
  steps: 25
  cpu_offload: true
```

执行：

```bash
python -m daynight_diffvpr.stage2_generate \
  --config configs/stage2_generate.yaml \
  --output outputs/stage2 --device cuda
python scripts/preview_stage2.py \
  --csv outputs/stage2/generated.csv \
  --output outputs/stage2/preview.jpg
```

输出：

```text
outputs/stage2/
├── images/000000.png, ...  # 合成夜景
├── generated.csv           # 原图/合成图/地点/seed/edge IoU/accepted
├── triplets.csv            # anchor=生成夜晚，positive=原始白天，negative=异地白天
├── preview.jpg             # 人工核查：逐张看建筑、道路有无变化
└── summary.json
```

默认 `min_edge_iou=0`：只记录结构启发指标，不据它删除数据。
Edge IoU **不是几何准确度证明**，夜间灯光变化可能改变边缘。请检查
`preview.jpg` 并人工清除严重改变路口、房屋、路牌的样本。
可以直接将 `generated.csv` 中错误生成样本的 `accepted` 改成 `0`，然后运行：

```bash
python scripts/rebuild_stage2_triplets.py --csv outputs/stage2/generated.csv \
  --output outputs/stage2/triplets.csv
```

该命令会严格使用人工接受的样本重建 E2 Triplet CSV；保留生成种子等完整历史记录。

### 4.2 用合成夜景训练 VPR（E2）

修改 `configs/stage2_vpr_train.yaml` 中现有 VPR 模型/参数，将
`training.triplets_csv: outputs/stage2/triplets.csv` 保持不变，再执行：

```bash
python -m daynight_diffvpr.train_vpr \
  --config configs/stage2_vpr_train.yaml --device cuda \
  --output outputs/E2_train
```

保存 `outputs/E2_train/vpr_last.pth`。
把 `configs/stage1_existing.yaml` 的 `model.checkpoint` 改成此路径：

```bash
python -m daynight_diffvpr.evaluate \
  --config configs/stage1_existing.yaml --device cuda \
  --output outputs/E2_eval
```

比较 `outputs/E0/metrics.json` 与 `outputs/E2_eval/metrics.json`：

```bash
python scripts/compare_metrics.py outputs/E0/metrics.json outputs/E2_eval/metrics.json
```
建议显式保证 E0/E2 使用相同 VPR 模型初始化与同一测试清单。

## 5. Stage 3：Diffusion + VPR 联合优化（E3）

Stage 3 **必须额外准备真实昼夜配对训练数据**，没有真实夜景就不能声称
“真实夜景 DDPM 去噪监督”。配对以**同地点**为准，最好有大致对齐的几何结构。

CSV 格式参考 `examples/train_pairs.csv`：

```csv
day,night,negative,place_id,negative_place_id,split
city/stationA/day1.jpg,city/stationA/night1.jpg,city/stationB/day1.jpg,stationA,stationB,train
city/stationB/day1.jpg,city/stationB/night1.jpg,city/stationA/day1.jpg,stationB,stationA,train
```

- `day`：当前地点白天照片
- `night`：**同地点真实夜晚**照片，不能只是 day 的复制文件
- `negative`：其他地点（不同 `place_id`）的照片
- 所有路径相对 `data.image_root`，必须是训练 split；程序校验 `split=train` 及负样本不同地点

修改 `configs/stage3_joint.yaml` 的 `model`、`data.image_root`、
`joint.pairs_csv`；第一次跑 2 个训练 step：

```yaml
diffusion:
  resolution: 256
  lora_rank: 4
  gradient_checkpointing: true
joint:
  pairs_csv: data/train_pairs.csv
  batch_size: 1
  max_steps: 2
  vpr_device: cpu
  save_every: 2
```

然后执行：

```bash
python -m daynight_diffvpr.stage3_joint \
  --config configs/stage3_joint.yaml --device cuda \
  --output outputs/E3_smoke
```

验证两份权重与日志生成、loss 有限（非 NaN）后，再把 `max_steps` 改成 100/1000
进行对照实验（实际步数依数据量、损失趋势与验证集决定）。

**4090 节省显存关键点：**
`resolution=256`、`batch_size=1`、`lora_rank=4`、冻结 ControlNet/VAE/文本编码器、
UNet gradient checkpointing，`vpr_device=cpu`（把 VPR 放 CPU，前向和反传可能很慢）。
若显存空闲且希望提速，试 `vpr_device=cuda`；OOM 就退回 CPU。不要假设“4090 必定不会 OOM”。
生成器两次反传串行进行，降低两个 UNet/VPR 大图同时存活时的峰值。

生成输出：

```text
outputs/E3_smoke/
├── diffusion_lora/
│   └── pytorch_lora_weights.safetensors    # Diffusers 可加载的 UNet LoRA
├── vpr_last.pth                            # 训练后的 VPR 权重
├── joint_training.jsonl                    # diff/place/struct/illum/vpr loss
└── config_snapshot.yaml
```

**真实多步生成检查**：在 `configs/stage2_generate.yaml` 加入：

```yaml
generation:
  lora_dir: outputs/E3_smoke/diffusion_lora
```

再次运行 Stage 2，**换一个输出目录**，检查同一白天输入生成的夜景是否保持地点。
最终 E3 在原始真实昼夜测试集评测：将 Stage1 的 `model.checkpoint` 指向
`outputs/E3_smoke/vpr_last.pth`，运行 `daynight_diffvpr.evaluate`。
测试时**不要把合成夜景替换为真实查询**却仍声称是夜间真实检索。

## 6. E4/E5 消融

复制 `configs/stage3_joint.yaml` 到独立文件，保持所有数据、seed、step 不变：

- E4：`joint.lambda_place: 0.0`
- E5：`joint.lambda_struct: 0.0`

输出各用不同目录，固定 E0/E2/E3/E4/E5 的 query、database、GT 和 R@K。
建议三次随机种子后报告均值/标准差。

## 7. 遇到报错优先看这里

- **`CUDA available: False`**：确认 NVIDIA driver 和安装的是 CUDA 版 Torch，而不是 CPU wheel。
- **`ModuleNotFoundError: diffusers`**：`pip install -e '.[diffusion,test]'`。
- **`403` / 权重下载失败**：登录 Hugging Face、检查模型页许可证、网络代理和磁盘空间。
- **`CUDA out of memory`**：Stage 3 先 256 / batch1 / vpr CPU；Stage2 使用 `cpu_offload: true`，减少分辨率和步数。
- **`VPR model must return BxD`**：适配器没有返回地点描述子；检查 `model.factory` 和 `model.output_key`。
- **`No trainable parameters remain`**：`freeze_modules` 把聚合器也冻住了；改成你真实模型的内部子模块名。
- **`No valid night labels`**：不要自己猜 Tokyo24/7 的 night/day 标签；使用官方或可信数据集标注。
- **`Stage3 must contain split ...`**：必须使用非测试的真实训练图像，且负样本地点与 anchor 不同。
- **`NaN` 或结构严重漂移**：降低 `proxy_timestep`（如 300）、`lr_lora`，减小权重 `lambda_place`，检查配对对齐。
- **Diffusers/PEFT 版本冲突**：按本工程 `pyproject.toml` 限制重装，不要任意升级最前沿 API。

## 8. 联合优化实现的研究边界

Stage 3 的生成反馈是 **single-step differentiable predicted-x0 proxy**，
不是端到端完全展开的 25 步 diffusion 反向传播。后者在 24GB 显存条件下
成本高得多，需要额外显存优化/截断反传。Stage 3 依赖同地点真实昼夜对，
VPR 梯度可作用于 LoRA，但好坏必须用原始真实夜间测试集验证。

本文与模型权重的下载和第三方使用许可分别遵守各自发布方条款。
