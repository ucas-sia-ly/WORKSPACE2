# DayNight-DiffVPR — Stage 1 + Stage 2 + Stage 3

**项目：昼夜跨域视觉地点识别 (VPR) 与任务驱动 Diffusion 联合训练。**
适用 RTX 4090 24GB + 现有 VPR 代码。实现三个阶段：

- **Stage 1：** 模型适配、CPU smoke test、Recall@1/5/10、VPR Triplet 微调。
- **Stage 2：** SD 1.5 + Canny ControlNet **真实多步** day-to-night img2img、输出元数据、
  结构相似性分数、图片对比审核、自动产生合成夜景 VPR triplets（E2）。
- **Stage 3：** 真实夜景扩散噪声预测 + 白天输入预测 night x0 的**可微单步训练代理**；
  VPR place / contrastive / structure / illumination 损失回传到 LoRA，交替更新 VPR（E3）。

## ✅ 最完整的中文安装与运行说明

**直接阅读 [`docs/INSTALL_AND_RUN_CN.md`](docs/INSTALL_AND_RUN_CN.md)**。
含下载模型/数据清单、配置、环境、命令、输出、4090 显存与常见报错。

> 诚实的验证状态：本工程的 CPU 数据集、生成接口 mock、梯度端到端单测已通过；
> 此交付环境没有 NVIDIA GPU，未下载 SD1.5/ControlNet/VPR 权重，
> 所以尚未完成真实 4090 训练/评测，亦不包含已训练 LoRA 或性能数字。

## 目录

```text
DayNight-DiffVPR/
├── daynight_diffvpr/
│   ├── evaluate.py, train_vpr.py        # Stage 1 + E2 offline VPR trainer
│   ├── stage2_generate.py               # Stage 2: image generation + manifest
│   ├── diffusion_common.py              # day/night preprocess + differentiable math
│   ├── stage3_joint.py                  # Stage 3: G-step and VPR-step
│   ├── manifest.py, images.py, models.py, retrieval.py
├── configs/
│   ├── stage1_existing.yaml, stage1_salad.yaml, stage1_demo.yaml
│   ├── stage1_triplet_train.yaml
│   ├── stage2_generate.yaml, stage2_vpr_train.yaml
│   └── stage3_joint.yaml
├── examples/                           # CSV templates (example paths)
├── scripts/
│   ├── doctor.py, preview_stage2.py
│   ├── create_demo.py, prepare_tokyo247.py
├── tests/                              # Stage 1–3 CPU tests
├── docs/
│   ├── INSTALL_AND_RUN_CN.md
│   └── stage2_stage3_plan.md
├── environment.yml, pyproject.toml
```

## 1-minute overview

```bash
conda env create -f environment.yml
conda activate daynight-diffvpr
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e '.[diffusion,test]'
python scripts/doctor.py
pytest -q

# After EDITING model / paths / CSV in YAML:
python -m daynight_diffvpr.evaluate --config configs/stage1_existing.yaml --device cuda --output outputs/E0
python -m daynight_diffvpr.stage2_generate --config configs/stage2_generate.yaml --output outputs/stage2
python scripts/preview_stage2.py --csv outputs/stage2/generated.csv
python -m daynight_diffvpr.train_vpr --config configs/stage2_vpr_train.yaml --output outputs/E2_train
python -m daynight_diffvpr.stage3_joint --config configs/stage3_joint.yaml --output outputs/E3_joint
```

**Important:** Above GPU commands are not valid until you have provided your existing VPR code adapter,
training/benchmark data and accessible Hugging Face models. The examples CSVs reference placeholders.
The Tokyo24/7 dataset should be used for evaluation, **not training**.

## References

- https://huggingface.co/docs/diffusers/v0.32.1/en/api/pipelines/controlnet
- https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
- https://huggingface.co/lllyasviel/sd-controlnet-canny
- https://github.com/serizba/salad
