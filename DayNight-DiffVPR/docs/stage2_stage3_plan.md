# Stage 2 / Stage 3 implemented interfaces and honest limitations

The architecture below IS IMPLEMENTED as runnable Python entrypoints. It has CPU
integration/gradient tests, but large diffusion checkpoints were NOT downloaded
or trained on an RTX 4090 in this execution environment.

## Stage 2 (offline E2)

`python -m daynight_diffvpr.stage2_generate --config configs/stage2_generate.yaml --output outputs/stage2`

- SD 1.5 img2img with Canny ControlNet; source = day image, condition = day image Canny,
  text = night style, pretrained model used without LoRA initially.
- Saves all candidates and explicit `accepted` metadata; heuristic edge IoU
  reports geometry similarity but does not guarantee same place.
- Optional `min_edge_iou > 0` filtering; default 0 = OFF to avoid accidental rejection.
- `triplets.csv`: synthetic night anchor, original day positive, different-place day negative.
- `scripts/preview_stage2.py`: side-by-side source/generated visual audit.
- Stage 1 VPR triplet trainer runs E2 on this CSV; real/night test unchanged.

## Stage 3 (task-driven LoRA E3)

`python -m daynight_diffvpr.stage3_joint --config configs/stage3_joint.yaml --output outputs/stage3`

A supervised pair CSV includes registered-by-place day, real night and different-place negative;
`split=train` is REQUIRED. During each step:

1. Real paired night latent receives DDPM noise; frozen ControlNet uses day Canny; UNet LoRA learns
   to predict noise under the *night* prompt. `L_diff` has a real night-domain target.
2. A separately noisy *day* latent with day Canny and night prompt gets a **single-step x0 proxy**:
   `z0=(zt-sqrt(1-a_t)*eps_theta(zt,t,night,day_canny))/sqrt(a_t)`. VAE decodes to RGB
   with gradients enabled. Frozen VPR features of the day, negative and proxy yield contrastive
   place loss, backpropagating to LoRA. Normalized Sobel + real-night global color statistics
   supply soft structure/illumination losses. G-step optimizes LoRA only.
3. VPR-step sees day, real night, different-place negative and detached proxy; updates
   trainable VPR weights only. G and V steps have independent optimizers.
4. Save `diffusion_lora/pytorch_lora_weights.safetensors` and `vpr_last.pth`.
5. For actual multi-step inference, load Stage3 LoRA through Stage2 `generation.lora_dir`,
   regenerate synthetic night images; evaluate VPR weights on original real-night benchmark.

**Limitations and experimental distinctions**:
- x0 is an epsilon-derived ONE-STEP training proxy, not a fully differentiated K-step
  DDIM/ControlNet/img2img unroll. Do not describe it as that; actual Stage2 inference is multi-step.
- Real-night pairing needs enough geographic alignment for paired Canny conditioning.
- Losses constrain but cannot guarantee preservation of place geometry or night appearance.
- Runtime/memory on a real RTX 4090 has not been measured. `resolution=256`,
  `batch_size=1`, bf16, ControlNet/VAE frozen, UNet LoRA rank=4,
  gradient checkpointing, CPU VPR, separated backwards minimize *expected* memory.
- `vpr_device=cpu` sends generated VPR input from GPU to CPU with autograd intact.
  It saves GPU memory but may be extremely slow. Try `cuda` if VRAM permits.
- The LoRA saved after joint training cannot be interpreted as scientifically better
  until E0/E2/E3 are evaluated with the same held-out protocol and multiple seeds.
- This code does not implement checkpoint/resume of optimizer state nor automatic
  structural registration, mask estimation, hard-negative mining, or hyperparam sweep.

## Ablations

- E0: original VPR, no generated training.
- E1: stage1 VPR fine-tuned on REAL night/day train triplets.
- E2: pretrained diffusion Stage2 offline-generated triplets -> VPR fine-tune.
- E3: Stage3 joint VPR checkpoint + learned LoRA; benchmark real night queries.
- E4: Stage3 same settings except lambda_place=0.
- E5: Stage3 same settings except lambda_struct=0.

Never use Tokyo24/7 query images as synthesis training seeds, LoRA targets, or VPR fine-tuning images.
