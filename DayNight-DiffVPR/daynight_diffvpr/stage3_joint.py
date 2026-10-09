"""Stage 3: paired night-domain diffusion LoRA + differentiable day->night VPR feedback.

Uses single-step x0 *training proxy*, NOT fully unrolled diffusion sampling.
Offline Stage 2 evaluates actual multi-step generation with the learned LoRA.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from PIL import Image
import yaml
from tqdm.auto import tqdm

from .diffusion_common import (DEFAULT_BASE,DEFAULT_CONTROLNET,DEFAULT_PROMPT,
    predict_x0, tensor_canny, differentiable_vpr_input, structure_loss,
    illumination_loss, save_lora_weights, seed_everything)
from .models import build_model


class PairedDayNightTriplets(Dataset):
    """Supervised same-place day/night and negative (other place) samples.

    Accepts CSV: day,night,negative,place_id,negative_place_id,split.
    These are real aligned-by-place night observations, NOT Tokyo24/7 evaluation data.
    """
    def __init__(self, csv_path: str | Path, root: str | Path, resolution: int = 256):
        self.root = Path(root).expanduser().resolve()
        self.resolution = int(resolution)
        if self.resolution < 128 or self.resolution % 8:
            raise ValueError('resolution must be >=128 and divisible by 8')
        with open(csv_path, newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            required={'day','night','negative','place_id','negative_place_id','split'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError('pairs CSV must contain '+','.join(sorted(required)))
            self.rows = list(reader)
        if not self.rows:
            raise ValueError('No training pairs provided')
        for row in self.rows:
            if not all(row.get(k) for k in ('day','night','negative','place_id','negative_place_id')):
                raise ValueError('Empty day/night/negative path or place ID')
            if row['split'].strip().lower() != 'train':
                raise ValueError('Stage 3 accepts only split=train; never test or val data')
            if self._path(row['day']) == self._path(row['night']):
                raise ValueError('Day and night must be separate observations')
            if row['place_id'] == row['negative_place_id']:
                raise ValueError('Negative must have a different place_id from day/night')
            for key in ('day','night','negative'):
                if not self._path(row[key]).is_file():
                    raise FileNotFoundError(self._path(row[key]))

    def _path(self, rel: str) -> Path:
        p = Path(rel).expanduser()
        return (self.root / p).resolve() if not p.is_absolute() else p.resolve()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        images = []
        for name in ('day','night','negative'):
            with Image.open(self._path(self.rows[i][name])) as img:
                image = img.convert('RGB').resize((self.resolution,self.resolution), Image.Resampling.BILINEAR)
                images.append(TF.to_tensor(image))
        return tuple(images)


def cosine_triplet(anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, margin: float = 0.2):
    return F.softplus((anchor*neg).sum(dim=1)-(anchor*pos).sum(dim=1)+margin).mean()


def place_objective(day_desc, generated_desc, negative_desc, margin: float = .2):
    """VPR-guided generation is contrastive, not just feature collapse to the day anchor."""
    cos = (day_desc*generated_desc).sum(dim=1)
    return (1-cos).mean() + cosine_triplet(generated_desc,day_desc,negative_desc,margin)


def train_vpr_step(vpr: nn.Module, optimizer, config: dict, day: torch.Tensor,
                   real_night: torch.Tensor, negative: torch.Tensor,
                   generated_proxy: torch.Tensor, device: torch.device) -> torch.Tensor:
    """VPR gradients never reach generator here; real + synthetic positives are both used."""
    cfg = config['joint']
    optimizer.zero_grad(set_to_none=True)
    vpr.train()
    for name in cfg.get('freeze_vpr_modules', []):
        module = vpr.backbone
        for chunk in name.split('.'):
            module = getattr(module, chunk)
        module.eval()
    with torch.enable_grad():
        day_input = differentiable_vpr_input(day.detach(),config).to(device)
        night_input = differentiable_vpr_input(real_night.detach(),config).to(device)
        neg_input = differentiable_vpr_input(negative.detach(),config).to(device)
        syn_input = differentiable_vpr_input(generated_proxy.detach(),config).to(device)
        desc_d,desc_n,desc_neg,desc_syn = vpr(torch.cat([day_input,night_input,neg_input,syn_input],dim=0)).chunk(4)
        margin = float(cfg.get('margin',.2))
        real_loss = cosine_triplet(desc_d,desc_n,desc_neg,margin)
        syn_loss = cosine_triplet(desc_d,desc_syn,desc_neg,margin)
        loss = .5*(real_loss+syn_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in vpr.parameters() if p.requires_grad],float(cfg.get('grad_clip',1)))
        optimizer.step()
    return loss.detach()


class JointDiffusion:
    def __init__(self, cfg: dict, device: torch.device):
        from diffusers import (AutoencoderKL, ControlNetModel,
                               UNet2DConditionModel, DDPMScheduler)
        from transformers import CLIPTokenizer, CLIPTextModel
        from peft import LoraConfig
        self.device = device
        settings = cfg['diffusion']
        self.precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model_id = settings.get('base_model',DEFAULT_BASE)
        control_id = settings.get('controlnet_id',DEFAULT_CONTROLNET)
        self.unet = UNet2DConditionModel.from_pretrained(model_id,subfolder='unet',torch_dtype=torch.float32).to(device)
        self.unet.requires_grad_(False)
        self.unet.add_adapter(LoraConfig(r=int(settings.get('lora_rank',4)),
            lora_alpha=int(settings.get('lora_rank',4)), init_lora_weights='gaussian',
            target_modules=['to_q','to_k','to_v','to_out.0']))
        if bool(settings.get('gradient_checkpointing',True)):
            self.unet.enable_gradient_checkpointing()
        # frozen ControlNet, all inference residuals run under no_grad
        self.controlnet = ControlNetModel.from_pretrained(control_id,torch_dtype=self.precision).to(device)
        self.controlnet.requires_grad_(False).eval()
        # VAE decoder weights frozen but autograd through its input is preserved.
        self.vae = AutoencoderKL.from_pretrained(model_id,subfolder='vae',torch_dtype=torch.float32).to(device)
        self.vae.requires_grad_(False).eval()
        self.vae.enable_slicing()
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_id,subfolder='scheduler')
        tok = CLIPTokenizer.from_pretrained(model_id, subfolder='tokenizer')
        enc = CLIPTextModel.from_pretrained(model_id,subfolder='text_encoder',torch_dtype=self.precision).to(device).eval()
        enc.requires_grad_(False)
        prompt = settings.get('prompt',DEFAULT_PROMPT)
        ids = tok([prompt], padding='max_length', max_length=tok.model_max_length,
                  truncation=True,return_tensors='pt').input_ids.to(device)
        with torch.no_grad():
            self.text_embeddings = enc(ids)[0].detach()
        del enc
        if device.type=='cuda':
            torch.cuda.empty_cache()

    @property
    def trainable_params(self):
        return [p for p in self.unet.parameters() if p.requires_grad]

    def latents(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            with torch.autocast('cuda',dtype=self.precision):
                latents = self.vae.encode(images*2-1).latent_dist.sample()
        return latents.float()*self.vae.config.scaling_factor

    def _predict(self, latent: torch.Tensor, t: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        cond = control.to(dtype=self.precision)
        with torch.no_grad():
            residuals, middle = self.controlnet(latent.to(dtype=self.precision),t,
                encoder_hidden_states=self.text_embeddings,
                controlnet_cond=cond,return_dict=False)
        with torch.autocast('cuda',dtype=self.precision):
            pred = self.unet(latent,t,encoder_hidden_states=self.text_embeddings,
                down_block_additional_residuals=[r.to(dtype=self.precision) for r in residuals],
                mid_block_additional_residual=middle.to(dtype=self.precision)).sample
        return pred.float()

    def diffusion_loss(self, real_night: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        z = self.latents(real_night)
        noise = torch.randn_like(z)
        timesteps = torch.randint(0,self.noise_scheduler.config.num_train_timesteps,
                                  (z.shape[0],),device=self.device,dtype=torch.long)
        z_noisy = self.noise_scheduler.add_noise(z,noise,timesteps)
        pred = self._predict(z_noisy,timesteps,control)
        return F.mse_loss(pred,noise)

    def day_to_night_proxy(self, day: torch.Tensor, control: torch.Tensor, proxy_t: int) -> torch.Tensor:
        """Single epsilon->x0 from a noised DAY latent conditioned on NIGHT prompt+DAY edges.

        This is a differentiable approximation of the actual multi-step img2img
        translation; uses NO real-night image as the noisy image input.
        """
        z = self.latents(day)
        steps = torch.full((z.shape[0],),int(proxy_t),device=self.device,dtype=torch.long)
        z_noisy = self.noise_scheduler.add_noise(z,torch.randn_like(z),steps)
        pred = self._predict(z_noisy,steps,control)
        alpha = self.noise_scheduler.alphas_cumprod.to(self.device)[steps]
        z0 = predict_x0(z_noisy,pred,alpha)
        # Avoid decoder overflow for extreme epsilon estimates; limits gradient beyond threshold.
        z0 = z0.clamp(-8,8)
        with torch.autocast('cuda',dtype=self.precision):
            decoded = self.vae.decode(z0/self.vae.config.scaling_factor).sample
        return (decoded.float()/2+0.5).clamp(0,1)


def frozen_model_params(model: nn.Module):
    """Generator step: VPR parameters frozen, image-input gradient still flows."""
    original = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return original


def restore_model_params(model: nn.Module, original: list[bool]):
    for p, old in zip(model.parameters(),original):
        p.requires_grad_(old)


def generator_step(diff: JointDiffusion, vpr: nn.Module, optimizer,
                   config: dict, day, night, negative, vpr_device: torch.device):
    cfg = config['joint']
    optimizer.zero_grad(set_to_none=True)
    original = frozen_model_params(vpr)
    control = tensor_canny(day,int(cfg.get('canny_low',100)),int(cfg.get('canny_high',200)))
    try:
        # Two backwards keep the real-night denoising graph and VPR-input graph
        # from being alive simultaneously, reducing 4090 peak memory.
        loss_diff = diff.diffusion_loss(night,control)
        lambda_diff = float(cfg.get('lambda_diff',1))
        if lambda_diff:
            (lambda_diff*loss_diff).backward()
        else:
            loss_diff = loss_diff.detach()
        proxy = diff.day_to_night_proxy(day,control,int(cfg.get('proxy_timestep',400)))
        with torch.no_grad():
            feat_day = vpr(differentiable_vpr_input(day,config).to(vpr_device))
            feat_neg = vpr(differentiable_vpr_input(negative,config).to(vpr_device))
        feat_generated = vpr(differentiable_vpr_input(proxy,config).to(vpr_device))
        lp = place_objective(feat_day,feat_generated,feat_neg,float(cfg.get('margin',.2)))
        ls = structure_loss(day,proxy)
        li = illumination_loss(proxy,night)
        gen_loss = (float(cfg.get('lambda_place',1))*lp+
                    float(cfg.get('lambda_struct',.05))*ls+
                    float(cfg.get('lambda_illum',.1))*li)
        if not gen_loss.requires_grad:
            raise RuntimeError('No generator gradient: check trainable UNet LoRA parameters')
        gen_loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(diff.trainable_params,float(cfg.get('grad_clip',1)))
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite diffusion LoRA gradients. Reduce proxy_timestep / learning rate.')
        optimizer.step()
        losses = {'diff':float(loss_diff.detach()),'place':float(lp.detach()),
                  'struct':float(ls.detach()),'illum':float(li.detach()),
                  'generator':float((gen_loss+lambda_diff*loss_diff).detach())}
        return proxy.detach(),losses
    finally:
        restore_model_params(vpr,original)


def train(config: dict, output: Path, gpu: str = 'cuda'):
    if not torch.cuda.is_available() or not gpu.startswith('cuda'):
        raise RuntimeError('Stage 3 needs NVIDIA CUDA. CPU-only unit tests cover gradient math.')
    output.mkdir(parents=True,exist_ok=True)
    joint = config['joint']
    seed_everything(int(joint.get('seed',42)))
    device = torch.device(gpu)
    vpr_device = torch.device(joint.get('vpr_device','cpu'))
    dataset = PairedDayNightTriplets(joint['pairs_csv'],config['data']['image_root'],
                                    resolution=int(config['diffusion'].get('resolution',256)))
    loader = DataLoader(dataset,batch_size=int(joint.get('batch_size',1)),shuffle=True,
                        num_workers=int(joint.get('num_workers',2)),pin_memory=True)
    vpr = build_model(config).to(vpr_device)
    for name in joint.get('freeze_vpr_modules',[]):
        module = vpr.backbone
        for chunk in name.split('.'):
            module = getattr(module,chunk)
        module.requires_grad_(False)
    params_vpr = [p for p in vpr.parameters() if p.requires_grad]
    if not params_vpr:
        raise ValueError('VPR no trainable parameters. Adjust joint.freeze_vpr_modules.')
    diff = JointDiffusion(config,device)
    optim_g = torch.optim.AdamW(diff.trainable_params,lr=float(joint.get('lr_lora',1e-5)))
    optim_v = torch.optim.AdamW(params_vpr,lr=float(joint.get('lr_vpr',1e-5)))
    steps = int(joint.get('max_steps',100))
    if steps < 1:
        raise ValueError('max_steps must >=1')
    step = 0
    log_path = output/'joint_training.jsonl'
    diff.unet.train()
    while step < steps:
        for day,night,negative in tqdm(loader,desc=f'Joint {step}/{steps}'):
            day,night,negative = [x.to(device,non_blocking=True) for x in (day,night,negative)]
            proxy, losses = generator_step(diff,vpr,optim_g,config,day,night,negative,vpr_device)
            l_vpr = train_vpr_step(vpr,optim_v,config,day,night,negative,proxy,vpr_device)
            step += 1
            row = {'step':step,**losses,'vpr':float(l_vpr)}
            with log_path.open('a',encoding='utf8') as f:
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(row)
            del proxy
            if step % int(joint.get('save_every',50)) == 0 or step==steps:
                save_lora_weights(diff.unet,output/'diffusion_lora')
                torch.save({'model_state_dict':{k:v.detach().cpu() for k,v in vpr.backbone.state_dict().items()},
                            'step':step},output/'vpr_last.pth')
                (output/'config_snapshot.yaml').write_text(yaml.safe_dump(config,allow_unicode=True),encoding='utf8')
            if step>=steps:
                break
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    ap.add_argument('--output',default='outputs/stage3')
    ap.add_argument('--device',default='cuda')
    args = ap.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf8'))
    print('Outputs:',train(config,Path(args.output),args.device))


if __name__=='__main__':
    main()
