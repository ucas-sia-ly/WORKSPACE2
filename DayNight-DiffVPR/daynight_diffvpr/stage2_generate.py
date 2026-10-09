"""Stage 2: SD1.5 + Canny ControlNet img2img offline day->night generation.

Example: python -m daynight_diffvpr.stage2_generate --config configs/stage2_generate.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from tqdm.auto import tqdm

from .diffusion_common import (DEFAULT_BASE, DEFAULT_CONTROLNET, DEFAULT_PROMPT,
                               DEFAULT_NEGATIVE, pil_edge, edge_iou, seed_everything)


def load_day_rows(csv_path: str | Path, root: str | Path) -> list[dict]:
    """Require explicit training split and place labels to prevent accidental evaluation leakage."""
    root = Path(root).expanduser().resolve()
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        needed = {'id','path','place_id','split'}
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError('day_csv needs columns id,path,place_id,split')
        rows = list(reader)
    if not rows:
        raise ValueError('day_csv empty')
    ids = set()
    clean = []
    for row in rows:
        if row['split'].strip().lower() != 'train':
            raise ValueError(f"Refusing to synthesize split={row['split']} for {row['id']}. Provide TRAIN data only.")
        if not row['id'] or not row['place_id'] or not row['path']:
            raise ValueError('id/path/place_id must not be empty')
        if row['id'] in ids:
            raise ValueError(f"Duplicate image id {row['id']}")
        ids.add(row['id'])
        image_path = Path(row['path']).expanduser()
        if not image_path.is_absolute():
            image_path = root / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        clean.append({'id':row['id'], 'path':str(image_path), 'place_id':row['place_id']})
    return clean


def build_training_triplets(accepted: list[dict], output: str | Path, seed: int = 42) -> int:
    """Generated night -> original day as positive; different place day as negative."""
    import random
    rng = random.Random(seed)
    unique_places = {r['place_id'] for r in accepted}
    if len(unique_places) < 2:
        raise ValueError('Need >= 2 distinct accepted place IDs for negatives; generation CSV is still saved.')
    by_place: dict[str, list[dict]] = {}
    for r in accepted:
        by_place.setdefault(r['place_id'], []).append(r)
    triplets = []
    for row in accepted:
        other_place = rng.choice(sorted(unique_places - {row['place_id']}))
        negative = rng.choice(by_place[other_place])['source_path']
        triplets.append({'anchor':row['generated_path'], 'positive':row['source_path'], 'negative':negative})
    with Path(output).open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=['anchor','positive','negative'])
        writer.writeheader()
        writer.writerows(triplets)
    return len(triplets)


def load_pipeline(config: dict, device: str):
    from diffusers import (ControlNetModel, StableDiffusionControlNetImg2ImgPipeline,
                           UniPCMultistepScheduler)
    if not str(device).startswith('cuda'):
        raise ValueError('Real SD1.5 generation requires CUDA; use CPU only for unit tests.')
    g = config['generation']
    dtype = torch.float16  # fp16 saves memory in inference
    controlnet = ControlNetModel.from_pretrained(
        g.get('controlnet_id',DEFAULT_CONTROLNET), torch_dtype=dtype, use_safetensors=True)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        g.get('base_model',DEFAULT_BASE), controlnet=controlnet, torch_dtype=dtype,
        safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if g.get('lora_dir'):
        pipe.load_lora_weights(g['lora_dir'])
    if g.get('cpu_offload',True):
        pipe.enable_model_cpu_offload(gpu_id=int(str(device).split(':')[-1]) if ':' in str(device) else 0)
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=bool(g.get('quiet',True)))
    return pipe


def run(config: dict, output: Path, device: str = 'cuda') -> dict:
    output.mkdir(parents=True, exist_ok=True)
    (output/'images').mkdir(exist_ok=True)
    inp, settings = config['train_data'], config['generation']
    rows = load_day_rows(inp['day_csv'], inp['image_root'])
    max_images = int(settings.get('max_images',0))
    if max_images > 0:
        rows = rows[:max_images]
    seed = int(settings.get('seed',42))
    seed_everything(seed)
    pipe = load_pipeline(config, device)
    size = int(settings.get('resolution',512))
    if size % 8 or size < 128:
        raise ValueError('generation.resolution must be >=128 and divisible by 8')
    min_edge_iou = float(settings.get('min_edge_iou',0.0))
    if not 0 <= min_edge_iou <= 1:
        raise ValueError('min_edge_iou must lie in [0,1]')
    accepted = []
    history = []
    for n, row in enumerate(tqdm(rows, desc='Day->night')):
        with Image.open(row['path']) as src:
            day = src.convert('RGB').resize((size,size), Image.Resampling.BILINEAR)
        control = pil_edge(day, size,size, int(settings.get('canny_low',100)), int(settings.get('canny_high',200)))
        item_seed = seed+n
        generator = torch.Generator(device=device).manual_seed(item_seed)
        with torch.inference_mode():
            night = pipe(
                prompt=settings.get('prompt',DEFAULT_PROMPT),
                negative_prompt=settings.get('negative_prompt',DEFAULT_NEGATIVE),
                image=day, control_image=control,
                strength=float(settings.get('strength',.65)),
                guidance_scale=float(settings.get('guidance_scale',6.0)),
                num_inference_steps=int(settings.get('steps',25)),
                controlnet_conditioning_scale=float(settings.get('control_scale',.8)),
                generator=generator,
            ).images[0]
        iou = edge_iou(day,night)
        keep = iou >= min_edge_iou
        generated_path = output/'images'/f'{n:06d}.png'
        # Always preserve candidates for audit; only accepted are sent to training.
        night.save(generated_path)
        record = {'source_id':row['id'], 'source_path':row['path'],
                  'generated_path':str(generated_path.resolve()), 'place_id':row['place_id'],
                  'seed':item_seed, 'edge_iou':round(iou,5), 'accepted':int(keep)}
        history.append(record)
        if keep:
            accepted.append(record)
    with (output/'generated.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=['source_id','source_path','generated_path','place_id','seed','edge_iou','accepted'])
        writer.writeheader()
        writer.writerows(history)
    n_triplets = 0
    if len({r['place_id'] for r in accepted}) >= 2:
        n_triplets = build_training_triplets(accepted, output/'triplets.csv',seed)
    result = {'candidates':len(history),'accepted':len(accepted),'triplets':n_triplets,
              'warning':'Edge IoU is only a heuristic. Manually review structure before VPR training.',
              'safety_checker_disabled':True}
    (output/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False), encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', default='outputs/stage2')
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text(encoding='utf-8')),Path(args.output),args.device)


if __name__ == '__main__':
    main()
