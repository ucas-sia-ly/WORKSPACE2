"""Generate a source/generated contact sheet to manually verify geometry.

python scripts/preview_stage2.py --csv outputs/stage2/generated.csv --output outputs/stage2/preview.jpg
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from PIL import Image, ImageDraw


def create_preview(manifest: str | Path, output: str | Path, max_rows: int = 16, tile: int = 256):
    with open(manifest, newline='', encoding='utf8') as f:
        rows=list(csv.DictReader(f))[:max_rows]
    if not rows:
        raise ValueError('No generated images')
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    canvas=Image.new('RGB',(tile*2,(tile+28)*len(rows)),'white')
    painter=ImageDraw.Draw(canvas)
    for i,row in enumerate(rows):
        y=i*(tile+28)
        for col,path in enumerate((row['source_path'],row['generated_path'])):
            with Image.open(path) as im:
                im=im.convert('RGB')
                im.thumbnail((tile,tile))
                canvas.paste(im,(col*tile,y))
        painter.text((4,y+tile+4),f"{row['source_id']} edge IoU {row['edge_iou']} accepted {row['accepted']}",fill='black')
    canvas.save(output,quality=90)
    return output


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',default='outputs/stage2/generated.csv')
    ap.add_argument('--output',default='outputs/stage2/preview.jpg')
    ap.add_argument('--max-rows',type=int,default=16)
    args=ap.parse_args()
    print(create_preview(args.csv,args.output,args.max_rows))
