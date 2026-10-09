"""Print side-by-side Recall@K from Stage 1/2/3 benchmark JSON files."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def compare(paths):
    for path in paths:
        data=json.loads(Path(path).read_text(encoding='utf8'))
        name=Path(path).parent.name
        metrics=data.get('metrics',{})
        print(f'\n{name} ({path})')
        for group,vals in metrics.items():
            nums=' | '.join(f'{key}:{float(val):.4f}' for key,val in vals.items() if key.startswith('R@'))
            print(f'  {group:10s} n={vals.get("queries",0)} | {nums}')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('metrics',nargs='+',help='metrics.json files, e.g. outputs/E0/metrics.json')
    args=p.parse_args()
    compare(args.metrics)
