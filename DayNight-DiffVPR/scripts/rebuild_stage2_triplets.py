"""After manually auditing generated.csv, rebuild triplets using accepted=1 rows only."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from daynight_diffvpr.stage2_generate import build_training_triplets


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',default='outputs/stage2/generated.csv')
    p.add_argument('--output',default='outputs/stage2/triplets.csv')
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    with open(args.csv,encoding='utf8',newline='') as f:
        rows=list(csv.DictReader(f))
    accepted=[r for r in rows if str(r['accepted']).strip().lower() in {'1','true','yes'}]
    count=build_training_triplets(accepted,args.output,args.seed)
    print(f'Saved {count} accepted source/synthetic/negative triplets to {args.output}')

if __name__=='__main__':
    main()
