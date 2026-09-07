"""Create one untrained initialization for all methods; never ingest old weights."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from mylewm.train_rbg import build_model,atomic_save
from mylewm.training_state import tensor_state_hash


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=3072)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    model=build_model()
    state=model.state_dict()
    artifact={'schema':'untrained_shared_initialization_v2','seed':args.seed,
              'model':state,'model_sha256':tensor_state_hash(state)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_save(artifact,args.output)
    print(artifact['model_sha256'],flush=True)


if __name__=='__main__':main()
