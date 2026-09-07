"""Common-shape inference microbenchmark; not end-to-end controller latency."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'lewm'))
import numpy as np
import torch
from mylewm.train_rbg import digest


def measure(operation,device,repeats,warmup=5):
    synchronize=lambda:torch.cuda.synchronize(device) if device.type=='cuda' else None
    for _ in range(warmup):operation()
    synchronize()
    baseline=None
    if device.type=='cuda':
        baseline=torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    elapsed=[]
    for _ in range(repeats):
        synchronize();start=time.perf_counter()
        operation();synchronize()
        elapsed.append((time.perf_counter()-start)*1000)
    return {'median_ms':float(np.median(elapsed)),'p95_ms':float(np.percentile(elapsed,95)),
            'repeats':repeats,'cuda_allocated_before_bytes':baseline,
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if baseline is not None else None}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--benchmark',choices=['pusht','libero10'],default='pusht')
    p.add_argument('--device',default='cpu')
    p.add_argument('--repeats',type=int,default=30)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.repeats<1:p.error('Require repeats >= 1')
    torch.set_num_threads(4);torch.manual_seed(771)
    device=torch.device(args.device)
    model=torch.load(args.checkpoint,map_location=device,weights_only=False).eval()
    model.requires_grad_(False)
    views=2 if args.benchmark=='libero10' else 1
    shape=(1,1,2,3,224,224) if views==2 else (1,1,3,224,224)
    pixels=torch.randn(shape,device=device)
    action_dim=28 if views==2 else 10
    z=torch.randn(1,3,192,device=device);a=torch.randn(1,3,action_dim,device=device)
    with torch.inference_mode():
        encoding=measure(lambda:model.encode({'pixels':pixels}),device,args.repeats)
        prediction=measure(lambda:model.predict(z,model.action_encoder(a)),device,args.repeats)
    report={'checkpoint_sha256':digest(args.checkpoint),'source_sha256':digest(Path(__file__)),
            'protocol':vars(args),'torch':torch.__version__,'platform':platform.platform(),
            'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),
            'threads':torch.get_num_threads(),'dtype':'float32; autocast disabled',
            'parameters':sum(p.numel() for p in model.parameters()),
            'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
            'buffer_bytes':sum(b.numel()*b.element_size() for b in model.buffers()),
            'checkpoint_bytes':args.checkpoint.stat().st_size,
            'encoder_one_observation':encoding,'predictor_one_history_window':prediction,
            'scope':'Synthetic preprocessed inputs; batch 1; excludes environment, preprocessing, CEM search and other GPU processes. CPU measurements are not GPU throughput.'}
    with args.output.open('x') as f:json.dump(report,f,default=str,indent=2)
    print(json.dumps(report,default=str),flush=True)


if __name__=='__main__':main()
