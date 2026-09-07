"""Read-only checkpoint diagnostics; never recalibrates the saved model."""
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'lewm'))
import torch
from mylewm.train_rbg import Clips,preprocess,digest
from mylewm.rbg import BlockSIGReg
from mylewm.calibrate_rbg_bn import calibrate, parameter_hash
from mylewm.bn_diagnostics import bn_statistics,candidate_invariance


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'.cache/stable-wm/pusht/rbg_v0/manifest.json')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--calibration-batches',type=int,default=32)
    p.add_argument('--calibration-batch-size',type=int,default=128)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(4); torch.manual_seed(1234)
    m=json.loads(args.manifest.read_text()); ds=Clips(m,True)
    x,a=preprocess(torch.utils.data.default_collate([ds[i] for i in range(args.batch_size)]),m,'cpu')
    original=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    calibrated=copy.deepcopy(original)
    calibration=calibrate(calibrated,Clips(m),m,preprocess,args.calibration_batches,
                          args.calibration_batch_size,'cpu')
    reg=BlockSIGReg()
    reports={}
    with torch.no_grad():
        # Only batchnorm mode changes in second branch. Keep dropout disabled.
        for mode in ('eval','batchnorm_batch_statistics','train_only_calibrated_eval'):
            model=copy.deepcopy(calibrated if mode=='train_only_calibrated_eval' else original)
            model.eval()
            if mode=='batchnorm_batch_statistics':
                for layer in model.modules():
                    if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm): layer.train()
            stats,handles=bn_statistics(model)
            z=model.encode({'pixels':x})['emb']
            pred=model.predict(z[:,:3],model.action_encoder(a))
            for handle in handles:handle.remove()
            shuffled=model.predict(z[:,:3],model.action_encoder(a.roll(1,0)))
            torch.manual_seed(1234)
            gaussian,cross=reg(z.transpose(0,1))
            means=z.float().mean(0)
            stds=z.float().std(0)
            reports[mode]={'prediction':float((pred-z[:,1:]).square().mean()),
                'bn_statistics':stats,
                'shuffled_action_prediction':float((shuffled-z[:,1:]).square().mean()),
                'action_effect_mse':float((shuffled-pred).square().mean()),
                'gaussian':float(gaussian),'cross':float(cross),
                'mean_coordinate_rms':float(means.square().mean().sqrt()),
                'mean_std':float(stds.mean()),
                'block_mean_rms':[float(v.square().mean().sqrt()) for v in means.chunk(4,-1)]}
            if mode!='batchnorm_batch_statistics':
                reports[mode]['candidate_invariance']=candidate_invariance(model,x,a)
    report={'checkpoint_sha256':digest(args.checkpoint),'batch_size':args.batch_size,
            'manifest_sha256':digest(args.manifest),'calibration':calibration,
            'weights_equal_all_routes':parameter_hash(original)==parameter_hash(calibrated),
            'main_evaluation_rule':'report saved and train-only-calibrated eval separately; never select by test success',
            'note':'Same validation clips; dropout disabled in both branches. BN branch is diagnostic only; saved weights unchanged.',
            'results':reports}
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({'output':str(args.output),'weights_equal_all_routes':report['weights_equal_all_routes'],
        'results':{name:{key:value for key,value in row.items() if key!='bn_statistics'}
                   for name,row in reports.items()}),flush=True)


if __name__=='__main__': main()
