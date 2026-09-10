"""Check native demo-state renders against recorded images; no training writes."""
import argparse
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from mylewm.paths import ROOT, libero_paths
for p in libero_paths(): sys.path.insert(0,str(p))
os.environ.setdefault('LIBERO_CONFIG_PATH',str(ROOT/'.cache/libero-config'))
os.environ.setdefault('MUJOCO_GL','egl')
import h5py
import numpy as np
from PIL import Image
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils.utils import postprocess_model_xml


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=ROOT/'.cache/libero-datasets/libero_10',
                        help='Directory containing the ten task HDF5 files')
    parser.add_argument('--output',type=Path,default=ROOT/'output/libero10/image_audit')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--task-id',type=int,choices=range(10),
                       help='Official LIBERO-10 task ID; use this for evaluation audits')
    group.add_argument('--task-index',type=int,choices=range(10),
                       help='Legacy index into lexically sorted local HDF5 files')
    parser.add_argument('--max-mae',type=float,default=10.)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    files=sorted(args.dataset.expanduser().glob('*.hdf5'))
    if len(files) != 10:
        parser.error('--dataset must contain the ten LIBERO-10 HDF5 files')
    if args.task_id is not None:
        name=benchmark.get_benchmark_dict()['libero_10']().get_task(args.task_id).name
        matches=[p for p in files if p.stem.removesuffix('_demo')==name]
        if len(matches)!=1: raise FileNotFoundError(f'Expected one HDF5 for task ID {args.task_id}: {name}')
        p=matches[0]
    else:
        p=files[0 if args.task_index is None else args.task_index]
    with h5py.File(p) as f:
        demo=f['data/demo_0']
        name=p.stem.removesuffix('_demo')
        env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/'libero_10'/f'{name}.bddl'),
                              camera_heights=128,camera_widths=128)
        try:
            tree=ET.fromstring(postprocess_model_xml(demo.attrs['model_file']))
            for node in tree.find('asset'):
                filename=node.get('file')
                if filename and not Path(filename).exists():
                    if '/assets/' not in filename: raise FileNotFoundError(filename)
                    new=Path(get_libero_path('assets'))/filename.split('/assets/',1)[1]
                    if not new.exists(): raise FileNotFoundError(new)
                    node.set('file',str(new))
            env.reset(); env.reset_from_xml_string(ET.tostring(tree,encoding='unicode'))
            rows=[]
            for t in (0,4,8):
                obs=env.set_init_state(demo['states'][t])
                for recorded,rendered in [('agentview_rgb','agentview_image'),('eye_in_hand_rgb','robot0_eye_in_hand_image')]:
                    target=demo[f'obs/{recorded}'][t].astype(np.float32)
                    image=obs[rendered].astype(np.float32)
                    if t==0:
                        target_path=args.output/f'{recorded}_recorded.png'
                        render_path=args.output/f'{recorded}_rendered.png'
                        if not target_path.exists(): Image.fromarray(target.astype(np.uint8)).save(target_path)
                        if not render_path.exists(): Image.fromarray(image.astype(np.uint8)).save(render_path)
                    rows.append({'t':t,'camera':recorded,'mae_native':float(np.abs(image-target).mean()),
                                 'mae_vertical_flip':float(np.abs(image[::-1]-target).mean()),
                                 'mae_horizontal_flip':float(np.abs(image[:,::-1]-target).mean())})
            report={'task':name,'mujoco_state_replay':'stored model XML; only asset file locations rewritten',
                    'comparisons':rows}
            import mujoco
            report['mujoco']=mujoco.__version__
            report['render_backend']=os.environ['MUJOCO_GL']
            report['max_mae_threshold']=args.max_mae
            report['passed']=all(r['mae_native']<=args.max_mae and
                r['mae_native']<r['mae_vertical_flip'] for r in rows)
            output=args.output/'report.json'
            if not output.exists(): output.write_text(json.dumps(report,indent=2))
            print(json.dumps(report),flush=True)
            if not report['passed']: raise RuntimeError('Demo-state image audit failed')
        finally: env.close()


if __name__=='__main__': main()
