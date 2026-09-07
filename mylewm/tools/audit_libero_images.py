"""Check native demo-state renders against recorded images; no training writes."""
import argparse
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT/'external/libero',Path(os.environ.get('LIBERO_MUJOCO_PATH',ROOT/'.cache/libero-runtime'))): sys.path.insert(0,str(p))
os.environ.setdefault('LIBERO_CONFIG_PATH',str(ROOT/'.cache/libero-config'))
os.environ.setdefault('MUJOCO_GL','egl')
import h5py
import numpy as np
from PIL import Image
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils.utils import postprocess_model_xml


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'.cache/stable-wm/libero10/rbg_v0/image_audit_v2')
    parser.add_argument('--task-index',type=int,default=0,choices=range(10))
    parser.add_argument('--export-textures',action='store_true')
    parser.add_argument('--max-mae',type=float,default=10.)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    p=sorted((ROOT/'.cache/libero-datasets/libero_10').glob('*.hdf5'))[args.task_index]
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
                if t==0 and args.export_textures:
                    model=env.sim.model._model
                    texdata=model.tex_rgb if hasattr(model,'tex_rgb') else model.tex_data
                    for tid in range(model.ntex):
                        width,height=int(model.tex_width[tid]),int(model.tex_height[tid])
                        channels=int(model.tex_nchannel[tid]) if hasattr(model,'tex_nchannel') else 3
                        start=int(model.tex_adr[tid])
                        texture=texdata[start:start+width*height*channels].reshape(height,width,channels)
                        if width>100:
                            Image.fromarray(texture).save(args.output/f'texture_{tid}.png')
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
