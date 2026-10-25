import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]

def put(path,text):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wait(path,timeout=15):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if path.exists():return
        time.sleep(.05)
    raise AssertionError(f'missing {path}')

class Fixture:
    def __init__(self,work):
        self.work=work;self.controller=work/'controller';self.home=work/'transport';self.workspace=self.home/'workspace';self.remote=self.workspace/'v2d-track-2'
        self.home.mkdir();self.controller.mkdir();self.events=work/'gpu-events.jsonl'
        for name in ['scripts','patches']:shutil.copytree(BASE/name,self.controller/name)
        bindir=work/'bin';bindir.mkdir()
        for name in ['fake_provider','fake_remote','fake_sync','fake_pause','flock','nvidia-smi','wc']:
            p=bindir/name;shutil.copyfile(HERE/'fake_transport.py',p);p.chmod(0o755)
        import re
        for name in ['launch.sh','watchdog.sh','start_run.sh']:
            p=self.controller/'scripts/vast'/name;s=p.read_text()
            for a,b in [('vastai','fake_provider'),('ssh','fake_remote'),('rsync','fake_sync'),('sleep','fake_pause')]:s=re.sub(r'(?<!\.)\b'+a+r'\b',b,s)
            p.write_text(s)
        (self.controller/'data/denoise_eval').mkdir(parents=True);put(self.controller/'data/denoise_eval/fixture_config.json','{}')
        self.env={**os.environ,'PATH':str(bindir)+os.pathsep+os.environ['PATH'],'TRANSPORT_HOME':str(self.home),
                  'GPU_PROJECT':str(self.remote),'GPU_EVENTS':str(self.events),'PACK_PYTHON':sys.executable,
                  'PYTHON':str(self.remote/'.venv-flash/bin/python')}
        self.provision()
    def provision(self):
        self.remote.mkdir(parents=True,exist_ok=True)
        for venv in ['.venv','.venv-flash']:
            wrapper=self.remote/venv/'bin/python'
            put(wrapper,f'#!{sys.executable}\nimport os,sys,runpy\n'
                "if len(sys.argv)>1 and sys.argv[1].endswith('make_tier2_screen.py'):sys.exit('unrequested screen setup')\n"
                "if len(sys.argv)>1 and (sys.argv[1]=='-' or sys.argv[1].endswith(('train_flash_sac.py','evaluate_policy.py'))):\n"
                f" runpy.run_path({str(HERE/'fake_gpu.py')!r},run_name='__main__')\n"
                "else:os.execv(sys.executable,[sys.executable,*sys.argv[1:]])\n")
            wrapper.chmod(0o755)
        kit=self.remote/'external/kit/v2d_submission_kit'
        shutil.copytree(BASE/'external/kit/v2d_submission_kit',kit,dirs_exist_ok=True)
        fake=kit/'flash_chord/scripts/evaluate_policy.py'
        fake.write_text("import sys,runpy\nsys.argv.insert(1,sys.argv[0])\n"+f"runpy.run_path({str(HERE/'fake_gpu.py')!r},run_name='__main__')\n")
        upstream=self.remote/'external/video_to_data';upstream.parent.mkdir(exist_ok=True)
        upstream.symlink_to(BASE/'external/video_to_data',target_is_directory=True)
        sample=pd.read_csv(kit/'data/track_2_sample_submission.csv',dtype=str).iloc[:20].copy()
        sample.row_id=[f'track_2/e{e:06d}/w000/f{f:06d}/o0' for e in range(2) for f in range(10)]
        sample.to_csv(kit/'data/track_2_sample_submission.csv',index=False)
        # The fake evaluator and 20-row roster make this a different kit; keep its own manifest consistent with it.
        manifest=json.loads((kit/'KIT_MANIFEST.json').read_text())
        for name in ['flash_chord/scripts/evaluate_policy.py','data/track_2_sample_submission.csv']:
            if name in manifest['files']:manifest['files'][name]=sha(kit/name)
        (kit/'KIT_MANIFEST.json').write_text(json.dumps(manifest))
    def run_spec(self,tier,name):
        run=self.work/name;run.mkdir();payload=run/'payload'
        for directory in ['tier_1_multiview_caption','tier_2_synthetic_noise','baseline']:
            base=payload/'data/challenge/track_2'/directory
            put(base/'meta/info.json',json.dumps({'total_episodes':2,'chunks_size':1000,'data_path':'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'}))
            canonical=base/'mesh/object/object_visual.glb';canonical.parent.mkdir(parents=True,exist_ok=True)
            trimesh.creation.box().export(canonical)
            records=[{'episode_index':e,'sequence_id':f'seq{e}','frames':10,'length':10,
                      'mesh':'mesh/object/object_visual.glb'} for e in range(2)]
            text=''.join(json.dumps(r)+'\n' for r in records)
            put(base/'meta/episodes_metadata.jsonl',text);put(base/'meta/episodes.jsonl',text)
            for e in range(2):
                poses=np.tile([0.,0.,0.,1.,0.,0.,0.],(10,1))
                table=pa.table({'frame_index':np.arange(10),'observation.object.pose':pa.array(poses.tolist(),type=pa.list_(pa.float64(),7)),
                                'observation.object.visible':np.ones(10,dtype=bool)}).replace_schema_metadata({b'objects':b'["object"]'})
                path=base/f'data/chunk-000/episode_{e:06d}.parquet';path.parent.mkdir(parents=True,exist_ok=True);pq.write_table(table,path)
        label={1:'clean',2:'raw_repaired',3:'tier3_synthetic'}[tier]
        arm='F6-rebalance-tail' if tier==1 else 'T2C-contact';total=349999104 if tier==1 else 49999872
        source=next(line for line in (BASE/'scripts'/('pilot_arms_350m.txt' if tier==1 else 'pilot_arms.txt')).read_text().splitlines() if line.startswith(arm+' '))
        put(self.controller/f'scripts/{arm}.txt',source+'\n')
        manifest={'tier':tier,'scoring_mode':'self','sequences':[]}
        mesh=payload/'data/fixture/mesh.ply';mesh.parent.mkdir(parents=True,exist_ok=True);trimesh.creation.box().export(mesh)
        mesh_remote=self.remote/mesh.relative_to(payload)
        for e in range(2):
            seq=f'seq{e}';base=payload/f'data/fixture/tier{tier}';ref=base/f'motion/whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet';ref.parent.mkdir(parents=True,exist_ok=True)
            pq.write_table(pa.Table.from_pylist([{'schema_version':'motion_v1','episode_index':e,'object_body_names':['object'],
                'object_mesh_paths':[str(mesh_remote)],'object_body_position':np.zeros((10,1,3)).tolist(),
                'object_body_wxyz':np.tile([1.,0.,0.,0.],(10,1,1)).tolist()}]),ref)
            poses=base/'sequences'/seq/'poses.npy';poses.parent.mkdir(parents=True,exist_ok=True);np.save(poses,np.tile(np.eye(4),(10,1,1)))
            support=base/f'motion/whole_body/reconstructed_stage/{seq}_support.usda';put(support,'#usda 1.0\ndef Xform "support_surfaces" {}\n')
            record={'path':str(self.remote/ref.relative_to(payload)),'sha256':sha(ref)}
            manifest['sequences'].append({'sequence':seq,'arms':{label:{'training_reference':record,'clean_scoring_reference':record,
                'support':{'path':str(self.remote/support.relative_to(payload)),'sha256':sha(support),'enabled':False}}},'assets':[{'path':str(mesh_remote),'sha256':sha(mesh)}]})
        put(payload/f'data/fixture/manifest{tier}.json',json.dumps(manifest))
        put(run/'env.sh','export BOOTSTRAP_TIERS="tier1 tier2"\nexport BUILD_SCREEN=0\n')
        put(run/'pre.sh',f'export MANIFEST="{self.remote}/data/fixture/manifest{tier}.json"\n'+
            f'archive="runs/{name}/frozen-inputs"\n'+
            'if [ -f "$archive/manifest.json" ]; then .venv/bin/python scripts/freeze_inputs.py restore --archive "$archive"; fi\n'+
            f'.venv/bin/python scripts/freeze_inputs.py capture --manifest "$MANIFEST" --work data/fixture/tier{tier} --archive "$archive"\n')
        put(run/'jobs.txt',''.join(f't{tier}-ep{e}|{arm}.txt|{arm}|{label}|seq{e}|'+('330012672 349999104' if tier==1 else '49999872')+'\n' for e in range(2)))
        return run,label,arm
    def command(self,args,env=None,timeout=40):
        return subprocess.run(args,cwd=self.controller,env={**self.env,**(env or {})},capture_output=True,text=True,timeout=timeout)
    def instance(self,run):
        # One rented instance per run directory, as in production; a relaunch of the same run keeps its instance.
        ids=self.__dict__.setdefault('instance_ids',{})
        return ids.setdefault(Path(run).resolve(),str(123+len(ids)))
    def launch(self,run,env=None):
        p=self.command(['bash',str(self.controller/'scripts/vast/launch.sh'),str(run),self.instance(run)],env)
        if p.returncode:raise AssertionError(p.stdout+p.stderr)
    def finished(self,run):
        log=self.workspace/f'{run.name}.log';end=time.monotonic()+60
        while time.monotonic()<end:
            if log.exists() and ('box done ' in log.read_text() or 'box failed ' in log.read_text()):break
            time.sleep(.05)
        text=log.read_text() if log.exists() else 'no log'
        if 'box done ' not in text:raise AssertionError(text)
    def stop_box(self):
        p=self.home/'box-processes'
        if p.exists():
            import signal
            for group in map(int,p.read_text().split()):
                try:
                    os.killpg(group,signal.SIGTERM);time.sleep(.15);os.killpg(group,signal.SIGKILL)
                except ProcessLookupError:pass
    def calls(self):return list(map(json.loads,self.events.read_text().splitlines()))
