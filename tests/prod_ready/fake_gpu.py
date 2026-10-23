"""Only the CUDA trainer/evaluator boundary; fixture checkpoints are JSON, never real policies."""
import hashlib,json,os,re,sys,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT=Path(os.environ['GPU_PROJECT'])
args=dict(a.split('=',1) for a in sys.argv[2:] if '=' in a)
def event(kind,**kw):
    with open(os.environ['GPU_EVENTS'],'a') as f:f.write(json.dumps({'kind':kind,'pid':os.getpid(),**kw})+'\n')
if sys.argv[1]=='-':print('false');sys.exit(0)
if sys.argv[1].endswith('train_flash_sac.py'):
    out=Path(args['output_dir'].strip("'"));out.mkdir(parents=True,exist_ok=True)
    ref=args['task.parquet'].strip("'");total=int(args['training.total_environment_steps'])
    event('train',out=str(out),resume=args.get('training.resume'),state=args.get('training.checkpoint_path'))
    policy=json.dumps({'reference':ref,'total':total})
    if total==349999104:
        for step in [310011904,330012672]:
            (out/f'policy_{step}.safetensors').write_text(policy)
            (out/f'state_{step}.safetensors').write_text(policy)
    else:
        (out/'policy_40001536.safetensors').write_text(policy);(out/'state_40001536.safetensors').write_text(policy)
    if os.environ.get('GPU_INTERRUPT') and args.get('training.resume')!='true':
        Path(os.environ['GPU_INTERRUPT']).touch()
        while True:time.sleep(.02)
    (out/f'policy_{total}.safetensors').write_text(policy);(out/f'state_{total}.safetensors').write_text(policy)
    print('training progress complete',flush=True)
else:
    checkpoint=Path(args['evaluation.checkpoint']);record=json.loads(checkpoint.read_text());step=int(checkpoint.stem.split('_')[1])
    row=pq.ParquetFile(record['reference']).read().to_pylist()[0]
    ref=np.c_[np.asarray(row['object_body_position'])[:,0],np.asarray(row['object_body_wxyz'])[:,0,[1,2,3,0]]]
    path=Path(args['evaluation.object_trajectories_output']);path.parent.mkdir(parents=True,exist_ok=True)
    repeat=next((m[1] for part in path.parts if (m:=re.fullmatch(r'.+-r([0-9]+)',part))),'00')
    offset=.01
    if step==330012672:offset=0 if repeat=='01' else .02
    if os.environ.get('GPU_FINAL_WORSE') and step==349999104:offset=.03
    if os.environ.get('GPU_BAD_REPEAT')==repeat:offset=.2
    achieved=ref.copy();achieved[:,0]+=offset
    if os.environ.get('GPU_EP18_DROP50') and step==49999872 and repeat=='02':
        achieved[10:20,2]=ref[0,2]-.201
    sys.path.insert(0,str(ROOT/'external/kit/v2d_submission_kit'))
    from v2dlb.submission_parquet import write_example
    n=len(ref);columns={k:np.zeros(n,dtype=np.int32) for k in ['environment_id','object_id','object_body_id']}
    columns.update(step=np.arange(n,dtype=np.int32),reference_frame=np.arange(n,dtype=np.int32),time_s=np.arange(n)/50,
                   environment_valid=np.ones(n,dtype=bool),object_body_name=np.array(['object']*n))
    suffix=['position_x','position_y','position_z','quaternion_x','quaternion_y','quaternion_z','quaternion_w']
    for prefix,poses in [('achieved',achieved),('reference',ref)]:
        for i,name in enumerate(suffix):columns[prefix+'_'+name]=poses[:,i]
    hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest();episode=int(row.get('episode_index',18))
    write_example(path,columns,episode_index=episode,eval_hash='a'*64,code_hash='b'*64,checkpoint_hash=hash)
    metrics={'checkpoint_sha256':hash,'world_count':1,'reset_mode':'explicit','environment_steps':step,'non_finite_worlds':[]}
    Path(args['evaluation.metrics_output']).write_text(json.dumps(metrics))
    event('evaluate',checkpoint=str(checkpoint),rollout=str(path),step=step,repeat=repeat)
