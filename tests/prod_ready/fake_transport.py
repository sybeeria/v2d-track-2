#!/usr/bin/env python3
"""Local-only provider filesystem/process model; no real provider/SSH executable is called."""
import fnmatch,json,os,re,shutil,signal,subprocess,sys,time
from pathlib import Path
home=Path(os.environ['TRANSPORT_HOME']);workspace=home/'workspace';tool=Path(sys.argv[0]).name;args=sys.argv[1:]
def event(kind,**kw):
    with (home/'transport.jsonl').open('a') as f:f.write(json.dumps({'kind':kind,**kw})+'\n')
def remap(s):return re.sub(r'(?<![A-Za-z0-9_./-])/workspace',str(workspace),s)
if tool=='fake_provider':
    if args[:2]==['show','instance']:print(json.dumps({'start_date':time.time(),'dph_total':1.26,'public_ipaddr':'offline.invalid','ports':{'22/tcp':[{'HostPort':'9'}]}}))
    elif args[:2]==['show','user']:print(json.dumps({'credit':10}))
    elif args[:2]==['destroy','instance']:event('destroy')
elif tool=='fake_remote':
    cmd=remap(args[-1]);event('remote',command=cmd)
    if 'setsid nohup' in cmd:
        name=next(workspace.glob('*.jobs')).stem
        with (workspace/f'{name}.log').open('w') as log:
            proc=subprocess.Popen(['bash',str(workspace/f'{name}.sh')],cwd=workspace,stdout=log,stderr=log,start_new_session=True)
        (home/'box-process').write_text(str(proc.pid))
        # A replacement box overwrites box-process; keep every group so stop_box also reaps replaced boxes.
        with (home/'box-processes').open('a') as f:f.write(f'{proc.pid}\n')
        print('launched')
    elif 'read -r pid' in cmd:
        event('quiesce')
        try:os.killpg(int((home/'box-process').read_text()),signal.SIGTERM)
        except ProcessLookupError:pass
    else:
        result=subprocess.run(['bash','-c',cmd],text=True);sys.exit(result.returncode)
elif tool=='fake_sync':
    sources=[];i=0;excludes=[];selection=None
    while i<len(args)-1:
        a=args[i]
        if a in ('-e','--exclude'):i+=1;excludes.extend([args[i]] if a=='--exclude' else [])
        elif a.startswith('--files-from='):selection=Path(a.split('=',1)[1]).read_text().splitlines()
        elif not a.startswith('-'):sources.append(a)
        i+=1
    dest=args[-1]
    def local(s):return Path(remap(s.split(':',1)[1])) if s.startswith('root@') else Path(s)
    target=local(dest);event('sync',sources=sources,dest=dest)
    for value in sources:
        src=local(value)
        if '*' in str(src):
            import glob
            paths=[Path(s) for s in glob.glob(str(src))]
        else:paths=[src]
        for src in paths:
            if not src.exists():sys.exit(23)
            if src.is_dir():
                base=target if value.endswith('/') else target/src.name
                for path in sorted(src.rglob('*')):
                    relative=path.relative_to(src)
                    if selection is not None and relative.as_posix() not in selection:continue
                    if any(any(fnmatch.fnmatch(part,pat) for part in relative.parts) for pat in excludes):continue
                    if path.is_dir():continue
                    out=base/relative;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,out)
            else:
                out=target/src.name if dest.endswith('/') or target.is_dir() else target
                out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,out)
    # Mapping only the remote filesystem namespace keeps all uploaded script bodies real.
    if dest.startswith('root@'):
        for p in workspace.glob('*.sh'):p.write_text(remap(p.read_text()))
        remote=workspace/'v2d-track-2'
        if os.environ.get('TRANSPORT_CORRUPT_CANDIDATE') and not (home/'candidate-corrupted').exists():
            candidate=next(remote.glob('runs/*/input-assets/assets/**/data.parquet'),None)
            if candidate is not None:
                candidate.write_bytes(candidate.read_bytes()+b'changed in transport')
                (home/'candidate-corrupted').touch()
        if (remote/'scripts/vast').exists():
            # GPU provision/download/install is a fixture boundary; it cannot run on this Mac.
            (remote/'scripts/gpu_bootstrap.sh').write_text('echo "GPU provision fixture BOOTSTRAP_TIERS=$BOOTSTRAP_TIERS"\n')
            for p in (remote/'scripts/vast').glob('*.sh'):p.write_text(remap(p.read_text()))
        if (remote/'payload-reference-paths').exists():pass
elif tool=='fake_pause':
    if args[0]=='600':
        (home/'periodic-pull').touch()
        while True:time.sleep(.05)
    else:time.sleep(.1)
elif tool=='flock':
    import fcntl
    try:fcntl.flock(int(args[-1]),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:sys.exit(1)
elif tool=='nvidia-smi':print('GPU 0: fixture')

elif tool=='wc':print(len(sys.stdin.buffer.read().splitlines()))
