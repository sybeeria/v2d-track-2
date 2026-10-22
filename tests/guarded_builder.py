"""Test-only guard: block clean-data and kit-packer reads through Python and Arrow APIs."""
import json
import runpy
import sys
from pathlib import Path
import pyarrow.parquet as pq

calls=[]
def check(path):
    if not isinstance(path,(str,bytes,Path)): return
    p=Path(path.decode() if isinstance(path,bytes) else path).resolve()
    if any('tier_1' in part or part=='baseline' for part in p.parts) or p.name.startswith('pack_submission.'):
        raise PermissionError('forbidden clean-data or kit-packer access')
    if p.suffix in {'.parquet','.npz','.csv'}: calls.append(str(p))
def audit(event,args):
    if event=='open': check(args[0])
sys.addaudithook(audit)
original_file,original_read=pq.ParquetFile,pq.read_table
def parquet_file(source,*args,**kwargs):
    check(source);return original_file(source,*args,**kwargs)
def read_table(source,*args,**kwargs):
    check(source);return original_read(source,*args,**kwargs)
pq.ParquetFile,pq.read_table=parquet_file,read_table
for callback in [lambda:open('/not-present/tier_1_multiview_caption/data/episode_000000.parquet'),
                 lambda:pq.read_table('/not-present/tier_1_multiview_caption/data/episode_000000.parquet'),
                 lambda:open('/not-present/pack_submission.py')]:
    try: callback()
    except PermissionError: pass
    else: raise AssertionError('access guard canary did not fail')
script=Path(sys.argv[1]);sys.argv=sys.argv[1:]
try:
    runpy.run_path(str(script),run_name='__main__')
finally:
    Path('guard-observed-paths.json').write_text(json.dumps(sorted(set(calls)),indent=2)+'\n')
