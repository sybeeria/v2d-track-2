#!/usr/bin/env python3
"""Split a generated jobs file across N single-GPU hosts: JOBS N OUTDIR -> OUTDIR/<stem>-host<k>-of<N>.txt.
Round-robin in file order (same rule as generate_jobs.py's host0/host1), and refuses anything but an exact partition."""
import sys
from pathlib import Path
jobs,n,out=Path(sys.argv[1]),int(sys.argv[2]),Path(sys.argv[3])
lines=[l for l in jobs.read_text().splitlines() if l and not l.startswith('#')]
names=[l.split('|')[0] for l in lines];sequences=[l.split('|')[4] for l in lines]
if n<1 or n>len(lines) or len(set(names))!=len(names) or len(set(sequences))!=len(sequences):sys.exit('need 1<=N<=jobs and unique names/sequences')
parts=[lines[k::n] for k in range(n)]
if sorted(sum(parts,[]))!=sorted(lines):sys.exit('split is not an exact partition')
out.mkdir(parents=True,exist_ok=True)
for k,part in enumerate(parts):
    target=out/f'{jobs.stem}-host{k}-of{n}.txt'
    if target.exists():sys.exit(f'{target} exists; use a fresh directory')
    target.write_text('\n'.join(part)+'\n');print(target,len(part),' '.join(l.split('|')[0] for l in part))
