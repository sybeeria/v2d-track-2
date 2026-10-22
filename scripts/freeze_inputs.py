#!/usr/bin/env python3
"""Keep the exact policy-input closure inside the existing watchdog's run archive."""
import argparse,hashlib,json,shlex,shutil,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def safe(path):
    path=Path(path).resolve();relative=path.relative_to(ROOT)
    if relative.parts[0]!='data' and relative.parts[:5]!=('external','video_to_data','robotic_grounding','flash_chord','src'):
        raise ValueError(f'input closure outside data/runtime assets: {path}')
    if '..' in relative.parts:raise ValueError('unsafe input path')
    return relative

def capture(args):
    manifest=json.loads(args.manifest.read_text());pending=[]
    for entry in manifest['sequences']:
        for arm in entry['arms'].values():
            for key in ('training_reference','clean_scoring_reference','support'):
                r=arm[key];p=Path(r['path'])
                if digest(p)!=r['sha256']:raise ValueError(f'input hash mismatch: {p}')
                pending.append(p)
        pending.extend(Path(a['path']) for a in entry.get('assets',[]))
        pending.append(args.work/'sequences'/entry['sequence']/'poses.npy')
    files={}
    while pending:
        path=pending.pop();relative=safe(path)
        if str(relative) in files:continue
        if not path.is_file():raise ValueError(f'missing frozen input {path}')
        files[str(relative)]=digest(path)
        if path.suffix=='.urdf':pending.extend(path.parent/e.attrib['filename'] for e in ET.parse(path).findall('.//mesh'))
        if path.suffix in ('.obj','.mtl'):
            for line in path.read_text().splitlines():
                if line.startswith('mtllib '):pending.extend(path.parent/n for n in shlex.split(line)[1:])
                elif line.startswith(('map_Kd ','map_Ka ','map_Ks ','map_d ','map_Bump ','bump ')):pending.append(path.parent/shlex.split(line)[-1])
    identity={'files':files,'input_manifest_sha256':digest(args.manifest)}
    if args.archive.exists():
        if json.loads((args.archive/'manifest.json').read_text())!=identity:raise ValueError('frozen inputs changed')
        for rel,h in files.items():
            if digest(args.archive/'files'/rel)!=h:raise ValueError('frozen archive corrupt')
        return
    args.archive.mkdir(parents=True)
    for rel,h in files.items():
        out=args.archive/'files'/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,out)
        if digest(out)!=h:raise ValueError('input changed during copy')
    (args.archive/'manifest.json').write_text(json.dumps(identity,indent=2)+'\n')
    print(f'frozen input closure: {len(files)} files')

def restore(args):
    manifest=json.loads((args.archive/'manifest.json').read_text())
    # Verify the complete archive before replacing any host-local regenerated input.
    for rel,h in manifest['files'].items():
        safe(ROOT/rel)
        if digest(args.archive/'files'/rel)!=h:raise ValueError(f'corrupt frozen input {rel}')
    for rel,h in manifest['files'].items():
        out=ROOT/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(args.archive/'files'/rel,out)
        if digest(out)!=h:raise ValueError(f'restored hash mismatch: {rel}')
    print(f'restored and rehashed input closure: {len(manifest["files"])} files')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['capture','restore']);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--manifest',type=Path);p.add_argument('--work',type=Path);args=p.parse_args()
    if args.action=='capture' and (not args.manifest or not args.work):p.error('capture requires manifest/work')
    try:{'capture':capture,'restore':restore}[args.action](args)
    except (OSError,ValueError,KeyError) as e:p.exit(1,f'error: {e}\n')
if __name__=='__main__':main()
