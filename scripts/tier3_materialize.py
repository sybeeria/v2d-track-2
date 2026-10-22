"""Copy a generated motion tree into a portable FlashCHORD asset closure.

The kit resolves /assets/... locators against the parquet's own assets ancestor.
Only generated mesh/URDF path columns are rewritten; motion arrays stay exact.
"""
import argparse
from pathlib import Path
import shutil
import pyarrow as pa
import pyarrow.parquet as pq


def materialize(motion_root: Path, out: Path):
    motion_root=motion_root.resolve()
    out=out.resolve()
    if out.name != 'assets':
        raise ValueError('output must be named assets for the kit resolver')
    if out.exists():
        raise FileExistsError(out)
    sources=sorted(motion_root.glob('whole_body/soma/sequence_id=*/robot_name=g1/data.parquet'))
    if not sources:
        raise ValueError('no generated G1 references')
    # Validate the asset closure before copying or rewriting anything.
    for path in sources:
        row=pq.ParquetFile(path).read().to_pylist()[0]
        for key in ('object_mesh_paths','object_urdf_paths'):
            for stored in row[key]:
                asset=Path(stored).resolve()
                asset.relative_to(motion_root)
                if not asset.is_file():
                    raise FileNotFoundError(asset)
    shutil.copytree(motion_root,out/'human_motion_data')
    outputs=[]
    for source in sources:
        path=out/'human_motion_data'/source.relative_to(motion_root)
        table=pq.ParquetFile(path).read()
        for key in ('object_mesh_paths','object_urdf_paths'):
            column=table[key].to_pylist()
            locators=[['/assets/human_motion_data/'+str(Path(v).resolve().relative_to(motion_root)) for v in values]
                      for values in column]
            table=table.set_column(table.schema.get_field_index(key),table.schema.field(key),pa.array(locators,type=table[key].type))
        pq.write_table(table,path)
        outputs.append(path)
    return outputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    for path in materialize(a.motion_root,a.out):print(path)


if __name__=='__main__':main()
