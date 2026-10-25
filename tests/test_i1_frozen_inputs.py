"""Public I1 CLI must score the archived screen bound to its training recipes, never a local rebuild."""
import hashlib
import json
from pathlib import Path
import shutil
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
from tests import test_runbook as runbook


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def shift(path, offset):
    table=pq.ParquetFile(path).read();key='object_body_position'
    values=[[[[v[0]+offset,v[1],v[2]] for v in frame] for frame in row] for row in table[key].to_pylist()]
    table=table.set_column(table.schema.get_field_index(key),key,pa.array(values,type=table.schema.field(key).type))
    pq.write_table(table,path)


def seal_screen(fixture, offset=0):
    """Production-shaped portable archive and per-member recipe bindings, wholly inside the test's tmp root."""
    archive=fixture.pulled/'frozen-screen';entries=[]
    local=json.loads((fixture.root/'data/screen/tier2/manifest.json').read_text())
    for entry in local['sequences']:
        seq=entry['sequence'];original=Path(entry['arms']['raw_repaired']['training_reference']['path'])
        arms={}
        for label in ('clean','raw_repaired'):
            base=archive/seq/label/'motion/whole_body/soma'/f'sequence_id={seq}'
            ref=base/'robot_name=g1/data.parquet';ref.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,ref)
            if offset:shift(ref,offset)
            mesh=base/'object/textured_mesh.obj';mesh.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(original.parent.parent/'object/textured_mesh.obj',mesh)
            support=archive/seq/label/'motion/whole_body/reconstructed_stage'/f'{seq}_support.usda'
            support.parent.mkdir(parents=True,exist_ok=True);support.write_text('#usda 1.0\ndef Xform "support_surfaces" {}\n')
            def record(p):return {'path':'/workspace/v2d-track-2/data/screen/tier2/'+p.relative_to(archive).as_posix(),'sha256':sha(p)}
            arms[label]={'training_reference':record(ref),'mesh':record(mesh),'support':{**record(support),'enabled':False}}
        for label in arms:arms[label]['clean_scoring_reference']=arms['clean']['training_reference']
        entries.append({'sequence':seq,'arms':arms})
    manifest={'schema_version':1,'sequences':entries}
    (archive/'manifest.json').write_text(json.dumps(manifest))
    lookup={e['sequence']:e['arms']['raw_repaired'] for e in entries}
    for path in fixture.pulled.glob('repeats/i1/*/*/raw_repaired/evaluation/*/49999872/trajectory.parquet'):
        seq=path.parents[1].name;arm=lookup[seq]
        bound={key:{k:arm[source][k] for k in ('path','sha256')} for key,source in
               [('training','training_reference'),('scoring','clean_scoring_reference'),('support','support')]}
        bound['support']['enabled']=arm['support']['enabled']
        (path.parents[4]/'recipe.json').write_text(json.dumps({'schema_version':2,'references':{f'raw_repaired/{seq}':bound}}))
    return archive


class I1FrozenInputs(unittest.TestCase):
    def setUp(self):
        self.f=runbook.I1Decisions();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.f.fixture(44,3);self.f.fixture(45,3)
        self.archive=seal_screen(self.f)

    def decision(self):
        result=self.f.decide(44,45)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        return json.loads((self.f.root/'decision.json').read_text())['verdict']

    def test_complete_archived_cohort_has_expected_go(self):
        self.assertEqual(self.decision(),'GO')

    def test_local_rebuild_cannot_change_a_bound_go(self):
        self.assertEqual(self.decision(),'GO')
        for ref in (self.f.root/'fixtures').glob('*/robot_name=g1/data.parquet'):shift(ref,.2)
        self.assertEqual(self.decision(),'GO')

    def test_local_clean_input_cannot_authorize_a_failing_archived_screen(self):
        seal_screen(self.f,.2)
        self.assertEqual(self.decision(),'NO-GO')

    def test_missing_archived_screen_is_refused_without_a_decision(self):
        shutil.rmtree(self.archive)
        result=self.f.decide(44,45)
        self.assertNotEqual(result.returncode,0,result.stdout)
        self.assertFalse((self.f.root/'decision.json').exists())

    def test_changed_archive_bytes_are_refused_without_a_decision(self):
        ref=next(self.archive.glob('*/raw_repaired/motion/whole_body/soma/sequence_id=*/robot_name=g1/data.parquet'))
        shift(ref,.01)
        result=self.f.decide(44,45)
        self.assertNotEqual(result.returncode,0,result.stdout)
        self.assertFalse((self.f.root/'decision.json').exists())

    def test_tier1_locator_is_refused_before_any_reference_read(self):
        manifest=self.archive/'manifest.json';content=json.loads(manifest.read_text())
        content['sequences'][0]['arms']['raw_repaired']['training_reference']['path']='/workspace/v2d-track-2/data/challenge/track_2/tier_1_multiview_caption/data/chunk-000/episode_000000.parquet'
        manifest.write_text(json.dumps(content))
        result=self.f.decide(44,45)
        self.assertNotEqual(result.returncode,0,result.stdout)
        self.assertIn('outside the frozen external screen',result.stderr)
        self.assertFalse((self.f.root/'decision.json').exists())

    def test_member_trained_on_another_reference_is_refused(self):
        recipe=next(self.f.pulled.glob('repeats/i1/*/*/recipe.json'))
        content=json.loads(recipe.read_text());next(iter(content['references'].values()))['training']['sha256']='0'*64
        recipe.write_text(json.dumps(content))
        result=self.f.decide(44,45)
        self.assertNotEqual(result.returncode,0,result.stdout)
        self.assertFalse((self.f.root/'decision.json').exists())

if __name__=='__main__':unittest.main(verbosity=2)
