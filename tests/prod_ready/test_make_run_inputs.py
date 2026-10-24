"""make_run.py frozen-input staging: partial archives fail on the Mac; a canary replacement reuses its Tier 2 closure."""
import hashlib,json,os,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path

BASE=Path(__file__).resolve().parents[2];READY=Path('data/brainstorm6/PROD-READY')
# Stand-in for generate_jobs.py on a box: one manifest per tier, built from whatever references the host holds.
GENERATOR='''import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def record(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
for tier,label in ((1,'clean'),(2,'raw_repaired')):
    work=ROOT/f'data/work/tier{tier}'
    arm={key:record(work/f'{key}.bin') for key in ('training_reference','clean_scoring_reference','support')}
    out=ROOT/f'data/brainstorm6/PROD-READY/generated/tier{tier}/manifest.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'tier':tier,'sequences':[{'sequence':'s0','arms':{label:arm},'assets':[]}]}))
'''

class MakeRunInputs(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='make-run-',dir='/tmp');self.addCleanup(self.temp.cleanup)
        self.w=Path(self.temp.name).resolve();self.ctrl=self.w/'controller'
        (self.ctrl/READY/'generated').mkdir(parents=True);shutil.copyfile(BASE/READY/'make_run.py',self.ctrl/READY/'make_run.py')
        (self.ctrl/READY/'generate_jobs.py').write_text(GENERATOR);shutil.copytree(BASE/READY/'generated/arms',self.ctrl/READY/'generated/arms')
        old=self.ctrl/'data/brainstorm5/PROD-PLAN/generate_jobs.py';old.parent.mkdir(parents=True);old.write_text('')
        self.jobs=self.w/'jobs.txt';self.jobs.write_text('t1-ep19|x|F6-rebalance-tail|clean|s0|330012672 349999104\n')
    def make_run(self,*extra):
        r=subprocess.run([sys.executable,'-B',str(self.ctrl/READY/'make_run.py'),'--jobs',str(self.jobs),*extra],capture_output=True,text=True)
        return r,self.ctrl/READY/'runs'/extra[extra.index('--name')+1]
    def host(self,label):
        """A fresh box at the one fixed project path, with its own bootstrap-regenerated references."""
        box=self.w/'workspace/v2d-track-2'
        if box.exists():box.rename(self.w/f'retired-{label}')
        (box/'scripts').mkdir(parents=True);shutil.copyfile(BASE/'scripts/freeze_inputs.py',box/'scripts/freeze_inputs.py')
        (box/'.venv/bin').mkdir(parents=True);(box/'.venv/bin/python').symlink_to(sys.executable)
        (box/READY).mkdir(parents=True);shutil.copyfile(self.ctrl/READY/'generate_jobs.py',box/READY/'generate_jobs.py')
        for tier in (1,2):
            work=box/f'data/work/tier{tier}';(work/'sequences/s0').mkdir(parents=True);(work/'sequences/s0/poses.npy').write_text(f'{label}{tier}')
            for key in ('training_reference','clean_scoring_reference','support'):(work/f'{key}.bin').write_text(f'{label}-tier{tier}-{key}')
        return box
    def boot(self,box,run):
        # box.sh sources pre.sh; its real 115 GB disk admission also runs, so the Mac needs that much free.
        return subprocess.run(['bash','-c',f'source "{run}/pre.sh"'],cwd=box,capture_output=True,text=True)

    def test_partial_frozen_pull_fails_before_staging(self):
        archive=self.w/'pulled/canary/frozen-tier2';(archive/'files/data').mkdir(parents=True)
        (archive/'files/data/half.bin').write_text('copy in progress')  # manifest.json is written last by capture
        r,run=self.make_run('--tier','2','--name','t2a','--frozen-from',str(archive))
        self.assertNotEqual(r.returncode,0,r.stdout);self.assertIn('frozen input archive',r.stderr);self.assertFalse(run.exists())

    def test_canary_replacement_reuses_its_frozen_tier2_inputs(self):
        r,run=self.make_run('--tier','1','--name','canary','--freeze-other-tier');self.assertEqual(r.returncode,0,r.stderr)
        first=self.host('a');r=self.boot(first,run);self.assertEqual(r.returncode,0,r.stderr)
        pulled=self.w/'pulled';shutil.copytree(first/'runs/canary',pulled/'canary')
        second=self.host('b');shutil.copytree(pulled/'canary',second/'runs/canary')  # launch.sh restores the pulled tree
        r=self.boot(second,run);self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual((second/'data/work/tier2/training_reference.bin').read_text(),'a-tier2-training_reference')
        self.assertEqual((second/'data/work/tier1/training_reference.bin').read_text(),'a-tier1-training_reference')
        r,t2=self.make_run('--tier','2','--name','t2a','--frozen-from',str(pulled/'canary/frozen-tier2'));self.assertEqual(r.returncode,0,r.stderr)

if __name__=='__main__':unittest.main()
