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
        # The box disk is a fixture boundary: pre.sh's real admission code reads BOX_FREE_BYTES, not this Mac's disk.
        python=box/'.venv/bin/python';python.parent.mkdir(parents=True)
        python.write_text(f'#!{sys.executable}\nimport os,shutil,sys,types\n'
            "if sys.argv[1:]==['-']:\n"
            " free=int(os.environ.get('BOX_FREE_BYTES',150*10**9));shutil.disk_usage=lambda path:types.SimpleNamespace(total=200*10**9,used=200*10**9-free,free=free)\n"
            " exec(sys.stdin.read())\n"
            "else:os.execv(sys.executable,[sys.executable,*sys.argv[1:]])\n");python.chmod(0o755)
        (box/READY).mkdir(parents=True);shutil.copyfile(self.ctrl/READY/'generate_jobs.py',box/READY/'generate_jobs.py')
        for tier in (1,2):
            work=box/f'data/work/tier{tier}';(work/'sequences/s0').mkdir(parents=True);(work/'sequences/s0/poses.npy').write_text(f'{label}{tier}')
            for key in ('training_reference','clean_scoring_reference','support'):(work/f'{key}.bin').write_text(f'{label}-tier{tier}-{key}')
        return box
    def boot(self,box,run,setup='',**env):
        return subprocess.run(['bash','-c',f'{setup}source "{run}/pre.sh"'],cwd=box,capture_output=True,text=True,env={**os.environ,**env})

    def test_partial_frozen_pull_fails_before_staging(self):
        archive=self.w/'pulled/canary/frozen-tier2';(archive/'files/data').mkdir(parents=True)
        (archive/'files/data/half.bin').write_text('copy in progress')  # manifest.json is written last by capture
        r,run=self.make_run('--tier','2','--name','t2a','--frozen-from',str(archive))
        self.assertNotEqual(r.returncode,0,r.stdout);self.assertIn('frozen input archive',r.stderr);self.assertFalse(run.exists())

    def test_production_k_is_refused_before_staging(self):
        for tier in (1,2,3):
            for k in (0,1,5):
                with self.subTest(tier=tier,k=k):
                    r,run=self.make_run('--tier',str(tier),'--name',f't{tier}-k{k}','--k',str(k))
                    self.assertNotEqual(r.returncode,0,r.stdout)
                    self.assertIn('production requires the frozen k=3',r.stderr)
                    self.assertFalse(run.exists())
        r,run=self.make_run('--tier','2','--name','k3','--k','3')
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('--k 3 --out runs/k3/qualification\n',(run/'post.sh').read_text())

    def test_canary_replacement_reuses_its_frozen_tier2_inputs(self):
        r,run=self.make_run('--tier','1','--name','canary','--freeze-other-tier');self.assertEqual(r.returncode,0,r.stderr)
        first=self.host('a');r=self.boot(first,run);self.assertEqual(r.returncode,0,r.stderr)
        pulled=self.w/'pulled';shutil.copytree(first/'runs/canary',pulled/'canary')
        second=self.host('b');shutil.copytree(pulled/'canary',second/'runs/canary')  # launch.sh restores the pulled tree
        r=self.boot(second,run);self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual((second/'data/work/tier2/training_reference.bin').read_text(),'a-tier2-training_reference')
        self.assertEqual((second/'data/work/tier1/training_reference.bin').read_text(),'a-tier1-training_reference')
        r,t2=self.make_run('--tier','2','--name','t2a','--frozen-from',str(pulled/'canary/frozen-tier2'));self.assertEqual(r.returncode,0,r.stderr)

    def test_box_below_the_disk_admission_is_refused(self):
        r,run=self.make_run('--tier','1','--name','canary');self.assertEqual(r.returncode,0,r.stderr)
        r=self.boot(self.host('a'),run,BOX_FREE_BYTES=str(114_999_999_999))
        self.assertNotEqual(r.returncode,0);self.assertIn('production disk admission failed: 114999999999 bytes free',r.stderr)

    def test_capture_interrupted_mid_copy_is_retried_on_the_same_host(self):
        r,run=self.make_run('--tier','1','--name','canary');self.assertEqual(r.returncode,0,r.stderr)
        box=self.host('a');big=box/'data/work/tier1/support.bin';big.write_bytes(os.urandom(1<<20))
        # A full disk (file size limit) stops the capture partway through the closure.
        r=self.boot(box,run,setup='ulimit -f 256; ');self.assertNotEqual(r.returncode,0);self.assertIn('File too large',r.stderr)
        self.assertFalse((box/'runs/canary/frozen-inputs').exists(),'a partial archive is visible to the watchdog pull')
        r=self.boot(box,run);self.assertEqual(r.returncode,0,r.stderr)
        archive=box/'runs/canary/frozen-inputs';files=json.loads((archive/'manifest.json').read_text())['files']
        self.assertEqual(sorted(files),[f'data/work/tier1/{n}' for n in ('clean_scoring_reference.bin','sequences/s0/poses.npy','support.bin','training_reference.bin')])
        self.assertEqual((archive/'files/data/work/tier1/support.bin').read_bytes(),big.read_bytes())

if __name__=='__main__':unittest.main()
