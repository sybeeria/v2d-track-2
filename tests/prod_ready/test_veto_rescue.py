"""merge --rescue: a vetoed Tier 2/3 episode moves only to a newly trained checkpoint with a strictly better veto class."""
import hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path

BASE=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))

class VetoRescue(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='/tmp');self.addCleanup(self.temp.cleanup);self.w=Path(self.temp.name)
        self.manifest=self.w/'manifest.json';self.manifest.write_text('{"tier": 2}')
        self.tier_root=self.w/'tier2'
        put(self.tier_root/'meta/info.json',{'total_episodes':2})
        (self.tier_root/'meta/episodes_metadata.jsonl').write_text(''.join(json.dumps({'episode_index':e,'sequence_id':f'seq{e}'})+'\n' for e in range(2)))
    def cohort(self,name,statuses,weights={}):
        root=self.w/name;mapping={};candidates=[]
        for episode,status in statuses.items():
            policy=root/f'policies/ep{episode}.safetensors';policy.parent.mkdir(parents=True,exist_ok=True);policy.write_text(weights.get(episode,f'{name}-{episode}'))
            mapping[str(episode)]=str(policy);candidates.append({'episode':episode,'status':status,'checkpoint_sha256':sha(policy)})
        put(root/'plan.json',{'tier':2,'manifest':str(self.manifest),'manifest_sha256':sha(self.manifest),'evaluator':{'id':'e'},'repeats':3,'ep18':False})
        put(root/'selected.json',mapping);put(root/'veto.json',{'candidates':candidates})
        put(root/'export-binding.json',{'selected_sha256':sha(root/'selected.json'),'veto_sha256':sha(root/'veto.json'),
                                        'checkpoints':{p:sha(p) for p in mapping.values()}});return root
    def merge(self,cohorts,rescues,out):
        command=[sys.executable,'-B',str(BASE/'scripts/production_repeats.py'),'merge','--tier-root',str(self.tier_root),'--out',str(self.w/out)]
        for c in cohorts:command+=['--cohort',str(c)]
        for r in rescues:command+=['--rescue',str(r)]
        return subprocess.run(command,capture_output=True,text=True)
    def read(self,out,name):return json.loads((self.w/out/name).read_text())

    def test_earliest_strictly_better_seed_wins(self):
        original=self.cohort('original',{0:'pass',1:'catastrophe'})
        seed43=self.cohort('seed43',{1:'catastrophe'});seed44=self.cohort('seed44',{1:'orientation'})
        seed45=self.cohort('seed45',{1:'pass'});seed46=self.cohort('seed46',{1:'pass'})
        r=self.merge([original],[seed43,seed44,seed45,seed46],'release');self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(self.read('release','selected.json'),{'0':str(original/'policies/ep0.safetensors'),'1':str(seed45/'policies/ep1.safetensors')})
        self.assertEqual(self.read('release','veto.json')['episodes'],{'0':{'original':'pass','final':'pass'},
                         '1':{'original':'catastrophe','final':'pass','cohort':str(seed45.resolve())}})
        r=self.merge([original],[seed43],'retained');self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(self.read('retained','selected.json')['1'],str(original/'policies/ep1.safetensors'))
        self.assertEqual(self.read('retained','veto.json')['episodes']['1'],{'original':'catastrophe','final':'catastrophe'})

    def test_fallback_cohort_never_replaces_a_passing_episode_or_reuses_a_checkpoint(self):
        primary=self.cohort('primary',{0:'pass',1:'orientation'})
        fallback=self.cohort('fallback',{0:'pass',1:'pass'})  # e.g. T2C-contact behind T2C-E4 (idea I1)
        r=self.merge([primary],[fallback],'release');self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(self.read('release','selected.json'),{'0':str(primary/'policies/ep0.safetensors'),'1':str(fallback/'policies/ep1.safetensors')})
        original=self.cohort('original',{0:'pass',1:'catastrophe'})
        same=self.cohort('same',{1:'pass'},{1:'original-1'})  # identical bytes: a re-evaluation, not a reseed
        r=self.merge([original],[same],'bad2');self.assertNotEqual(r.returncode,0);self.assertIn('newly trained checkpoint',r.stderr)
        self.assertFalse((self.w/'bad2').exists())

    def test_a_reseed_checkpoint_gets_one_cohort_whichever_cohort_repeats_it(self):
        original=self.cohort('original',{0:'pass',1:'orientation'})
        seed43=self.cohort('seed43',{1:'catastrophe'});again=self.cohort('again',{1:'pass'},{1:'seed43-1'})
        r=self.merge([original],[seed43,again],'release');self.assertNotEqual(r.returncode,0);self.assertIn('newly trained checkpoint',r.stderr)
        self.assertFalse((self.w/'release').exists())

    def test_veto_classes_are_bound_like_the_selection(self):
        original=self.cohort('original',{0:'pass',1:'catastrophe'});seed43=self.cohort('seed43',{1:'orientation'})
        put(seed43/'veto.json',{'candidates':[{'episode':1,'status':'pass','checkpoint_sha256':sha(seed43/'policies/ep1.safetensors')}]})
        r=self.merge([original],[seed43],'release');self.assertNotEqual(r.returncode,0);self.assertIn('veto.json',r.stderr)
        self.assertFalse((self.w/'release').exists())

if __name__=='__main__':unittest.main()
