"""Attempt custody through prepare/run/qualify; GPU execution stays behind a fake executable."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests import test_pipeline_b8 as fixtures


class AttemptCustody(unittest.TestCase):
    setUp = fixtures.Pipeline.setUp
    command = fixtures.Pipeline.command
    roster = fixtures.Pipeline.roster
    manifest = fixtures.Pipeline.manifest
    cohort = fixtures.Pipeline.cohort
    evaluator = fixtures.Pipeline.evaluator

    def refused(self, cohort, error):
        result = self.command('qualify', '--out', cohort)
        self.assertEqual(result.returncode, 1, result.stdout+result.stderr)
        self.assertIn(error, result.stderr)
        self.assertEqual({name: (cohort/name).exists() for name in
            ('selected.json', 'selection.json', 'veto.json', 'export-binding.json')},
            {'selected.json': False, 'selection.json': False, 'veto.json': False, 'export-binding.json': False})

    def test_attempt_is_visible_at_spawn_and_matches_the_seal(self):
        cohort = self.cohort([0], evaluated=False)
        env = self.evaluator()
        fake = Path(env['PYTHON'])
        fake.write_text(fake.read_text().replace("hydra = Path(args['hydra.run.dir'])/'.hydra'",
            "result = Path(args['evaluation.metrics_output']).parent\n"
            "attempt = result/'evaluation.attempt.json'\n"
            "(result/'observed-at-spawn.json').write_bytes(attempt.read_bytes() if attempt.exists() else b'null')\n"
            "hydra = Path(args['hydra.run.dir'])/'.hydra'"))
        run = self.command('run', '--out', cohort, env=env)
        self.assertEqual(run.returncode, 0, run.stderr)
        plan = json.loads((cohort/'plan.json').read_text())
        arm = json.loads(Path(plan['manifest']).read_text())['sequences'][0]['arms']['raw_repaired']
        checkpoint = cohort.parent/'source/t2-ep00/T2C-contact/raw_repaired/train/seq0'/f'policy_{fixtures.SHORT}.safetensors'
        expected = {'checkpoint_sha256': fixtures.sha(checkpoint), 'evaluator': plan['evaluator'],
            'input': 'raw_repaired', 'training_reference': arm['training_reference'],
            'scoring_reference': arm['clean_scoring_reference'], 'support': arm['support']}
        seals = sorted(cohort.rglob('evaluation.done.json'))
        self.assertEqual(len(seals), 3)
        for seal in seals:
            self.assertEqual(json.loads(seal.with_name('observed-at-spawn.json').read_text()), expected)
            attempt = seal.with_name('evaluation.attempt.json')
            self.assertFalse(attempt.is_symlink())
            self.assertEqual(attempt.stat().st_nlink, 1)
            self.assertEqual(json.loads(attempt.read_text()), expected)
            self.assertEqual({k: v for k, v in json.loads(seal.read_text()).items()
                              if k not in ('trajectory_sha256', 'metrics_sha256')}, expected)
        result = self.command('qualify', '--out', cohort)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((cohort/'selected.json').read_text()), {'0': str(checkpoint.resolve())})

    def test_runtime_change_after_prepare_is_fatal_even_when_every_member_fails(self):
        real_git = shutil.which('git')
        bindir = self.work/'bin'; bindir.mkdir()
        git = bindir/'git'
        git.write_text(f'#!{sys.executable}\nimport os,sys\n'
            "if sys.argv[3:6] == ['diff','HEAD','--binary']:\n sys.stdout.write('changed runtime after preparation\\n')\n"
            f'else: os.execv({real_git!r}, [{real_git!r}, *sys.argv[1:]])\n')
        git.chmod(0o755)
        for tier in (2, 3):
            with self.subTest(tier=tier):
                cohort = self.cohort([0], tier=tier, evaluated=False)
                env = {**self.evaluator(FAKE_FAIL_ALL='1'), 'PATH': str(bindir)+os.pathsep+os.environ['PATH']}
                identity = subprocess.run([sys.executable, '-c',
                    f'import sys; sys.path.insert(0,{str(fixtures.ROOT/"scripts")!r}); '
                    'from pilot_queue import evaluator_identity,FLASH; print(evaluator_identity(FLASH))'],
                    env={**os.environ, **env}, capture_output=True, text=True, check=True)
                self.assertNotEqual(identity.stdout.strip(), json.loads((cohort/'plan.json').read_text())['evaluator'])
                run = self.command('run', '--out', cohort, env=env)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertEqual((cohort/'RUN_COMPLETE').read_text(), 'test_day exit 1\n')
                self.assertEqual(len(list(cohort.rglob('evaluation.done.json'))), 0)
                self.assertEqual(len([p for p in cohort.rglob('config.yaml') if p.stat().st_size]), 3)
                self.refused(cohort, 'checkpoint/runtime/input attempt provenance mismatch')

    def failed_fixture(self, tier, output, shifts=None):
        cohort = self.cohort([0], tier=tier, shifts=shifts)
        seals = sorted(cohort.rglob('evaluation.done.json'))
        for seal in seals:
            done = json.loads(seal.read_text())
            fixtures.put(seal.with_name('evaluation.attempt.json'),
                         {k: v for k, v in done.items() if k not in ('trajectory_sha256', 'metrics_sha256')})
        seal = seals[-1]
        if output == 'failed':
            seal.unlink()
            seal.with_name('trajectory.parquet').unlink()
            startup = seal.parent/'hydra/.hydra/config.yaml'
            startup.parent.mkdir(parents=True)
            startup.write_text('evaluation:\n  world_count: 1\n')
            member = cohort/'cohort'/f't{tier}-ep00-r03'
            fixtures.put(member/'summary.json', {'candidates': []})
        else:
            metrics = seal.with_name('metrics.json')
            record = json.loads(metrics.read_text()); record['non_finite_worlds'] = [0]
            fixtures.put(metrics, record)
            done = json.loads(seal.read_text()); done['metrics_sha256'] = fixtures.sha(metrics)
            fixtures.put(seal, done)
        return cohort, seal.with_name('evaluation.attempt.json').relative_to(cohort)

    def test_failed_and_invalid_members_require_regular_matching_attempts(self):
        for tier in (2, 3):
            for output in ('failed', 'invalid'):
                original, relative = self.failed_fixture(tier, output)
                for mutation in ('missing', 'checkpoint', 'evaluator', 'input', 'training_reference',
                                 'scoring_reference', 'support', 'symlink', 'hardlink', 'directory', 'malformed'):
                    with self.subTest(tier=tier, output=output, mutation=mutation):
                        cohort = self.work/f'{original.parent.name}-{output}-{mutation}'
                        shutil.copytree(original, cohort, symlinks=True)
                        attempt = cohort/relative
                        if mutation in ('missing', 'symlink', 'hardlink', 'directory'):
                            data = attempt.read_bytes(); attempt.unlink()
                            if mutation in ('symlink', 'hardlink'):
                                target = self.work/f'{cohort.name}-record'; target.write_bytes(data)
                                if mutation == 'symlink': attempt.symlink_to(target)
                                else: os.link(target, attempt)
                            elif mutation == 'directory': attempt.mkdir()
                            error = ('evaluation attempt missing; cohort incomplete' if mutation == 'missing'
                                     else 'separate regular cohort artifacts required')
                        elif mutation == 'malformed':
                            attempt.write_text('{')
                            error = 'Expecting property name'
                        else:
                            record = json.loads(attempt.read_text()); record[mutation] = 'changed'
                            fixtures.put(attempt, record)
                            error = 'checkpoint/runtime/input attempt provenance mismatch'
                        self.refused(cohort, error)

    def test_legacy_failed_and_invalid_members_without_any_attempts_stay_incomplete(self):
        for output in ('failed', 'invalid'):
            with self.subTest(output=output):
                cohort, _ = self.failed_fixture(2, output)
                for attempt in cohort.rglob('evaluation.attempt.json'): attempt.unlink()
                self.assertEqual(len(list(cohort.rglob('evaluation.attempt.json'))), 0)
                self.refused(cohort, 'evaluation attempt missing; cohort incomplete')

    def test_legacy_failure_stands_only_when_a_sealed_member_already_decides_catastrophe(self):
        for output in ('failed', 'invalid'):
            for shifts, published in (([0., .2, 0.], True), ([0., 0., 0.], False)):
                with self.subTest(output=output, shifts=shifts):
                    cohort, _ = self.failed_fixture(2, output, {fixtures.SHORT: shifts})
                    for attempt in cohort.rglob('evaluation.attempt.json'): attempt.unlink()
                    if not published:
                        self.refused(cohort, 'evaluation attempt missing; cohort incomplete')
                        continue
                    result = self.command('qualify', '--out', cohort)
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                    candidate = json.loads((cohort/'veto.json').read_text())['candidates'][0]
                    self.assertEqual((candidate['status'], [m['catastrophe'] for m in candidate['members']]),
                                     ('catastrophe', [False, True, True]))
                    self.assertEqual(candidate['members'][2], {'repeat': 3, 'failed': True, 'catastrophe': True, 'orientation': False})
