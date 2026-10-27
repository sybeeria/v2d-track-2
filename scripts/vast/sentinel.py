#!/usr/bin/env python3
"""Mac-side billing sentinel; launchd runs it every 5 min (scripts/vast/install_sentinel.sh). Sends Telegram alerts.

A watchdog trusts its own view: on Oct 27 `vastai destroy` failed with a 401 (2FA expired) yet exited 0, so t1h5's
watchdog wrote STOPPED and exited while the instance kept billing (TRIAGE BOX-10). This checks Vast's own instance
list against the watchdogs that still hold their instance lock, and scans watchdog logs for costly failures.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG = Path(os.environ.get('V2D_TELEGRAM_ENV', Path.home() / '.config/v2d/telegram.env'))
ORPHAN_AFTER = 900  # start_run.sh arms the watchdog a few minutes after the instance starts billing
API_FAIL_AFTER = 600
UNREACHABLE_AFTER = 1800
REPEAT_AFTER = 3600
LOG_EVENTS = [
    ('destroy-failed', re.compile(r'destroy failed \((.*)\)'), 'destroy FAILED ({}); the instance is still billing'),
    ('backup-failed', re.compile(r'backup failed \((.*)\)'), 'backup failed ({}); the latest work may not be saved'),
    ('vast-error', re.compile(r'Failed with error (\d+)'), 'Vast API error {} in the watchdog log'),
    ('early-stop', re.compile(r'destroy requested for instance \d+ \((?!finished\))(.*)\)'), 'stopped early: {}'),
]
STATE_LINE = re.compile(r'^(\S+Z) state=(\S+) spend=\$([0-9.]+)')


def check(state, now, instances, bound, logs, exposed=0):
    """instances: Vast instance dicts, or None when the API call failed. bound: instance ids whose watchdog guards
    them. logs: {run name: new watchdog.log lines}. exposed: runs that may still bill (lock held, or no STOPPED
    receipt). Updates state; returns the alert texts due now."""
    due = []
    if instances is None:
        since = state.setdefault('api_fail_since', now)
        seen = len(state.get('instances', [])) or len(bound) or exposed
        if now - since >= API_FAIL_AFTER and seen:
            due.append(('api', f'Vast API failing for {int((now - since) / 60)} min (2FA login expired?): watchdogs '
                               f'cannot destroy; {seen} instance(s) or run(s) may be billing. Run: vastai tfa login '
                               f'--method-type totp --code <6 digits>'))
    else:
        state.pop('api_fail_since', None)
        state['instances'] = [i['id'] for i in instances]
        orphans = {k: v for k, v in state.get('orphan_since', {}).items() if int(k) in state['instances']}
        for i in instances:
            key = str(i['id'])
            if i['id'] in bound:
                orphans.pop(key, None)
                continue
            if now - orphans.setdefault(key, now) >= ORPHAN_AFTER:
                due.append((f'orphan-{key}', f"instance {key} ({i.get('actual_status')}, "
                                             f"${i.get('dph_total') or 0:.2f}/h) has no live watchdog and bills "
                                             f'unattended. Destroy: vastai destroy instance {key} -y'))
        state['orphan_since'] = orphans
    unreachable = state.setdefault('unreachable_since', {})
    for run, lines in logs.items():
        for line in lines:
            for kind, pattern, text in LOG_EVENTS:
                match = pattern.search(line)
                if match:
                    due.append((f'{run}-{kind}', f'{run}: ' + text.format(*match.groups())))
            match = STATE_LINE.match(line)
            if not match:
                continue
            stamp = datetime.fromisoformat(match[1].replace('Z', '+00:00')).timestamp()
            if match[2] != 'unreachable':
                unreachable.pop(run, None)
            elif stamp - unreachable.setdefault(run, stamp) >= UNREACHABLE_AFTER:
                minutes = int((stamp - unreachable[run]) / 60)
                due.append((f'{run}-unreachable', f'{run}: box unreachable for {minutes} min (spend ~${float(match[3]):.2f}); '
                                                  'the watchdog cannot stop it, only the cap or the box deadman can (BOX-9)'))
    sent, alerts = state.setdefault('sent', {}), []
    for key, text in dict(due).items():
        if now - sent.get(key, 0) >= REPEAT_AFTER:
            sent[key] = now
            alerts.append((key, text))
    return alerts


def send(text):
    env = dict((k.strip(), v.strip()) for k, v in (line.split('=', 1) for line in CONFIG.read_text().splitlines()
                                                   if '=' in line and not line.startswith('#')))
    data = urllib.parse.urlencode({'chat_id': env['TELEGRAM_CHAT_ID'], 'text': text}).encode()
    with urllib.request.urlopen(f"https://api.telegram.org/bot{env['TELEGRAM_BOT_TOKEN']}/sendMessage", data, timeout=20) as r:
        if not json.load(r).get('ok'):
            raise RuntimeError('telegram refused the message')


def deliver(key, text, dry_run):
    """True once the alert is out. Errors are printed by type only: their text can contain the token-bearing URL."""
    if dry_run:
        print(text)
        return True
    try:
        send(text)
    except Exception as error:
        print(datetime.now().isoformat(timespec='seconds'), 'send failed', key, type(error).__name__,
              getattr(error, 'code', ''))
        return False
    print(datetime.now().isoformat(timespec='seconds'), 'sent', key)
    return True


def vast_instances():
    try:
        done = subprocess.run(['vastai', 'show', 'instances', '--raw'], capture_output=True, text=True, timeout=60)
        if done.returncode:
            return None
        instances = json.loads(done.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    return instances if isinstance(instances, list) else None


def held(paths):
    """Paths some process has open; a live watchdog keeps its instance lock open until it exits."""
    if not paths:
        return set()
    out = subprocess.run(['lsof', '-Fn', '--', *map(str, paths)], capture_output=True, text=True).stdout
    return {Path(line[1:]) for line in out.splitlines() if line.startswith('n')}


def armed(lock, run):
    """A watchdog rewrites its instance lock when it takes it and logs 'watchdog start' only once it can guard; one
    that cannot read the instance's metadata holds the lock unarmed (BOX-3)."""
    log = run / 'watchdog.log'
    if not log.is_file():
        return False
    for line in reversed(log.read_text(errors='replace').splitlines()):
        if f' watchdog start id={lock.stem} ' in line:
            return datetime.fromisoformat(line[:20].replace('Z', '+00:00')).timestamp() >= int(lock.stat().st_mtime)
    return False


def new_lines(log, offsets, first):
    size = log.stat().st_size
    offset = offsets.get(str(log), size if first else 0)
    offset = 0 if size < offset else offset
    with log.open('rb') as f:
        f.seek(offset)
        chunk = f.read()
    complete = chunk[:chunk.rfind(b'\n') + 1]
    offsets[str(log)] = offset + len(complete)
    return complete.decode(errors='replace').splitlines()


def main():
    parser = argparse.ArgumentParser(description='Vast billing sentinel')
    parser.add_argument('--state', type=Path, help='state directory (default ~/.config/v2d/sentinel)')
    parser.add_argument('--locks', type=Path, default=REPO / 'data/vast-controller-locks')
    parser.add_argument('--dry-run', action='store_true', help='print alerts instead of sending them (needs --state)')
    parser.add_argument('--test', action='store_true', help='send one test message and exit')
    a = parser.parse_args()
    if a.test and a.dry_run:
        parser.error('--test sends a real message; it cannot be a dry run')
    if a.test:
        sys.exit(0 if deliver('test', 'V2D sentinel test: Vast billing alerts will arrive in this chat.', False) else 1)
    if a.dry_run and a.state is None:
        parser.error('--dry-run needs its own --state, so it cannot consume alerts meant for the real chat')
    a.state = a.state or Path.home() / '.config/v2d/sentinel'
    a.state.mkdir(parents=True, exist_ok=True)
    path = a.state / 'state.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    first = not state
    locks = {int(p.stem): p for p in a.locks.glob('*.lock') if p.stem.isdecimal()}
    owners = {i: Path(t) for i, t in ((i, p.read_text().strip()) for i, p in locks.items()) if t}
    open_locks = held(list(locks.values()))
    bound = {i for i, p in locks.items()
             if (p in open_locks or p.resolve() in open_locks) and i in owners and armed(p, owners[i])}
    exposed = len({owners[i] for i, p in locks.items()
                   if i in owners and (p in open_locks or p.resolve() in open_locks or not (owners[i] / 'STOPPED').exists())})
    offsets, logs = state.setdefault('offsets', {}), {}
    for run in set(owners.values()):
        if (run / 'watchdog.log').is_file():
            logs[run.name] = new_lines(run / 'watchdog.log', offsets, first)
    # Log offsets advance as lines are read, so an alert is kept until it is delivered.
    pending = state.setdefault('pending', {})
    pending.update({k: '\U0001F6A8 V2D ' + t for k, t in check(state, time.time(), vast_instances(), bound, logs, exposed)})
    for key, text in list(pending.items()):
        if deliver(key, text, a.dry_run):
            del pending[key]
            state['sent'][key] = time.time()  # the hourly cooldown runs from delivery, not from queueing
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(state))
    tmp.replace(path)


if __name__ == '__main__':
    main()
