"""Debug: trace action selection for all tasks — standalone."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quick_test import _pick_action
import httpx

BASE = 'http://localhost:7860'
c = httpx.Client(timeout=30)

for task in ['pan_aadhaar_link', 'passport_fresh', 'driving_licence', 'vehicle_registration']:
    obs = c.post(f'{BASE}/reset', json={'task': task, 'seed': 99}).json()['observation']
    completed_raw = []
    print(f"\n{'='*70}\n{task} seed=99\n{'='*70}")
    for step in range(1, 40):
        avail = obs.get('available_actions', [])
        hint = obs.get('status_summary', '')
        recent_done = set(s.split(":")[0] for s in completed_raw[-12:])
        act, par = _pick_action(obs, avail, recent_done)
        r = c.post(f'{BASE}/step', json={'action_type': act, 'parameters': par}).json()
        obs = r['observation']
        completed_raw = obs.get('completed_steps', [])
        err = obs.get('last_action_error', '')
        marker = ' !!ERR' if err else ''
        print(f"  s{step:2d} act={act:25s} hint={repr(hint[:60]):62s}{marker} {err[:50] if err else ''}")
        if r.get('done'):
            print(f"  DONE reward={r.get('reward', 0):.2f}")
            break
