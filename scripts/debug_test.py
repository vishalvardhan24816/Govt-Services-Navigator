"""Debug failing seeds — standalone, uses quick_test agent logic."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quick_test import _pick_action, _extract_form_data
import httpx
c = httpx.Client(timeout=30)
BASE = 'http://localhost:7860'

for task, seed in [('passport_fresh', 0), ('passport_fresh', 13), ('driving_licence', 31415), ('vehicle_registration', 999), ('vehicle_registration', 7777)]:
    obs = c.post(f'{BASE}/reset', json={'task': task, 'seed': seed}).json()['observation']
    completed_raw = []
    for step in range(1, 36):
        avail = obs.get('available_actions', [])
        recent_done = set(s.split(":")[0] for s in completed_raw[-12:])
        act, par = _pick_action(obs, avail, recent_done)
        r = c.post(f'{BASE}/step', json={'action_type': act, 'parameters': par}).json()
        obs = r['observation']
        completed_raw = obs.get('completed_steps', [])
        err = obs.get('last_action_error')
        e = err[:50] if err else 'ok'
        print(f"{task}|{seed}|{step:2d}|{act:25s}|r={r.get('reward',0):.2f}|{e}")
        if r.get('done'):
            break
    print()
