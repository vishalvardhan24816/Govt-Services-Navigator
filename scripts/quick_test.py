"""Quick local test: run a simple rule-based agent against all 4 tasks with multiple seeds.

This is a LOCAL TEST UTILITY only — not part of the inference script.
The rule-based agent here verifies that the environment is solvable and
rewards are reachable. The actual inference.py uses a pure LLM baseline.
"""
import sys, json, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import httpx

BASE = 'http://localhost:7860'
c = httpx.Client(timeout=30)

# ── Standalone rule-based agent (local testing only) ──

ISSUE_TO_FIX = {
    "name_mismatch": "aadhaar_name", "address_mismatch": "aadhaar_address",
    "dob_mismatch": "aadhaar_dob", "aadhaar_address_outdated": "aadhaar_address",
    "address_outdated": "aadhaar_address", "address_proof_invalid": "address_proof",
    "photo_invalid": "photo", "insurance_expired": "insurance",
    "invoice_discrepancy": "invoice", "inspection_failure": "chassis",
    "missing_puc": "puc", "hypothecation_required": "hypothecation",
    "medical_cert_missing": "medical", "photo_rejected": "photo",
}

def _extract_form_data(obs):
    summary = obs.get("citizen_summary", "")
    docs = obs.get("citizen_documents", {})
    form = {}
    m = re.search(r'Citizen:\s*([^,\n]+)', summary)
    if m: form["applicant_name"] = m.group(1).strip()
    for dk in ("aadhaar_card", "birth_certificate"):
        dd = docs.get(dk)
        if isinstance(dd, dict):
            f = dd.get("fields", dd)
            if "dob" in f: form["dob"] = str(f["dob"]); break
        else:
            m2 = re.search(r"'dob':\s*'(\d{4}-\d{2}-\d{2})'", str(dd or ""))
            if m2: form["dob"] = m2.group(1); break
    m = re.search(r'Current Address:\s*([^\n]+)', summary)
    if m: form["present_address"] = m.group(1).strip().rstrip(",")
    m = re.search(r'Father:\s*([^,\n]+)', summary)
    if m: form["father_name"] = m.group(1).strip()
    m = re.search(r'Mother:\s*([^,\n]+)', summary)
    if m: form["mother_name"] = m.group(1).strip()
    m = re.search(r'Gender:\s*([^,\n]+)', summary)
    if m: form["gender"] = m.group(1).strip()
    return form

def _pick_action(obs, avail, recent_done):
    pending = obs.get("pending_issues", [])
    phase = obs.get("current_phase", "").lower()
    day = obs.get("simulated_day", 0)
    # Fix issues first
    if pending and "fix_document" in avail:
        for issue, target in ISSUE_TO_FIX.items():
            if issue in " ".join(pending).lower():
                return "fix_document", {"target": target}
    if pending and "gather_document" in avail and "fix_document" not in avail:
        return "gather_document", {"target": "all"}
    # Priority order
    for act in ["check_prerequisites", "compare_documents", "gather_document",
                 "fill_form", "pay_fee", "book_appointment", "take_test", "submit_application"]:
        if act in avail and act not in recent_done:
            par = {}
            if act == "take_test":
                par = {"test_type": "practical" if "driving" in phase or "permanent" in phase else "written"}
            elif act == "gather_document":
                par = {"target": "all"}
            elif act == "fill_form":
                par = _extract_form_data(obs)
            return act, par
    if "wait" in avail and day < 15:
        return "wait", {"days": 7}
    if "check_status" in avail:
        return "check_status", {}
    return avail[0] if avail else "check_status", {}


SEEDS = [42, 123, 7, 99, 2024, 0, 1, 13, 256, 999, 7777, 31415]
tasks = ['pan_aadhaar_link', 'passport_fresh', 'driving_licence', 'vehicle_registration']
all_results = {}

for task in tasks:
    task_scores = []
    for seed in SEEDS:
        obs = c.post(f'{BASE}/reset', json={'task': task, 'seed': seed}).json()['observation']
        reward, errs = 0.0, 0
        completed_raw = []
        for step in range(1, 36):
            avail = obs.get('available_actions', [])
            recent_done = set(s.split(":")[0] for s in completed_raw[-12:])
            act, par = _pick_action(obs, avail, recent_done)
            r = c.post(f'{BASE}/step', json={'action_type': act, 'parameters': par}).json()
            obs = r['observation']
            completed_raw = obs.get('completed_steps', [])
            reward = r.get('reward', 0.0)
            err = obs.get('last_action_error')
            if err: errs += 1
            if r.get('done'): break
        task_scores.append(reward)
        status = "PASS" if reward >= 0.80 else "FAIL"
        print(f"  {task} seed={seed}: {reward:.2f} [{status}] steps={step} errors={errs}")
    avg = sum(task_scores) / len(task_scores)
    mn = min(task_scores)
    all_results[task] = {'avg': avg, 'min': mn, 'scores': task_scores}
    print(f"  >>> {task}: avg={avg:.2f} min={mn:.2f}\n")

print("=" * 60)
print("SUMMARY")
print("=" * 60)
all_pass = True
for task, info in all_results.items():
    status = "PASS" if info['min'] >= 0.80 else "WARN" if info['avg'] >= 0.80 else "FAIL"
    if info['min'] < 0.80: all_pass = False
    scores = " ".join(f"{s:.2f}" for s in info['scores'])
    print(f"  {task}: avg={info['avg']:.2f} min={info['min']:.2f} [{status}] scores=[{scores}]")
overall = sum(r['avg'] for r in all_results.values()) / len(all_results)
print(f"  Overall avg: {overall:.2f}")
print(f"  All min >= 0.80: {all_pass}")
