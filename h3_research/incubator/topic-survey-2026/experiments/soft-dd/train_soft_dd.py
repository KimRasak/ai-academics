"""I1: Seed-consistent soft-DD calibration (trainable) + I3: variance predictor.

Soft model: soft(video) = mean_i sigmoid((s_i - t + b) / T), s_i = RAFT per-pair
max_rad, t = released threshold. Monotone in every s_i; T->0 recovers hard DD.
Trained params: T, b (gradient descent, torch).
Objective (train seeds only): minimize within-prompt across-seed variance
  + lambda * fidelity to released binary (BCE at 0.5)
  + mu * between-prompt separation (negative ICC penalty).
Report: within-prompt SD / pseudo-flip rate on HELD-OUT seeds, ICC, Spearman
vs binary. I3: leave-one-prompt-out variance prediction (48 model-prompt pts).
"""
import json, glob, statistics, math, random
from collections import defaultdict
from pathlib import Path
import torch

root = Path(__file__).resolve().parent
seed_data = root.parents[3] / 'projects' / 'seed-sensitivity' / 'data'
mf = root / 'data'
meta = {  # model -> (vt glob patterns, seed range note)
 'H3': [seed_data/'scivbench/e3_new_vt/h3_base/*.json', seed_data/'scivbench/vt_full/h3_base/*.json',
        seed_data/'scivbench/e3_s6_11_vt/h3_base/*.json'],
 'Wan': [seed_data/'wan_cross/vt2/*.json', seed_data/'wan_cross/vt_s6_11/*.json'],
 'LTX': [seed_data/'ltx_cross/vt/*.json'],
}
# locate motion json per (model, prompt, seed)
vid2case = {}
for p in glob.glob(str(mf/'*.json')):
    d = json.load(open(p))
    v = Path(d['video']).stem  # e.g. <sha>_seed6
    vid2case[v] = d

def vt_scores(patterns):
    byp = defaultdict(dict)
    for pat in patterns:
        for p in glob.glob(str(pat)):
            d = json.load(open(p)); vid = d.get('video_id','')
            if '_seed' not in vid: continue
            byp[vid.rsplit('_seed',1)[0]][int(vid.rsplit('_seed',1)[1])] = d
    return byp

def soft_batch(S, t, T, b):
    # S: [N, P] tensor of per-pair scores (padded), mask [N,P]
    return torch.sigmoid((S - t + b) / T).sum(dim=1) / MASK.sum(dim=1)

# ---------------- build dataset ----------------
data = {}  # (model, prompt, seed) -> dict(scores=list, thres, count_num, VT, binary)
for model, pats in meta.items():
    for pat in pats:
        for p in glob.glob(str(pat)):
            d = json.load(open(p)); vid = d.get('video_id','')
            if '_seed' not in vid: continue
            pid, s = vid.rsplit('_seed',1)
            key = (model, pid, int(s))
            mfkey = vid
            if mfkey not in vid2case: continue
            data[key] = {'sc': vid2case[mfkey]['static_scores'], 't': vid2case[mfkey]['thres'],
                         'c': vid2case[mfkey]['count_num'], 'VT': d['VT'], 'bin': d['dynamic_degree']}
models = sorted({k[0] for k in data})
print('models:', models, '| total videos with motion feats:', len(data))

def pad_matrix(seqs, device='cpu'):
    P = max(len(s) for s in seqs)
    X = torch.zeros(len(seqs), P); M = torch.zeros(len(seqs), P)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = torch.tensor(s); M[i, :len(s)] = 1.0
    return X.to(device), M.to(device)

def train_model_group(model, train_seeds, test_seeds, epochs=400, lr=0.05, lam=0.5):
    byp = defaultdict(dict)
    for (m, pid, s), r in data.items():
        if m == model: byp[pid][s] = r
    tr = [(pid, s) for pid in byp for s in train_seeds if s in byp[pid]]
    te = [(pid, s) for pid in byp for s in test_seeds if s in byp[pid]]
    print(f'[dbg {model}] byp={len(byp)} tr={len(tr)} te={len(te)}', flush=True)
    seqs = [data[(model,pid,s)]['sc'] for pid,s in tr]
    ts   = [data[(model,pid,s)]['t']  for pid,s in tr]
    X, M = pad_matrix(seqs); tvec = torch.tensor(ts).unsqueeze(1)
    yb = torch.tensor([1.0 if data[(model,pid,s)]['bin'] else 0.0 for pid,s in tr])
    pids = [pid for pid,s in tr]
    uniq = sorted(set(pids)); pid_idx = torch.tensor([uniq.index(p) for p in pids])
    logT = torch.nn.Parameter(torch.tensor(1.5)); b = torch.nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.Adam([logT, b], lr=lr)
    for ep in range(epochs):
        soft = torch.sigmoid((X - tvec + b) / torch.exp(logT)); soft = (soft*M).sum(1)/M.sum(1)
        # within-prompt variance
        wv = 0.0
        for gi in range(len(uniq)):
            sel = pid_idx == gi
            if sel.sum() > 1: wv = wv + soft[sel].var(unbiased=False)
        wv = wv / len(uniq)
        sep = soft.var(unbiased=False) - wv  # between-prompt variance total - within
        fid = torch.nn.functional.binary_cross_entropy(soft.clamp(1e-4, 1-1e-4), yb)
        loss = wv - lam*sep + lam*fid
        opt.zero_grad(); loss.backward(); opt.step()
    T = float(torch.exp(logT).detach()); bb = float(b.detach())
    # evaluate on test seeds: pseudo-flip = |soft(a)-soft(b)| crossing 0.5 OR arg dispersion
    def soft_of(pid, s):
        r = data[(model,pid,s)]
        Xs, Ms = pad_matrix([r['sc']]); tv = torch.tensor([[r['t']]])
        return float(((torch.sigmoid((Xs - tv + bb)/T)*Ms).sum(1)/M.sum(1))[0])
    te_ok = [(pid,s) for pid,s in te if (model,pid,s) in data]
    print(f'[dbg {model}] te_ok={len(te_ok)}', flush=True)
    te_soft = {(pid,s): soft_of(pid,s) for pid,s in te_ok}
    # within-prompt SD of soft vs binary on TEST
    sds_soft, sds_bin = [], []
    print(f'[dbg {model}] te_soft={len(te_soft)} sds_pre', flush=True)
    _p,_s = te_ok[0]; print(f'[dbg {model}] sample key=({_p!r},{_s!r}) type_s={type(_s)} in_te_soft={(_p,_s) in te_soft} | lookup probe:', [ (p,s) in te_soft for p,s in te_ok[:3]], flush=True)
    n_loop = 0
    for pid in sorted({p for p,_ in te_ok}):
        ss = [te_soft.get((pid,s)) for s in test_seeds]
        ss = [x for x in ss if x is not None]
        n_loop += 1
        if n_loop == 1:
            print(f'[dbg {model}] pid={pid} test_seeds={test_seeds} len(ss)={len(ss)}', flush=True)
        if len(ss) > 1:
            sds_soft.append(statistics.pstdev(ss))
            sds_bin.append(statistics.pstdev([data[(model,pid,s)]['bin'] for s in test_seeds if (model,pid,s) in data]))
    print(f'[dbg {model}] n_loop={n_loop} sds_soft={len(sds_soft)}', flush=True)
    # pseudo flip rate: pairs whose soft values straddle 0.5
    flips = tot = 0
    for pid in {p for p,_ in te_ok}:
        ss = [te_soft[(p,s)] for s in test_seeds if (p,s) in te_soft]
        for a, b2 in __import__('itertools').combinations(ss, 2):
            tot += 1
            if (a-0.5)*(b2-0.5) < 0: flips += 1
    return {'T': T, 'b': bb, 'soft_sd_test': statistics.mean(sds_soft),
            'bin_sd_test': statistics.mean(sds_bin),
            'pseudo_flip_test': flips/tot if tot else None}

results = {}
splits = {'0-5/6-11': (list(range(6)), list(range(6,12))),
          '0-3|6-9': (list(range(4)), list(range(6,10)))}
for model in models:
    results[model] = {}
    for sname,(tr,te) in splits.items():
        r = train_model_group(model, tr, te)
        results[model][sname] = r
        print(f'{model} [{sname}] T={r["T"]:.4f} b={r["b"]:.4f} '
              f'test softSD={r["soft_sd_test"]:.4f} binSD={r["bin_sd_test"]:.4f} '
              f'pseudoFlip={r["pseudo_flip_test"]}')
json.dump(results, open(root/'results/soft_dd_training.json','w'), indent=2)
print('WROTE results/soft_dd_training.json')
