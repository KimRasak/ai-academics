"""Build traindata3: cross-emotion pairs (ref emotion != target emotion) to teach
prompt-overrides-reference, plus the preservation samples.

Cross pairs use char 1534125952 (v2 videos: angry/joyful/sad/surprised real audio,
portraits available). Preservation samples from outputs_preserve (10).
"""
import json, os

H3 = "/gluster_hdd_cv/user/jinzili/code/anime_style/model_demo/minimax_h3"
V2 = H3 + "/outputs/20260825_expressions_v2"
V2REF = H3 + "/inputs/expressions_0825_v2"
PRES = "/gluster_hdd_cv/user/jinzili/tmp/pilot/outputs_preserve"

EMO = {
    "angry": "She frowns angrily.",
    "joyful": "She smiles joyfully.",
    "sad": "She looks sad.",
    "surprised": "She looks surprised.",
}

rows = []

# --- cross-emotion pairs (ref=A portrait, target=B video, caption=B) for char 1534125952 ---
for ref_emo in EMO:
    for tgt_emo in EMO:
        if ref_emo == tgt_emo:
            continue
        rows.append({
            "id": f"cross_{ref_emo}_to_{tgt_emo}",
            "task": "ref2va",
            "caption": "<Picture 1> " + EMO[tgt_emo] + " Keep the same anime character.",
            "target_video": V2 + f"/{tgt_emo}_1534125952_v1_raw.mp4",
            "target_audio": None,
            "reference_images": [V2REF + f"/{ref_emo}_1534125952_v1_raw.png"],
            "reference_videos": [], "reference_audios": [],
        })

# --- preservation samples (ref same emotion as target) ---
jobs = json.load(open("/gluster_hdd_cv/user/jinzili/tmp/pilot/jobs_preserve.json"))
for j in jobs:
    rows.append({
        "id": j["case_id"], "task": "ref2va",
        "caption": j["prompt"], "target_audio": None,
        "target_video": PRES + "/" + j["case_id"] + "_base.mp4",
        "reference_images": j["references"],
        "reference_videos": [], "reference_audios": [],
    })

out = "/gluster_hdd_cv/user/jinzili/tmp/pilot/traindata3"
os.makedirs(out, exist_ok=True)
with open(out + "/train.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", len(rows), "samples (", 12, "cross +", len(rows) - 12, "pres )")
