"""Build expanded EmoH3 training manifest — clean preservation targets only.
18 samples: 8 existing v2/v3 (char 1534125952) + 10 new preservation videos.
"""
import json, os

H3 = "/gluster_hdd_cv/user/jinzili/code/anime_style/model_demo/minimax_h3"
V2 = H3 + "/outputs/20260825_expressions_v2"
V3 = H3 + "/outputs/20260825_expressions_v3"
V2REF = H3 + "/inputs/expressions_0825_v2"
PRES = "/gluster_hdd_cv/user/jinzili/tmp/pilot/outputs_preserve"

rows = []

# --- existing v2 raw (4) ---
for emo, cap in [
    ("angry", "She frowns angrily."),
    ("joyful", "She smiles joyfully."),
    ("sad", "She looks sad."),
    ("surprised", "She looks surprised."),
]:
    stem = f"{emo}_1534125952_v1_raw"
    rows.append({
        "id": "v2_" + emo, "task": "ref2va",
        "caption": "<Picture 1> " + cap + " Keep the same anime character.",
        "target_video": V2 + "/" + stem + ".mp4", "target_audio": None,
        "reference_images": [V2REF + "/" + stem + ".png"],
        "reference_videos": [], "reference_audios": [],
    })

# --- v3 seed1234 variants (4) ---
for emo, cap in [
    ("angry", "She frowns angrily."),
    ("joyful", "She smiles joyfully."),
    ("sad", "She looks sad."),
    ("surprised", "She looks surprised."),
]:
    stem = f"{emo}_1534125952_v1_raw_seed1234"
    rows.append({
        "id": "v3s_" + emo, "task": "ref2va",
        "caption": "<Picture 1> " + cap + " Keep the same anime character.",
        "target_video": V3 + "/" + stem + ".mp4", "target_audio": None,
        "reference_images": [V2REF + "/" + stem + ".png"],
        "reference_videos": [], "reference_audios": [],
    })

# --- new preservation videos (10) from jobs_preserve.json ---
jobs = json.load(open("/gluster_hdd_cv/user/jinzili/tmp/pilot/jobs_preserve.json"))
for j in jobs:
    rows.append({
        "id": j["case_id"], "task": "ref2va",
        "caption": j["prompt"], "target_audio": None,
        "target_video": PRES + "/" + j["case_id"] + "_base.mp4",
        "reference_images": j["references"],
        "reference_videos": [], "reference_audios": [],
    })

out = "/gluster_hdd_cv/user/jinzili/tmp/pilot/traindata2"
os.makedirs(out, exist_ok=True)
with open(out + "/train.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", len(rows), "samples")
for r in rows:
    print(r["id"], "->", r["target_video"].split("/")[-1])
