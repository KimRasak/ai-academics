import json

H3 = "/gluster_hdd_cv/user/jinzili/code/anime_style/model_demo/minimax_h3"
V2 = H3 + "/outputs/20260825_expressions_v2"
V2REF = H3 + "/inputs/expressions_0825_v2"
PILOT = "/gluster_hdd_cv/user/jinzili/tmp/pilot/outputs"
PILOTREF = H3 + "/inputs/expressions_0825"

rows = []

# --- v2 emotion-preservation targets (reliable; reference shows the emotion) ---
v2_map = [
    ("angry", "She frowns angrily.", "angry_1534125952_v1_raw"),
    ("joyful", "She smiles joyfully.", "joyful_1534125952_v1_raw"),
    ("sad", "She looks sad.", "sad_1534125952_v1_raw"),
    ("surprised", "She looks surprised.", "surprised_1534125952_v1_raw"),
]
for emo, cap, stem in v2_map:
    rows.append({
        "id": "v2_" + emo,
        "task": "ref2va",
        "caption": "<Picture 1> " + cap + " Keep the same anime character.",
        "target_video": V2 + "/" + stem + ".mp4",
        "target_audio": None,
        "reference_images": [V2REF + "/" + stem + ".png"],
        "reference_videos": [],
        "reference_audios": [],
    })

# --- pilot cross-emotion targets (reference shows source emotion, target is the output) ---
pilot_map = [
    ("x01_neutral_to_joyful", "She smiles joyfully."),
    ("x02_neutral_to_sad", "She looks sad."),
    ("x03_angry_to_happy", "She laughs happily."),
    ("x05_amused_to_surprised", "She looks surprised."),
    ("x08_curious_to_angry", "She looks angry."),
]
pilot_refs = {
    "x01_neutral_to_joyful": "a15_neutral_1537882489_v1.png",
    "x02_neutral_to_sad": "a15_neutral_1537882489_v1.png",
    "x03_angry_to_happy": "a02_angry_1534927518_v1.png",
    "x05_amused_to_surprised": "a01_amused_1534125952_v1.png",
    "x08_curious_to_angry": "a07_curious_1536617199_v1.png",
}
for cid, cap in pilot_map:
    rows.append({
        "id": cid,
        "task": "ref2va",
        "caption": "<Picture 1> " + cap + " Keep the same anime character.",
        "target_video": PILOT + "/" + cid + ".mp4",
        "target_audio": None,
        "reference_images": [PILOTREF + "/" + pilot_refs[cid]],
        "reference_videos": [],
        "reference_audios": [],
    })

out = "/gluster_hdd_cv/user/jinzili/tmp/pilot/traindata"
import os
os.makedirs(out, exist_ok=True)
with open(out + "/train.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", len(rows), "samples to", out + "/train.jsonl")
