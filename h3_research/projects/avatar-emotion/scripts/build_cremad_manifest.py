#!/usr/bin/env python3
"""Build M3 training manifest from CREMA-D VideoFlash + AudioWAV.
Output: cremad_manifest.json rows {video, audio, actor, emotion, sentence, duration}
Emotion codes: ANG ANX DIS FEA HAP NEU SAD (SAP≈FEA second take suffix)."""
import json, subprocess, sys
from pathlib import Path

CREMAD = Path('/ssd2/jinzili/avdata/cremad')
OUT = Path('/ssd2/jinzili/avdata/m3_manifest.json')
EMO = {'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear', 'HAP': 'happy',
       'NEU': 'neutral', 'SAD': 'sad', 'ANX': 'anxious'}

def ffprobe_dur(p):
    try:
        r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                            '-of', 'default=noprint_wrappers=1:nokey=1', str(p)],
                           capture_output=True, text=True, timeout=20)
        return float(r.stdout.strip())
    except Exception:
        return None

def main():
    vids = sorted((CREMAD / 'VideoFlash').glob('*.flv'))
    rows, bad = [], 0
    for v in vids:
        parts = v.stem.split('_')
        if len(parts) < 3:
            continue
        actor, sent, emo = parts[0], parts[1], parts[2]
        audio = CREMAD / 'AudioWAV' / f"{v.stem}.wav"
        if not audio.exists():
            bad += 1
            continue
        rows.append({'video': str(v), 'audio': str(audio), 'actor': int(actor),
                     'emotion': EMO.get(emo, emo.lower()), 'sentence': sent})
    # durations for a subsample to estimate
    import random
    random.seed(0)
    for r in random.sample(rows, min(60, len(rows))):
        d = ffprobe_dur(r['video'])
        r['dur_sample'] = d
    import collections
    c = collections.Counter(r['emotion'] for r in rows)
    actors = sorted({r['actor'] for r in rows})
    json.dump({'rows': rows, 'n': len(rows), 'emotion_dist': dict(c),
               'n_actors': len(actors), 'actors': actors}, open(OUT, 'w'))
    print(f"manifest: {len(rows)} rows | {len(actors)} actors | bad {bad}")
    print('emotion dist:', dict(c))

if __name__ == '__main__':
    main()
