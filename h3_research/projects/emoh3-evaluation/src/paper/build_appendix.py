#!/usr/bin/env python3
"""Emit the full-verdict appendix (LaTeX longtable + markdown) from the verified matrix.

Source of truth = DEFINITIVE_RESULTS.md + raw judge logs, already cross-checked. Config key:
  (model, case, steps, seed, frames) -> "word:score"
"""
import json
from pathlib import Path

H = Path("/Users/jinzili/code/ai-academics/h3_research")
CASES = ["x01_neutral_to_joyful","x02_neutral_to_sad","x03_angry_to_happy",
         "x04_joyful_to_worried","x05_amused_to_surprised","x06_anxious_to_relieved",
         "x07_sad_to_cheerful","x08_curious_to_angry"]
SHORT = {"x01_neutral_to_joyful":"x01","x02_neutral_to_sad":"x02","x03_angry_to_happy":"x03",
         "x04_joyful_to_worried":"x04","x05_amused_to_surprised":"x05","x06_anxious_to_relieved":"x06",
         "x07_sad_to_cheerful":"x07","x08_curious_to_angry":"x08"}
TARGET = {"x01":"joyful","x02":"sad","x03":"happy","x04":"worried","x05":"surprised",
          "x06":"relieved","x07":"cheerful","x08":"angry"}

# ---- verified full matrix ----
# (model, case, steps, seed, frames) -> verdict
M = {}

def put(model, case, steps, seed, frames, verdict):
    M[(model, case, steps, seed, frames)] = verdict

# base 8s s42 f4 (pilot): all 8
for c, v in [("x01_neutral_to_joyful","neutral:50"),("x02_neutral_to_sad","neutral:50"),
             ("x03_angry_to_happy","Happy:80"),("x04_joyful_to_worried","neutral:50"),
             ("x05_amused_to_surprised","neutral:100"),("x06_anxious_to_relieved","neutral:50"),
             ("x07_sad_to_cheerful","neutral:50"),("x08_curious_to_angry","neutral:100")]:
    put("base", c, 8, 42, 4, v)
# base 16s
put("base","x01_neutral_to_joyful",16,42,8,"Happy:90")
put("base","x03_angry_to_happy",16,42,8,"Happy:90")
put("base","x01_neutral_to_joyful",16,1234,8,"Neutral:50")
put("base","x07_sad_to_cheerful",16,1234,8,"neutral:50")
put("base","x02_neutral_to_sad",16,1234,8,"neutral:100")  # decisive control

# v1 8s s42 f4 (lora)
for c, v in [("x01_neutral_to_joyful","Happy:90"),("x02_neutral_to_sad","Neutral:80"),
             ("x03_angry_to_happy","Happy:90"),("x04_joyful_to_worried","Neutral:80"),
             ("x05_amused_to_surprised","Neutral:100"),("x06_anxious_to_relieved","sad:80"),
             ("x07_sad_to_cheerful","happy:80"),("x08_curious_to_angry","Neutral:100")]:
    put("v1", c, 8, 42, 4, v)

# v2 8s s42 f4 (lora2) + f8 (judge8, subset) + s1234 f4 (lora2s)
v2_42_4 = {"x01_neutral_to_joyful":"Happy:90","x02_neutral_to_sad":"Neutral:80",
           "x03_angry_to_happy":"Happy:90","x04_joyful_to_worried":"Neutral:80",
           "x05_amused_to_surprised":"surprised:80","x06_anxious_to_relieved":"sad:80",
           "x07_sad_to_cheerful":"happy:80","x08_curious_to_angry":"Neutral:100"}
for c,v in v2_42_4.items():
    put("v2", c, 8, 42, 4, v)
for c,v in [("x01_neutral_to_joyful","Happy:90"),("x05_amused_to_surprised","Neutral:100"),
            ("x06_anxious_to_relieved","sad:80"),("x07_sad_to_cheerful","happy:80")]:
    put("v2", c, 8, 42, 8, v)
for c,v in [("x01_neutral_to_joyful","neutral:50"),("x02_neutral_to_sad","Neutral:100"),
            ("x03_angry_to_happy","Neutral:100"),("x04_joyful_to_worried","Neutral:100"),
            ("x05_amused_to_surprised","Neutral:100"),("x06_anxious_to_relieved","sad:100"),
            ("x07_sad_to_cheerful","Neutral:50"),("x08_curious_to_angry","Neutral:100")]:
    put("v2", c, 8, 1234, 4, v)
# v2 16s
for c,v in [("x01_neutral_to_joyful","happy:80"),("x05_amused_to_surprised","Neutral:100"),
            ("x07_sad_to_cheerful","happy:80")]:
    put("v2", c, 16, 42, 8, v)
put("v2","x07_sad_to_cheerful",16,1234,8,"neutral:50")
put("v2","x01_neutral_to_joyful",16,1234,8,"happy:80")

# v3 16s s42 (rows 1-8) + s1234 (rows 9-16) from v3_judge_results.jsonl
v3_42 = {"x01":"happy:80","x02":"neutral:100","x03":"happy:80","x04":"neutral:50",
         "x05":"neutral:100","x06":"neutral:50","x07":"happy:80","x08":"neutral:100"}
v3_1234 = {"x01":"happy:80","x02":"sad:80","x03":"happy:90","x04":"neutral:100",
           "x05":"happy:80","x06":"happy:80","x07":"happy:80","x08":"neutral:100"}
for k,v in v3_42.items():
    c = [x for x in CASES if SHORT[x]==k][0]
    put("v3", c, 16, 42, 8, v)
for k,v in v3_1234.items():
    c = [x for x in CASES if SHORT[x]==k][0]
    put("v3", c, 16, 1234, 8, v)
put("v3","x02_neutral_to_sad",16,999,8,"neutral:80")
put("v3","x07_sad_to_cheerful",16,999,8,"Neutral:50")

# v4 16s s42 (rows 1-8) + s1234 (rows 9-16) from v4_judge_results.jsonl
v4_42 = {"x01":"Neutral:50","x02":"neutral:50","x03":"neutral:100","x04":"neutral:50",
         "x05":"Neutral:100","x06":"Neutral:50","x07":"neutral:50","x08":"neutral:50"}
v4_1234 = {"x01":"Neutral:50","x02":"sad:80","x03":"Neutral:50","x04":"neutral:100",
           "x05":"Neutral:50","x06":"neutral:100","x07":"happy:80","x08":"neutral:50"}
for k,v in v4_42.items():
    c = [x for x in CASES if SHORT[x]==k][0]
    put("v4", c, 16, 42, 8, v)
for k,v in v4_1234.items():
    c = [x for x in CASES if SHORT[x]==k][0]
    put("v4", c, 16, 1234, 8, v)

# ---- emit LaTeX longtable ----
def ltx_esc(s):
    return s.replace("_","\\_")

COLS = [("base","8s s42 f4"),("base","16s s42 f8"),("base","16s s1234 f8"),
        ("v1","8s s42 f4"),("v2","8s s42 f4"),("v2","8s s42 f8"),("v2","8s s1234 f4"),
        ("v2","16s s42 f8"),("v2","16s s1234 f8"),
        ("v3","16s s42 f8"),("v3","16s s1234 f8"),("v3","16s s999 f8"),
        ("v4","16s s42 f8"),("v4","16s s1234 f8")]
def cfg(model, col):
    steps, seed, frames = 8,42,4
    parts = col.split()
    steps = int(parts[0].replace("s",""))
    seed = int(parts[1].replace("s",""))
    frames = int(parts[2].replace("f",""))
    return (model, steps, seed, frames)

lines = []
lines.append("% Appendix: full judge verdicts (Qwen2-VL-7B, do_sample=False). WORD:score; "
             "bold = matches target emotion.")
lines.append(r"\begin{center}\small")
lines.append(r"\begin{tabular}{@{}l" + "c"*len(COLS) + r"@{}}")
lines.append(r"\toprule")
hdr = [r"Task (target)"] + [ltx_esc(c) for c in [" ".join(x) for x in COLS]]
lines.append(" & ".join(hdr) + r" \\")
lines.append(r"\midrule")
for case in CASES:
    short = SHORT[case]
    row = [rf"{short} ({TARGET[short]})"]
    for model, col in COLS:
        m,s,seed,f = cfg(model,col)
        v = M.get((model,case,s,seed,f), "--")
        word = v.split(":")[0].lower()
        hit = (word == TARGET[short])
        cell = f"\\textbf{{{v}}}" if hit else v
        row.append(cell)
    lines.append(" & ".join(row) + r" \\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}\end{center}")
tex = "\n".join(lines)

# wrap in a standalone appendix file
app = r"""% Appendix for EmoH3 paper — full judge verdict matrix (auto-generated).
% Include with \input{paper_emoh3_appendix} (requires booktabs).
\section{Full judge verdicts}
""" + tex + r"""

All verdicts are Qwen2-VL-7B (do\_sample=False) over a fixed 12-class emotion taxonomy with a
confidence score; a cell is bold when the judged emotion equals the task target. ``f4/f8'' is the
number of judge frames; ``s42/s1234/s999'' the sampling seed; ``8s/16s'' the inference steps.
Empty (--) cells are configurations that were not run.
"""
(H / "paper_emoh3_appendix.tex").write_text(app)

# markdown version
md = ["# Appendix: Full judge verdicts (Qwen2-VL-7B, do_sample=False)",
      "", "| Task (target) | " + " | ".join(" ".join(x) for x in COLS) + " |",
      "|---|" + "---|"*len(COLS)]
for case in CASES:
    short = SHORT[case]
    row = [f"{short} ({TARGET[short]})"]
    for model, col in COLS:
        m,s,seed,f = cfg(model,col)
        v = M.get((model,case,s,seed,f), "--")
        word = v.split(":")[0].lower()
        row.append(f"**{v}**" if word==TARGET[short] else v)
    md.append("| " + " | ".join(row) + " |")
(H / "paper_emoh3_appendix.md").write_text("\n".join(md) + "\n")

print("wrote paper_emoh3_appendix.tex and .md")
print("sample LaTeX rows:")
print("\n".join(lines[:4]))
print("...")
print("\n".join(lines[-3:]))
