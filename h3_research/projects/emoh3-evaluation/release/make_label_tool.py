#!/usr/bin/env python3
"""Generate the blind human-labeling tool (single self-contained index.html).

Reads human_label/MAP.json + the actual files on disk and emits human_label/index.html.
The HTML contains no source/model/condition information: annotators see only item ids
in a randomized order, exactly the same frames the VLM judges saw, and the same
forced-choice 15-class question.

Run: python3 make_label_tool.py
Then open human_label/index.html in a browser (no server needed).
"""
import json
from pathlib import Path

HL = Path(__file__).resolve().parent / "human_label"
MAP = json.loads((HL / "MAP.json").read_text())

EMO = ["happy", "joyful", "sad", "angry", "worried", "cheerful", "surprised", "relieved",
       "anxious", "curious", "neutral", "disgusted", "fearful", "amused", "other"]

# rebuild frame lists from disk (the pack was re-encoded to .jpg after MAP.json was written)
items = []
for iid in sorted(MAP):
    files = sorted(p.name for p in (HL / iid).glob("f*.jpg"))
    if not files:
        continue
    items.append({"id": iid, "kind": MAP[iid]["kind"], "frames": files})

manifest = json.dumps(items, separators=(",", ":"))
emojson = json.dumps(EMO)

HTML = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>Anime emotion labeling (blind)</title>
<style>
 :root { color-scheme: light dark; }
 body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; margin: 0 auto;
        max-width: 1100px; padding: 16px; line-height: 1.5; }
 h1 { font-size: 18px; margin: 0 0 6px; }
 #bar { height: 6px; background: #ddd; border-radius: 3px; overflow: hidden; margin: 8px 0 12px; }
 #barfill { height: 100%; background: #3b82f6; width: 0%; transition: width .2s; }
 #frames { display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 10px; }
 #frames img { height: 150px; border-radius: 4px; cursor: zoom-in; background: #eee; }
 #frames.zoom img { height: 380px; }
 .btns { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
 button { font-size: 15px; padding: 8px 12px; border-radius: 6px; border: 1px solid #999;
          background: #f6f6f6; cursor: pointer; }
 button:hover { background: #e8e8e8; }
 button.sel { background: #3b82f6; color: #fff; border-color: #2563eb; }
 .key { font-size: 11px; opacity: .55; }
 #conf { margin: 4px 0 12px; }
 #done { display: none; }
 textarea { width: 100%; height: 160px; font-family: ui-monospace, Menlo, monospace; font-size: 12px; }
 .muted { color: #666; font-size: 13px; }
 .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
</style></head><body>
<h1>动漫表情标注（盲测）</h1>
<p class="muted">看这些按时间顺序排列的帧，判断角色表达的情绪。<b>强制选择</b>：必须从下面 15 个词里选一个
（和 VLM 判官完全相同的题目与词表），然后选把握程度。快捷键：字母键选情绪，<b>1/2/3</b> 选把握，
<b>Z</b> 回退上一题，点图片可放大。进度自动保存在本机浏览器里，可中断后继续。</p>
<div id="bar"><div id="barfill"></div></div>
<div class="row"><div id="counter" class="muted"></div><div id="kindtag" class="muted"></div></div>
<div id="frames"></div>
<div class="btns" id="emobtns"></div>
<div id="conf" class="row"><span class="muted">把握程度：</span>
 <button data-c="50">1 低 (50)</button>
 <button data-c="75">2 中 (75)</button>
 <button data-c="95">3 高 (95)</button>
 <button id="back">Z 回退</button>
</div>
<div id="done">
 <h2>标注完成，谢谢！</h2>
 <p class="muted">点下面按钮下载结果文件，把它交回给我（放在 <code>human_label/</code> 目录下即可）。
 如果下载被拦，直接全选复制文本框内容发我。</p>
 <p><button id="dl">下载 labels_&lt;你的名字&gt;.json</button>
    <input id="who" placeholder="标注者代号（如 A / B）" value="A"></p>
 <textarea id="out"></textarea>
</div>
<script>
const ITEMS = __MANIFEST__;
const EMO = __EMO__;
const KEYMAP = {}; // letter -> emotion
const used = new Set();
for (const e of EMO) { let k = e[0]; let i = 0;
  while (used.has(k) && i < e.length) { k = e[++i] || k + i; }
  used.add(k); KEYMAP[k] = e; }
const STORE = "animelabel_v1";
let state = JSON.parse(localStorage.getItem(STORE) || "{}");
let idx = ITEMS.findIndex(it => !state[it.id]);
if (idx < 0) idx = ITEMS.length;
let pending = null; // emotion chosen, waiting for confidence
let tStart = Date.now();

const $ = id => document.getElementById(id);

function renderButtons() {
  const box = $("emobtns"); box.innerHTML = "";
  for (const [k, e] of Object.entries(KEYMAP)) {
    const b = document.createElement("button");
    b.innerHTML = e + ' <span class="key">' + k.toUpperCase() + '</span>';
    b.onclick = () => choose(e);
    b.dataset.emo = e;
    box.appendChild(b);
  }
}

function draw() {
  if (idx >= ITEMS.length) { finish(); return; }
  const it = ITEMS[idx];
  $("counter").textContent = "第 " + (idx + 1) + " / " + ITEMS.length + " 题";
  $("kindtag").textContent = it.kind === "portrait" ? "（单张立绘）" : "（视频 8 帧）";
  $("barfill").style.width = (100 * idx / ITEMS.length) + "%";
  const f = $("frames"); f.innerHTML = ""; f.classList.remove("zoom");
  for (const fn of it.frames) {
    const im = document.createElement("img");
    im.src = it.id + "/" + fn; im.alt = "";
    im.onclick = () => f.classList.toggle("zoom");
    f.appendChild(im);
  }
  pending = null; tStart = Date.now();
  for (const b of document.querySelectorAll("#emobtns button")) b.classList.remove("sel");
}

function choose(e) {
  pending = e;
  for (const b of document.querySelectorAll("#emobtns button"))
    b.classList.toggle("sel", b.dataset.emo === e);
}

function commit(c) {
  if (!pending) return;
  const it = ITEMS[idx];
  state[it.id] = { emotion: pending, confidence: c, ms: Date.now() - tStart };
  localStorage.setItem(STORE, JSON.stringify(state));
  idx++; draw();
}

function back() {
  if (idx > 0) { idx--; delete state[ITEMS[idx].id];
    localStorage.setItem(STORE, JSON.stringify(state)); draw(); }
}

function finish() {
  $("frames").innerHTML = ""; $("emobtns").innerHTML = ""; $("conf").style.display = "none";
  $("barfill").style.width = "100%"; $("counter").textContent = "";
  $("done").style.display = "block";
  const payload = { protocol: "15-class forced choice, blind, same frames as VLM judges",
                    taxonomy: EMO, n_items: ITEMS.length, labels: state };
  $("out").value = JSON.stringify(payload, null, 1);
  $("dl").onclick = () => {
    payload.annotator = $("who").value || "A";
    const blob = new Blob([JSON.stringify(payload, null, 1)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "labels_" + payload.annotator + ".json";
    a.click();
  };
}

document.addEventListener("keydown", ev => {
  if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
  const k = ev.key.toLowerCase();
  if (k === "z") { back(); return; }
  if (k === "1") { commit(50); return; }
  if (k === "2") { commit(75); return; }
  if (k === "3") { commit(95); return; }
  if (KEYMAP[k]) { choose(KEYMAP[k]); ev.preventDefault(); }
});
for (const b of document.querySelectorAll("#conf button[data-c]"))
  b.onclick = () => commit(parseInt(b.dataset.c, 10));
$("back").onclick = back;
renderButtons(); draw();
</script></body></html>
"""

out = HTML.replace("__MANIFEST__", manifest).replace("__EMO__", emojson)
(HL / "index.html").write_text(out)
print("wrote", HL / "index.html", "items:", len(items))
print("keyboard map:")
used2 = set()
for e in EMO:
    k = e[0]
    i = 0
    while k in used2 and i < len(e):
        i += 1
        k = e[i] if i < len(e) else k + str(i)
    used2.add(k)
    print("  ", k, "->", e)
