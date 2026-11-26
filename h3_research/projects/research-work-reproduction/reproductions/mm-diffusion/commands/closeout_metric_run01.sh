#!/usr/bin/env bash
set -euo pipefail

# Conservative closeout watcher for the active MM-Diffusion metric run. It
# never resumes/restarts sampling. Evaluation runs only after an exact complete
# media inventory and successful full audit.
MMROOT=/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/mm-diffusion
RUN_ID=ddpm_ddim25_metric_run01_seed42_preimport_gpu0124_20260918
RUN_ROOT="$MMROOT/outputs/$RUN_ID"
LOG_DIR="$MMROOT/logs"
AUDIT_DIR="$LOG_DIR/$RUN_ID.media_audit"
STATE="$LOG_DIR/$RUN_ID.closeout_status.txt"
# Match only the sampler command.  Matching RUN_ID alone also matches this
# watcher script's own command line, which would make it wait forever after
# the sampler exits.
SAMPLER_PATTERN="multimodal_sample_sr.py.*$RUN_ID"

while pgrep -f "$SAMPLER_PATTERN" >/dev/null; do
  sleep 60
done

original_count=$(find "$RUN_ROOT/landscape.pt/original" -type f -name '*.mp4' 2>/dev/null | wc -l)
sr_count=$(find "$RUN_ROOT/landscape.pt/sr_mp4" -type f -name '*.mp4' 2>/dev/null | wc -l)
wav_count=$(find "$RUN_ROOT/landscape.pt/audios" -type f -name '*.wav' 2>/dev/null | wc -l)
printf 'sampler_terminal\noriginal=%s\nsr=%s\nwav=%s\n' "$original_count" "$sr_count" "$wav_count" > "$STATE"

if [[ "$original_count" != 2048 || "$sr_count" != 2048 || "$wav_count" != 2048 ]]; then
  printf 'closeout=invalid_incomplete_media_no_evaluation\n' >> "$STATE"
  exit 2
fi

"$MMROOT/tools/audit_metric_run_media.sh" "$RUN_ROOT" "$AUDIT_DIR"
if [[ $(sed -n 's/^ffprobe_failure_lines=//p' "$AUDIT_DIR/summary.txt") != 0 ]]; then
  printf 'closeout=invalid_decode_failure_no_evaluation\n' >> "$STATE"
  exit 3
fi

printf 'audit=passed\n' >> "$STATE"
"$MMROOT/commands/evaluate_metric_run_separately.sh" original \
  "${RUN_ID}_eval_original" 5 > "$LOG_DIR/${RUN_ID}_eval_original.log" 2>&1
"$MMROOT/commands/evaluate_metric_run_separately.sh" sr_mp4 \
  "${RUN_ID}_eval_sr_mp4" 5 > "$LOG_DIR/${RUN_ID}_eval_sr_mp4.log" 2>&1
printf 'closeout=audited_and_both_independent_evaluations_terminal\n' >> "$STATE"
