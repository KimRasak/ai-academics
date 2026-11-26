# Appendix: Full judge verdicts (Qwen2-VL-7B, do_sample=False)

| Task (target) | base 8s s42 f4 | base 16s s42 f8 | base 16s s1234 f8 | v1 8s s42 f4 | v2 8s s42 f4 | v2 8s s42 f8 | v2 8s s1234 f4 | v2 16s s42 f8 | v2 16s s1234 f8 | v3 16s s42 f8 | v3 16s s1234 f8 | v3 16s s999 f8 | v4 16s s42 f8 | v4 16s s1234 f8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| x01 (joyful) | neutral:50 | Happy:90 | Neutral:50 | Happy:90 | Happy:90 | Happy:90 | neutral:50 | happy:80 | happy:80 | happy:80 | happy:80 | -- | Neutral:50 | Neutral:50 |
| x02 (sad) | neutral:50 | -- | neutral:100 | Neutral:80 | Neutral:80 | -- | Neutral:100 | -- | -- | neutral:100 | **sad:80** | neutral:80 | neutral:50 | **sad:80** |
| x03 (happy) | **Happy:80** | **Happy:90** | -- | **Happy:90** | **Happy:90** | -- | Neutral:100 | -- | -- | **happy:80** | **happy:90** | -- | neutral:100 | Neutral:50 |
| x04 (worried) | neutral:50 | -- | -- | Neutral:80 | Neutral:80 | -- | Neutral:100 | -- | -- | neutral:50 | neutral:100 | -- | neutral:50 | neutral:100 |
| x05 (surprised) | neutral:100 | -- | -- | Neutral:100 | **surprised:80** | Neutral:100 | Neutral:100 | Neutral:100 | -- | neutral:100 | happy:80 | -- | Neutral:100 | Neutral:50 |
| x06 (relieved) | neutral:50 | -- | -- | sad:80 | sad:80 | sad:80 | sad:100 | -- | -- | neutral:50 | happy:80 | -- | Neutral:50 | neutral:100 |
| x07 (cheerful) | neutral:50 | -- | neutral:50 | happy:80 | happy:80 | happy:80 | Neutral:50 | happy:80 | neutral:50 | happy:80 | happy:80 | Neutral:50 | neutral:50 | happy:80 |
| x08 (angry) | neutral:100 | -- | -- | Neutral:100 | Neutral:100 | -- | Neutral:100 | -- | -- | neutral:100 | neutral:100 | -- | neutral:50 | neutral:50 |
