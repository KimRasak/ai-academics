# Round 36：Sci-VBench judge endpoint feasibility

## Findings

- The H3 FL2VA and Ref2VA ports 30011/30020 expose the generation pipeline only. Their `/v1/models` identify MiniMax-H3 and are not multimodal judge endpoints.
- Probes of likely local ports 30001, 8000, 8001 and 8080 on host 034 returned connection refused.
- Remote storage does contain the existing `Qwen2-VL-7B-Instruct` checkpoint at `/gluster_hdd_cv/user/jinzili/weights/reward_models/Qwen2-VL-7B-Instruct` (~16 GB), and historical project scripts show it was used for a separate emotion judge. This is not the Sci-VBench released Qwen3.5 judge and must not be substituted silently.
- Existing `sglang310`/candidate environment transformer import was too slow for a bounded probe and no live judge process/endpoint was found. No judge request was sent.

## Decision

Judge Gate remains blocked by endpoint/model compatibility, not by H3 generation. Do not use the Qwen2 emotion judge as Sci-VBench judge. The fresh VT evaluation is complete and valid under its stated implementation; judge dimensions remain unreported.
