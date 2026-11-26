# Round 3：官方 H3-Base 三案例复现

日期：2026-09-07

## 服务

在 `osa00-h100-034` 上通过 gpu_launcher 启动的无 LoRA official Base：

- FL2VA：GPUs 0–3，TP=2，Ulysses=2，port 30011
- Ref2VA：GPUs 4–7，TP=2，Ulysses=2，port 30020
- model path：`/gluster_hdd_cv/user/jinzili/weights/MiniMax-H3`
- `lora_path=null`
- 两个服务运行期间和任务完成后 `/health` 都返回 `{"status":"ok"}`

## 结果

| Case | Variant/port | Video ID | Polls | Output bytes | Duration | Video | Audio |
|---|---|---|---:|---:|---:|---|---|
| T2VA | FL2VA / 30011 | `fc53e5db-7cf1-4274-8526-47af94e658c1` | 45 | 1,659,821 | 10.125 s | H.264, 1344×768, 24 FPS | AAC, 32 kHz, 2 ch |
| FL2VA | FL2VA / 30011 | `0b86c4c6-a023-4db8-a00a-b717bd5a9c07` | 33 | 1,071,181 | 8.000 s | H.264, 1344×768, 24 FPS | AAC, 32 kHz, 2 ch |
| Ref2VA | Ref2VA / 30020 | `c0463989-85d2-448c-9327-895fa4e95c35` | 54 | 851,078 | 5.175 s | H.264, 1344×768, 24 FPS | AAC, 32 kHz, 2 ch |

所有任务 `final_status.status=completed`。runner 持续轮询至 terminal status，再下载 MP4；不是一次查询后的伪成功。

## 产物

`official_requests/{t2va,fl2va,ref2va}/` 中各保存：

- `request.json`
- `submit_response.json`
- `status_history.json`
- `final_status.json`
- `runner_summary.json`
- `ffprobe.json`
- `output.mp4`
- `service.log`（FL2VA 与 Ref2VA）

三条输出的 SHA-256：

- T2VA：`a5970e101c70f37b54a5b19ede779c624ee5255c5faea4fd285463e9065d8f2f`
- FL2VA：`1c95d37a10500506e27baa4dd62a85d81badc43b9f9dca728e4bfe19415805bc`
- Ref2VA：`1f8f61f48a456eca75d0abbc78cc92ed2d759f0b2f5f2e64ef5621d24d17b3f9`

## 服务日志证据

FL2VA 日志显示完整 49/49 denoise、decode 和 output save；该请求记录 `Pixel data generated successfully in 264.22 seconds`，峰值显存约 59,348 MB。Ref2VA 日志显示完整 49/49 denoise、decode 和 output save；该请求记录约 216.76 s、峰值约 59,478 MB。T2VA 同一 FL2VA 服务也完成并已归档状态与输出；三者均未触发 OOM 或 traceback。

## 结论边界

这完成了官方三条 768p **运行链路**复现：请求完成、媒体规格符合官方 envelope、输出可解码。它不是逐像素复现，也没有证明 Context-IR 质量、感知质量或官方 benchmark 分数完全复现。下一步必须在冻结的 Sci-VBench pilot 上按原始 prompt 生成并评测，不能把这三个 demo 当成论文结果。
