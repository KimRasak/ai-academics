# Outputs

No local reproduction output has been generated yet.

When GPUs become available, store each case under:

- `t2va/`;
- `fl2va/`;
- `ref2va/`.

Each case must include the generated MP4, server log, request response, media
metadata and comparison report. Do not copy the official reference MP4 into a
generated-output directory.

The service-level log and launch metadata are retained outside an individual
case, under `../server_logs/VARIANT_TIMESTAMP/`. Start it only through
`../commands/launch_h3_base_service.sh VARIANT GPU0 GPU1 GPU2 GPU3`, after the
ownership gate has passed and a human has confirmed the GPUs are available.
After an operator stops it, retain the output of
`../tools/audit_h3_service_turnover.sh VARIANT GPU0 GPU1 GPU2 GPU3` alongside
the server log before reusing the GPU set.

When a Base service is ownership-cleared and running, use
`../commands/run_frozen_official_case.sh CASE`. It refuses to overwrite an
existing case directory, hashes a copied official request script and generated
MP4, and uses the durable polling runner to retain submit/status/final evidence.
Capture the server log separately before stopping the service.

It then writes `postrun_audit.json`, which checks frozen-payload identity,
terminal status and the 768p/24fps/32-kHz-stereo media contract. It explicitly
does not claim perceptual equivalence to the reference MP4 or condition-following
quality, because no official/pre-registered threshold exists for that purpose.
