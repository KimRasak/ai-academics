# Environment

Remote environment:

`/gluster_hdd_cv/user/jinzili/env/t2v-turbo-vc2`

Key versions: Python 3.10.12, PyTorch 2.3.0+cu121, torchvision 0.18.0+cu121,
diffusers 0.30.0, transformers 4.44.0, accelerate 0.29.3, xFormers
0.0.26.post1, NumPy 1.26.4 and PyAV 12.0.0.

The complete freeze is retained with the successful smoke at
`current-code-release-protocol-smoke-v2/environment.freeze.txt`, SHA256
`13ccd08887e155cdac66c2aecf5d66b076ed6d0a74456b2669155839d33c8050`.

PyAV 17.1.0 was rejected because it is incompatible with torchvision 0.18's
`write_video` API.

Paper-day VBench evaluation environment:

`/gluster_hdd_cv/user/jinzili/env/vbench-paper-20240529`

- Python 3.10.12, PyTorch 2.3.0+cu121, torchvision 0.18.0+cu121;
- transformers 4.33.2, timm 0.9.16, pyiqa 0.1.8, Pillow 12.3.0;
- setuptools 69.5.1 for the legacy `pkg_resources` dependency;
- 105-line freeze at
  `data/research-work-reproduction/t2v-turbo/environment/vbench-paper-20240529/environment.freeze.txt`;
- freeze SHA256
  `cc3941a085b50a54634c090042aee27f9de1e60e0ef8ed6ac4eefd10ae66d1ba`.

GRIT dimensions use the separate environment:

`/gluster_hdd_cv/user/jinzili/env/vbench-grit-paper-20240529`

- Python 3.11.9, PyTorch 2.3.0+cu121, torchvision 0.18.0+cu121;
- Detectron2 0.6 at commit
  `d1e04565d3bec8719335b88be9e9b961bf3ec464`;
- Pillow 9.5.0 because Detectron2 0.6 uses the removed `Image.LINEAR` alias;
- boto3 1.43.96, an undeclared GRIT runtime dependency;
- 97-line freeze at
  `data/research-work-reproduction/t2v-turbo/environment/vbench-grit-paper-20240529/environment.freeze.txt`;
- freeze SHA256
  `5e469763740e562a22ab2f3c1f68a20fe05b7de389ea9edd3ae430d4eea5cb00`.
