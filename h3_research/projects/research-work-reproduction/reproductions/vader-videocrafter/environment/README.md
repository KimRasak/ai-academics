# Environment

Remote environment:

`/gluster_hdd_cv/user/jinzili/env/vader-videocrafter`

Target official core:

- Python 3.10
- PyTorch 2.3.0+cu121
- torchvision 0.18.0+cu121
- torchaudio 2.3.0+cu121
- CUDA runtime 12.1
- VADER requirements pinned by the repository
- xformers 0.0.26.post1

Documented deviation:

- the repository leaves `gradio` unpinned; it is excluded because it is not
  needed for offline training/inference and causes modern dependency-resolution
  backtracking against the paper-era Transformers/Hugging Face versions.
- the repository imports `importlib_resources` but does not declare it in the
  environment path exercised here. `importlib-resources==6.4.0` was added after
  the official entry point failed with `ModuleNotFoundError`.

Verification completed on 2026-09-16:

- CUDA available: yes; runtime reported by PyTorch: 12.1;
- `train_t2v_lora.py --help`: exit code 0 after the documented dependency fix;
- `decord==0.6.0`: imports and decodes successfully despite `pip check`
  reporting that its wheel metadata is unsupported on this platform;
- both released LoRA files load on CPU, each with 396 tensors;
- PickScore LoRA rank: 16; HPS+Aesthetic LoRA rank: 8;
- final `pip freeze`:
  `/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/vader/environment/pip-freeze.txt`;
- `pip freeze` SHA256:
  `f329668f1a7c5b47399444c127a2310ccc1e22bfaaf5cf22940802a38feb72ca`.

VBench temporal diagnostics use a separate environment:

`/gluster_hdd_cv/user/jinzili/env/vbench-temporal`

- Python 3.10.12;
- PyTorch 2.3.0+cu121 and torchvision 0.18.0+cu121;
- NumPy 1.24.2, SciPy 1.15.3 and OpenCV 4.11.0;
- 34 packages are frozen; `uv pip check` reports Decord wheel
  metadata/platform incompatibility, while Decord imports and the completed
  evaluator runs succeed;
- freeze file:
  `/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/vader/environment/vbench-temporal-pip-freeze.txt`;
- freeze SHA256:
  `ba802f2aeb7fcad6bc72627479075e4c2a36c25f423e3bb14504163e8034c147`.

An earlier attempt to call `pip freeze` produced an empty file because uv
environments do not include pip by default. It was immediately overwritten
using `uv pip freeze --python`; the empty hash is not an accepted artifact.
