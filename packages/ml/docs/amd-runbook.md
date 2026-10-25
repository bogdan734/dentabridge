# Training on an AMD Developer Cloud MI300X

The recipe runs on any ROCm PyTorch; these are the steps used for the published runs. The
cloud is DigitalOcean's: a droplet bills until it is **destroyed**, powered off or not.

1. Create a GPU Droplet: 1x MI300X (`gpu-mi300x1-192gb`: 20 vCPU, 240 GiB RAM, 720 GB boot
   disk), image Quick Start (ROCm + PyTorch, Ubuntu 24.04), your SSH public key.
   Credits expire 30 days after they are deposited unless the confirmation email says otherwise.
2. From the repository root on your machine: `AMD_SSH_KEY=~/.ssh/<key> scripts/amd/sync.sh root@<ip>`.
3. On the droplet, in `~/dentabridge`: `scripts/amd/bootstrap.sh`, then `scripts/amd/pilot.sh`.
   The PyTorch wheel is `2.9.1+rocm6.4`, matching the image's host driver (amdgpu-dkms 6.12,
   ROCm 6.4); a newer ROCm user space on an older kernel driver is not assumed to work.
4. The pilot (`configs/train/pilot-50.toml`: 50 steps) prints `run.json` with the throughput.
   Real runs: `python -m dentabridge_ml.train sft train --out var/train.jsonl`, then
   `python -m dentabridge_ml.train run packages/ml/configs/train/mi300x-base.toml var/train.jsonl --out var/runs/<name>`.
   Evaluate with `python -m dentabridge_ml.eval run --backend transformers:<merged dir> --split val --batch-size 16 --image-max-px 448 --out-dir var/eval/<name>`.
5. `scripts/amd/fetch.sh root@<ip>` copies adapters, `run.json` files and results to `var/amd/`.
6. Destroy the droplet in the console. Scratch disk contents do not survive.

One change per ablation run, `run.json` records the config digest, the commit, the hardware and
the throughput; the cost is hours times the console price.
