# Direct Soft-Policy Sampling via Langevin Dynamics

<p align="center">
<font size=5>📑</font>

## Installation

```bash
# Create environemnt
conda create -n nclql python=3.11 numpy tqdm tensorboardX matplotlib scikit-learn black snakeviz ipykernel setproctitle numba pyyaml
conda activate nclql

pip install -r requirements.txt
# One of: Install jax WITH CUDA 
pip install --upgrade "jax[cuda12]==0.4.27" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install -e .
```

## Run
```bash
# Run one experiment
XLA_FLAGS='--xla_gpu_deterministic_ops=true' CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python scripts/train_mujoco.py
```
