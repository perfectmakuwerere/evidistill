# EviDistill

**Evidence-weighted activation distillation from large to small language models for long-context evidence use.**

Capstone project, M.S. Applied AI, Stevens Institute of Technology
Author: Perfect-Princess Makuwerere (pmakuwer@stevens.edu)
Advisors: CongCong Xu and Min Song
Lab: IntelliSyS Lab

---

## Overview

Small language models lose much more accuracy than large ones when the evidence needed to answer a question is buried in a long input among distractors. Knowledge distillation can transfer a teacher's behavior to a student, but current methods either copy only the final output or match internal hidden states with **every token weighted equally**. In long inputs, most tokens are filler or distractors, so uniform supervision mostly teaches the student how the teacher handles noise.

**EviDistill** concentrates activation supervision where the teacher actually depends on the evidence:

1. **Counterfactual evidence discovery.** Run the teacher on the original input, on the input with the evidence removed, and on the input with a misleading passage. Positions and layers whose internal states change the most are the ones that depend on the evidence.
2. **Evidence weights.** Combine the teacher's question-to-context attention with the size of its counterfactual reaction into per-position, per-layer weights (with a floor `r_min`, attention sinks excluded, stop-gradient).
3. **Weighted activation distillation.** Match projected student hidden states to teacher hidden states at selected layers, over context, question and answer positions, weighted by the evidence weights.
4. **Reaction supervision.** Train the student to change its internal state the way the teacher does when the evidence is removed.

Objective:

```
L = L_answer + λ_KD · L_KD + λ_act · L_act + λ_react · L_react
```

## Setup

| | |
|---|---|
| Teacher | Qwen3-8B |
| Student | Qwen3-1.7B (main), Qwen3-0.6B (capacity check) |
| Training data | Synthetic needle tasks, NQ multi-document QA, HotpotQA (distractor) |
| Evaluation | RULER (4K/8K), LongBench QA (≤8K), accuracy by input length and evidence position |
| Codebase | [DSKD](https://github.com/songmzhang/DSKD) (Zhang et al., EMNLP 2024) |

## Baselines

| ID | Method |
|---|---|
| B0 | Student fine-tuning only (SFT) |
| B1 | Output-level KD (vanilla KD) |
| B2 | DSKD (Zhang et al., EMNLP 2024), run with minimal documented changes |
| B3 | Uniform activation KD (same loss as ours, all weights = 1) |
| B4 | Alternative token weightings (teacher uncertainty, representation salience) |
| **Ours** | **EviDistill (evidence-weighted activation KD + reaction supervision)** |

Ablations: answer-only vs. context positions, no `L_react`, attention-only weights, reaction-only weights, each counterfactual alone.

## Repository layout

```
evidistill/
├── dskd/            # DSKD codebase (baselines + our loss in code/criterions/)
├── diagnosis/       # diagnosis notebook (retrieval vs. answer gap by length and depth)
├── data_prep/       # builds long-context datasets and counterfactual inputs
├── scripts/         # run scripts for each method
└── README.md
```

## Status

- [x] Diagnosis notebook (Qwen3-8B vs. Qwen3-1.7B, synthetic needle task)
- [x] DSKD environment set up on lab server (pentek5)
- [ ] DSKD reproduction, GPT-2 1.5B → GPT-2 120M on Dolly (in progress)
- [ ] Qwen3 pilot (Qwen3-1.7B → Qwen3-0.6B, short inputs)
- [ ] Long-context datasets and counterfactual inputs
- [ ] Evidence weight precomputation and sanity check
- [ ] EviDistill loss in `dskd/code/criterions/`
- [ ] Main experiments (Qwen3-8B → Qwen3-1.7B, up to 8K tokens)

## Environment setup

Tested on 8 × RTX 2080 Ti (11 GB), driver CUDA 13.2, no job scheduler.

```bash
# 1. Miniconda in home folder (no admin rights needed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash && source ~/.bashrc

# 2. Environment
conda create -n dskd python=3.10 -y
conda activate dskd
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "transformers==4.44.2" "peft==0.12.0" "deepspeed>=0.14,<0.17" \
            rouge_score gdown huggingface_hub editdistance tqdm numpy

# 3. CUDA compiler for DeepSpeed (fixes "CUDA_HOME does not exist")
conda install -c nvidia/label/cuda-12.1.0 cuda-nvcc -y
conda env config vars set CUDA_HOME=$CONDA_PREFIX
conda deactivate && conda activate dskd

# 4. Check
python -c "import torch, deepspeed; print(torch.cuda.is_available(), torch.cuda.device_count(), deepspeed.__version__)"
```

## Reproducing DSKD (GPT-2 setting)

```bash
cd dskd
export DSKD=$(pwd)

# Data (authors' processed Dolly data); must end up as data/dolly/{train,dev}.jsonl
gdown --folder https://drive.google.com/drive/folders/1ZUsNVgWevACV9D-AHVNi9C7PX_2itzb8 -O data

# Models
huggingface-cli download openai-community/gpt2 model.safetensors --local-dir model_hub/gpt2/gpt2-base
huggingface-cli download MiniLLM/teacher-gpt2-1.5B --local-dir model_hub/gpt2/teacher-gpt2-1.5B

# Paths, fp16, and memory settings
for f in scripts/gpt2/*_gpt2_base.sh scripts/eval/run_eval.sh; do
  sed -i "s#path_to_dskd_project#$DSKD#; s#path_to_teacher_sft_ckpt#$DSKD/model_hub/gpt2/teacher-gpt2-1.5B#" $f
done
sed -i 's/^PRECISION="bf16"/PRECISION="fp16"/' scripts/gpt2/*_gpt2_base.sh
sed -i 's/ds_config_bf16.json/ds_config.json/' scripts/eval/eval_main.sh
sed -i 's/^BATCH_SIZE=4/BATCH_SIZE=2/; s/^GRAD_ACC=2/GRAD_ACC=4/; s/^EVAL_BATCH_SIZE=32/EVAL_BATCH_SIZE=8/' scripts/gpt2/*_gpt2_base.sh

# Train (runs in background; survives logout)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup bash scripts/gpt2/dskd_gpt2_base.sh > /dev/null 2>&1 &
tail -f outputs/gpt2/gpt2-base/dual_space_kd/*/train.log

# Evaluate (absolute path to the saved epoch* checkpoint)
bash scripts/eval/run_eval.sh /abs/path/to/epochX_stepY 8
```

### Changes from the original DSKD configuration

| Change | Reason | Effect on method |
|---|---|---|
| fp16 instead of bf16 | RTX 2080 Ti has no bf16 support | None intended |
| Per-GPU batch 2 × accumulation 4 (was 4 × 2) | 11 GB GPU memory | None: effective batch stays 32 |
| Evaluation batch 8 (was 32) | Memory during generation | None: speed only |
| `expandable_segments:True` | Reduces memory fragmentation | None |

### Reproduction results (Rouge-L, GPT-2 1.5B → 120M)

| Method | Dolly (paper) | Dolly (ours) | Avg 5 sets (paper) | Avg 5 sets (ours) |
|---|---|---|---|---|
| SFT | 22.94 | TBD | 16.62 | TBD |
| Vanilla KD (KL) | 24.54 | TBD | 17.63 | TBD |
| DSKD (KL) | 24.70 | TBD | 18.69 | TBD |

## Notes for running on a shared server

- Check free GPUs with `nvidia-smi` before starting; choose GPUs with the `GPUS=(...)` line in each script.
- Stop your jobs with `pkill -f distillation.py`. Finished jobs release GPUs automatically.

## Acknowledgments

This project builds on the official DSKD implementation:

```bibtex
@inproceedings{zhang2024dskd,
  title     = {Dual-Space Knowledge Distillation for Large Language Models},
  author    = {Zhang, Songming and Zhang, Xue and Sun, Zengkui and Chen, Yufeng and Xu, Jinan},
  booktitle = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  year      = {2024}
}
```

The GPT-2 teacher checkpoint is from MiniLLM (Gu et al., ICLR 2024).
