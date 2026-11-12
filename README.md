# Awesome AI Training [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> A curated list of **LLM/SLM pretraining codebases, from-scratch training recipes, and pretraining frameworks**.
>
> If you can't run a training loop on raw tokens with it, it doesn't belong here.

Most "open source AI" lists are catalogs of weight drops with an inference snippet attached. This one is not. Every entry below either **trains a language model from scratch** or is **infrastructure for doing so**. No inference-only repos. No finetuning wrappers. No RL/post-training stacks. No eval harnesses, agent frameworks, or CLIs.

**Snapshot:** 2026-11-13. Links checked against live repositories at that date. Repo activity changes weekly — PRs welcome for corrections.

---

## Scope

**Included**

- From-scratch pretraining codebases (model definition + training loop + configs)
- Pretraining frameworks and "cores" (no weights of their own)
- Architecture-specific training stacks (MoE, linear attention, SSM, quantized)
- Decentralized / low-communication pretraining systems
- Data pipelines that are the published half of a pretraining recipe

**Excluded**

- Weights-only or inference-only releases, regardless of license
- Finetuning / SFT / RLHF / RLVR / post-training tooling
- Serving engines, eval harnesses, agent frameworks, CLIs, SDKs
- Architecture-only implementations with no training path (unless explicitly paired with a listed trainer)

A permissive license is **not** an inclusion criterion. Public training code is.

---

## Legend

| Marker | Meaning |
|---|---|
| ★ | Underrated — high technical merit, low visibility |
| 🟢 | Fully open: training code + weights + data/recipe public |
| 🔵 | Train-it-yourself: full pipeline published, you produce the weights |
| ⚙️ | Framework / core: no weights of its own |
| ⚠️ | Architecture or kernels only — bring your own trainer from the frameworks section |

---

## Contents

- [Where to Start](#where-to-start)
- [Nano and Educational Trainers](#nano-and-educational-trainers)
- [Small and Mid-Sized Open Model Codebases](#small-and-mid-sized-open-model-codebases)
- [Large and Historical Trainers](#large-and-historical-trainers)
- [Pretraining Frameworks](#pretraining-frameworks)
- [Alternative Architecture Stacks](#alternative-architecture-stacks)
- [Decentralized and Low-Communication Training](#decentralized-and-low-communication-training)
- [Data Pipeline Companions](#data-pipeline-companions)
- [Borderline Cases](#borderline-cases)
- [Excluded by Scope](#excluded-by-scope)
- [License Caveats](#license-caveats)
- [Contributing](#contributing)
- [Maintenance](#maintenance)
- [License](#license)

---

## Where to Start

| Your hardware / goal | Start here |
|---|---|
| One consumer GPU, learn the loop | [nanoGPT](https://github.com/karpathy/nanoGPT), [MiniMind](https://github.com/jingyaogong/minimind), [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) |
| 8×H100, want a full chat model fast | [nanochat](https://github.com/karpathy/nanochat), [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) |
| Maximum reproducibility (code + data + recipe) | [OLMo-core](https://github.com/allenai/OLMo-core), [Apertus](https://github.com/swiss-ai/pretrain-code), [LLM360](https://github.com/LLM360/k2v2_train) |
| Production cluster, 4D parallelism | [torchtitan](https://github.com/pytorch/torchtitan), [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) |
| Non-transformer architectures | [flame](https://github.com/fla-org/flame), [RWKV-LM](https://github.com/BlinkDL/RWKV-LM), [zcookbook](https://github.com/Zyphra/zcookbook) |
| No datacenter, commodity/internet hardware | [OpenDiLoCo](https://github.com/PrimeIntellect-ai/OpenDiLoCo), [templar](https://github.com/one-covenant/templar) |

---

## Nano and Educational Trainers

Single GPU through 8×H100. Full pipelines, readable code. 🔵

| Project | Org | Size | Training entry point | Source |
|---|---|---|---|---|
| nanoGPT | Karpathy | GPT-2 124M–1.6B | `train.py` | [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) |
| build-nanogpt | Karpathy | GPT-2 | `train.py` (video companion) | [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) |
| minGPT | Karpathy | educational | `mingpt/trainer.py` + projects | [karpathy/minGPT](https://github.com/karpathy/minGPT) |
| llm.c | Karpathy | GPT-2 ≤1.6B | `train_gpt2.cu` / `train_gpt2.py` (C/CUDA) | [karpathy/llm.c](https://github.com/karpathy/llm.c) |
| llama2.c | Karpathy | 260K–15M | `train.py` (TinyStories) | [karpathy/llama2.c](https://github.com/karpathy/llama2.c) |
| nanochat | Karpathy | ~0.15B–1.75B | `speedrun.sh` — tokenize → pretrain → SFT → eval | [karpathy/nanochat](https://github.com/karpathy/nanochat) |
| modded-nanogpt ★ | Keller Jordan | GPT-2 124M | `train_gpt.py` (speedrun record) | [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) |
| LLMs-from-scratch ★ | rasbt | small GPT | pretraining + finetuning, book companion code | [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) |
| MiniMind (+V/O, minimind-3) | jingyaogong | 26M–198M-A64M | `trainer/train_pretrain.py`; minimind-3 Qwen3-aligned | [jingyaogong/minimind](https://github.com/jingyaogong/minimind) |
| TinyStories | roneneldan / MSR | 1M–33M | training code + synthetic dataset | [roneneldan/TinyStories](https://github.com/roneneldan/TinyStories) |
| SimpleStories ★ | simple-stories | tiny LMs | `simple_stories_train`, one A100, based on `train_gpt2.py` | [simple-stories/simple_stories_train](https://github.com/simple-stories/simple_stories_train) |
| femtoGPT ★ | keyvank | tiny GPT | pure-Rust training + inference | [keyvank/femtoGPT](https://github.com/keyvank/femtoGPT) |
| Parameter Golf | OpenAI | ≤16 MB artifact | `train_gpt.py`; <10 min on 8×H100, FineWeb, weights+code counted | [openai/parameter-golf](https://github.com/openai/parameter-golf) |
| nanowhale | Hugging Face | ~110M MoE (DeepSeek-V4 arch) | full from-scratch code, configs, tokenizer | [huggingface/nanowhale](https://github.com/huggingface/nanowhale) |
| picotron | Hugging Face | educational | minimal 4D-parallelism trainer | [huggingface/picotron](https://github.com/huggingface/picotron) |

---

## Small and Mid-Sized Open Model Codebases

Published pretraining pipelines, generally with data mixtures and checkpoints. 🟢

| Project | Org | Size | Training entry point | Source |
|---|---|---|---|---|
| SmolLM / SmolLM2 / SmolLM3 | Hugging Face | 135M–3B | `text/pretraining` (nanotron), configs + data mixtures | [huggingface/smollm](https://github.com/huggingface/smollm) |
| TinyLlama | jzhang38 | 1.1B / 3T tokens | pretrain scripts (lit-gpt based) | [jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama) |
| OpenLLaMA | OpenLM / Berkeley | 3B / 7B / 13B | EasyLM training configs | [openlm-research/open_llama](https://github.com/openlm-research/open_llama) |
| OpenELM ★ | Apple | 270M–3B | CoreNet training recipes | [apple/corenet](https://github.com/apple/corenet) |
| MobileLLM ★ | Meta | 125M–1B | training code + dataset prep + multi-node setup | [facebookresearch/MobileLLM](https://github.com/facebookresearch/MobileLLM) |
| MobileLLM-R1 ★ | Meta | 950M | reproduces pretraining, mid-training and SFT stages + data sources | [facebookresearch/MobileLLM-R1](https://github.com/facebookresearch/MobileLLM-R1) |
| Cerebras-GPT | Cerebras | 111M–13B | ModelZoo training configs | [Cerebras/modelzoo](https://github.com/Cerebras/modelzoo) |
| Pythia | EleutherAI | 70M–12B | GPT-NeoX configs, 154 intermediate checkpoints | [EleutherAI/pythia](https://github.com/EleutherAI/pythia) |
| Pleias / nanoPleias ★ | PleIAs | 350M–1B | nanotron fork used for pretraining | [Pleias/nanotron-pleias](https://github.com/Pleias/nanotron-pleias) |
| RWKV-LM (v4–v8) | BlinkDL | 0.1B–14B | per-version trainers; `RWKV-v8/` experimental | [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) |
| HRM-Text ★ | Sapient | 1B | full pretraining pipeline; ~40B tokens, 1.9 days on 16 GPUs | [sapientinc/HRM-Text](https://github.com/sapientinc/HRM-Text) |
| HRM ★ | Sapient | 27M | hierarchical reasoning trainer, 1000-sample regime | [sapientinc/HRM](https://github.com/sapientinc/HRM) |
| OLMo / OLMo 2 / OLMo 3 / OLMo Hybrid | Ai2 | 1B–32B; Hybrid 7B | `src/scripts/official/` (Hybrid = Gated DeltaNet + attention, 3:1) | [allenai/OLMo-core](https://github.com/allenai/OLMo-core) · [allenai/OLMo](https://github.com/allenai/OLMo) |
| OLMoE | Ai2 | 7B-A1B | MoE training code | [allenai/OLMoE](https://github.com/allenai/OLMoE) |
| Amber / Crystal | LLM360 | 7B | full code + data + all intermediate checkpoints | [LLM360/Amber](https://github.com/LLM360/Amber) · [LLM360/Crystal](https://github.com/LLM360/Crystal) |
| K2-65B / K2-V2 | LLM360 | 65B / 70B | `k2-train`, `k2v2_train` (+ `k2-data-prep`) | [LLM360/k2-train](https://github.com/LLM360/k2-train) · [LLM360/k2v2_train](https://github.com/LLM360/k2v2_train) |
| DCLM ★ | Apple / UW | 7B | `open_lm` trainer + open data pool | [mlfoundations/dclm](https://github.com/mlfoundations/dclm) |
| MAP-Neo | M-A-P | 7B bilingual | training code + Matrix data pipeline | [multimodal-art-projection/MAP-NEO](https://github.com/multimodal-art-projection/MAP-NEO) |
| Apertus | Swiss AI (EPFL/ETH/CSCS) | 8B / 70B | `pretrain-code` (Megatron-LM based), 1000+ languages | [swiss-ai/pretrain-code](https://github.com/swiss-ai/pretrain-code) |
| Marin ★ | Stanford / Open Athena | 8B | open lab notebook; every experiment as a PR (Levanter) | [marin-community/marin](https://github.com/marin-community/marin) |
| OpenSeek ★ | BAAI | 1.4B-A0.4B and up | community DeepSeek reproduction; OpenSeek-Small trained on 720B tokens | [FlagAI-Open/OpenSeek](https://github.com/FlagAI-Open/OpenSeek) |
| Poro 2 ★ | Silo AI / LUMI | 8B–34B | custom Megatron-LM fork used for the runs | [LumiOpen/Megatron-LM-lumi](https://github.com/LumiOpen/Megatron-LM-lumi) |
| JetMoE ★ | MyShell / MIT | 8B-A2B | open training code, public datasets, ~$0.1M total cost | [myshell-ai/JetMoE](https://github.com/myshell-ai/JetMoE) |
| OpenMoE ★ | XueFuzhao | 0.5B–8B MoE | full MoE pretraining stack | [XueFuzhao/OpenMoE](https://github.com/XueFuzhao/OpenMoE) |
| TransNormerLLM ★ | OpenNLPLab | 1B–15B | linear-attention LLM implementation + training scripts | [OpenNLPLab/TransnormerLLM](https://github.com/OpenNLPLab/TransnormerLLM) |
| GatedDeltaNet / GatedDeltaNet-2 ★ | NVlabs | 1.3B | "train your own" on SlimPajama / FineWeb-Edu | [NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet) · [NVlabs/GatedDeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2) |

---

## Large and Historical Trainers

Older, but the training code still runs and the recipes are documented. 🟢

| Project | Org | Size | Training entry point | Source |
|---|---|---|---|---|
| GPT-J ★ | EleutherAI / kingoflolz | 6B | mesh-transformer-jax (JAX/TPU) | [kingoflolz/mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax) |
| GPT-Neo | EleutherAI | 1.3B / 2.7B | `train.py` | [EleutherAI/gpt-neo](https://github.com/EleutherAI/gpt-neo) |
| GPT-NeoX-20B | EleutherAI | 20B | `train.py` (also a general framework) | [EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox) |
| OPT ★ | Meta | 125M–175B | metaseq + the full public training logbook | [facebookresearch/metaseq](https://github.com/facebookresearch/metaseq) |
| BLOOM | BigScience | 560M–176B | Megatron-DeepSpeed + BigScience train configs | [bigscience-workshop/Megatron-DeepSpeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed) |
| YaLM-100B | Yandex | 100B | training code shipped with the release | [yandex/YaLM-100B](https://github.com/yandex/YaLM-100B) |
| GLM-130B ★ | THUDM | 130B | `pretrain_glm.py` | [THUDM/GLM-130B](https://github.com/THUDM/GLM-130B) |
| Yuan-1.0 / Yuan-2.0 / Yuan2.0-M32 ★ | IEIT | 2B–102B | `docs/pretrain.md`; M32 MoE trained from scratch | [IEIT-Yuan/Yuan-2.0](https://github.com/IEIT-Yuan/Yuan-2.0) · [IEIT-Yuan/Yuan2.0-M32](https://github.com/ieit-yuan/yuan2.0-m32) |
| CPM-Live ★ | OpenBMB | 7B+ | live training code, logs and checkpoints | [OpenBMB/CPM-Live](https://github.com/OpenBMB/CPM-Live) |
| TigerBot | TigerResearch | 7B–180B | `train/` scripts; pretraining mixture documented in tech report | [TigerResearch/TigerBot](https://github.com/TigerResearch/TigerBot) |
| MPT | MosaicML | 7B / 30B | llm-foundry pretraining YAMLs | [mosaicml/llm-foundry](https://github.com/mosaicml/llm-foundry) |
| RedPajama-INCITE | Together | 3B / 7B | GPT-NeoX + published data recipe | [togethercomputer/RedPajama-Data](https://github.com/togethercomputer/RedPajama-Data) |

---

## Pretraining Frameworks

The "cores". No weights of their own — these are what the model repos above are built on. ⚙️

| Framework | Org | What it trains | Source |
|---|---|---|---|
| Megatron-LM (+ Megatron-Core) | NVIDIA | large-scale transformer / MoE; `pretrain_gpt.py` | [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) |
| Megatron-DeepSpeed | BigScience / DeepSpeed | BLOOM- and YaLM-class runs | [bigscience-workshop/Megatron-DeepSpeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed) |
| DeepSpeed | DeepSpeed | ZeRO optimization engine | [deepspeedai/DeepSpeed](https://github.com/deepspeedai/DeepSpeed) |
| NeMo | NVIDIA | Megatron-backed GPT pretraining | [NVIDIA/NeMo](https://github.com/NVIDIA/NeMo) |
| torchtitan | PyTorch | native PyTorch 4D parallelism, pretrain → SFT | [pytorch/torchtitan](https://github.com/pytorch/torchtitan) |
| nanotron | Hugging Face | 3D parallelism; SmolLM's trainer | [huggingface/nanotron](https://github.com/huggingface/nanotron) |
| LitGPT | Lightning AI | `litgpt pretrain`, 20+ architectures | [Lightning-AI/litgpt](https://github.com/Lightning-AI/litgpt) |
| llm-foundry (+ Composer) | MosaicML | MPT/DBRX trainer | [mosaicml/llm-foundry](https://github.com/mosaicml/llm-foundry) · [mosaicml/composer](https://github.com/mosaicml/composer) |
| OLMo-core | Ai2 | OLMo 2 / 3 / Hybrid; `src/scripts/train` | [allenai/OLMo-core](https://github.com/allenai/OLMo-core) |
| open_lm ★ | ML Foundations | DCLM's trainer, standalone | [mlfoundations/open_lm](https://github.com/mlfoundations/open_lm) |
| EasyLM | young-geng | JAX LLM training (OpenLLaMA) | [young-geng/EasyLM](https://github.com/young-geng/EasyLM) |
| Levanter (+ Haliax) | Stanford | JAX, reproducibility-first (Marin) | [stanford-crfm/levanter](https://github.com/stanford-crfm/levanter) |
| MaxText | Google | JAX / TPU reference trainer | [AI-Hypercomputer/maxtext](https://github.com/AI-Hypercomputer/maxtext) |
| Paxml / Praxis | Google | JAX / TPU large-scale training | [google/paxml](https://github.com/google/paxml) · [google/praxis](https://github.com/google/praxis) |
| t5x | Google | encoder-decoder LM pretraining | [google-research/t5x](https://github.com/google-research/t5x) |
| fairseq / fairseq2 ★ | Meta | `language_modeling` / recipe-based pretraining | [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq) · [facebookresearch/fairseq2](https://github.com/facebookresearch/fairseq2) |
| ColossalAI | HPC-AI | Gemini / sharded pretraining | [hpcaitech/ColossalAI](https://github.com/hpcaitech/ColossalAI) |
| InternEvo | Shanghai AI Lab | InternLM trainer | [InternLM/InternEvo](https://github.com/InternLM/InternEvo) |
| BMTrain | OpenBMB | ZeRO-style training library | [OpenBMB/BMTrain](https://github.com/OpenBMB/BMTrain) |
| FlagScale ★ | BAAI / FlagOS | training-focused toolkit; distributed RWKV support | [flagos-ai/FlagScale](https://github.com/flagos-ai/FlagScale) |
| TencentPretrain ★ | Tencent | multi-modal pretraining toolkit | [Tencent/TencentPretrain](https://github.com/Tencent/TencentPretrain) |
| PaddleNLP (+ ERNIE) | Baidu | `llm/` pretraining pipelines | [PaddlePaddle/PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP) |
| MindFormers ★ | Huawei | Ascend one-click pretraining | [mindspore-ai/mindformers](https://github.com/mindspore-ai/mindformers) |
| veScale ★ | ByteDance | DTensor hyperscale training (partial OSS of internal stack) | [volcengine/veScale](https://github.com/volcengine/veScale) |
| Scaling ★ | Aleph Alpha | distributed 3D-parallel library with dedicated LLM module | [Aleph-Alpha-Research/scaling](https://github.com/Aleph-Alpha-Research/scaling) |
| Motif-3 training example ★ | Motif | torchtitan-based MoE trainer, train-only, targets B200 | [MotifTechnologies/motif3-training-example](https://github.com/MotifTechnologies/motif3-training-example) |
| Megatron-LLM ★ | EPFL | Megatron fork for open pretraining | [epfllm/Megatron-LLM](https://github.com/epfllm/Megatron-LLM) |
| llm-random ★ | IDEAS NCBR | research trainer (sparsity, MoE-Mamba) | [llm-random/llm-random](https://github.com/llm-random/llm-random) |
| Alpa (archived) | Alpa | auto-parallel JAX training | [alpa-projects/alpa](https://github.com/alpa-projects/alpa) |
| mesh-tensorflow | Google | classic TF model parallelism | [tensorflow/mesh](https://github.com/tensorflow/mesh) |

---

## Alternative Architecture Stacks

MoE, linear attention, SSM, and quantized training.

| Project | Org | What you train | Source |
|---|---|---|---|
| flash-linear-attention + flame ★ | fla-org | linear-attention kernels + minimal torchtitan-based trainer | [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) · [fla-org/flame](https://github.com/fla-org/flame) |
| Mamba / Mamba-2 ⚠️ | state-spaces | SSM architecture + kernels; pretrain via flame / Megatron-LM / zcookbook | [state-spaces/mamba](https://github.com/state-spaces/mamba) |
| zcookbook ★ | Zyphra | hybrid SSM / Transformer / MoE training cookbook | [Zyphra/zcookbook](https://github.com/Zyphra/zcookbook) |
| MegaBlocks ★ | Databricks | dropless MoE training | [databricks/megablocks](https://github.com/databricks/megablocks) |
| Tutel ★ | Microsoft | MoE training framework | [microsoft/tutel](https://github.com/microsoft/tutel) |
| FastMoE | laekov | MoE plugin for PyTorch | [laekov/fastmoe](https://github.com/laekov/fastmoe) |
| TorchScale ★ | Microsoft | Magneto / LongNet foundation architectures | [microsoft/torchscale](https://github.com/microsoft/torchscale) |
| RetNet ★ | Microsoft | retentive network training (`unilm/retnet`) | [microsoft/unilm](https://github.com/microsoft/unilm) |
| BitNet ⚠️ | Microsoft | 1-bit / 1.58-bit training tips + reference code (`unilm/bitnet`) | [microsoft/unilm](https://github.com/microsoft/unilm) |
| TTT ⚠️ | test-time-training | TTT-Linear / TTT-MLP layer implementations (PyTorch / JAX) | [test-time-training/ttt-lm-pytorch](https://github.com/test-time-training/ttt-lm-pytorch) · [ttt-lm-jax](https://github.com/test-time-training/ttt-lm-jax) |

---

## Decentralized and Low-Communication Training

Pretraining without a single datacenter.

| Project | Org | What it does | Source |
|---|---|---|---|
| OpenDiLoCo ★ | Prime Intellect | DiLoCo-style distributed pretraining | [PrimeIntellect-ai/OpenDiLoCo](https://github.com/PrimeIntellect-ai/OpenDiLoCo) |
| DisTrO ★ | Nous Research | low-latency distributed optimizers, massive comm reduction | [NousResearch/DisTrO](https://github.com/NousResearch/DisTrO) |
| templar ★ | one-covenant | incentivized internet-wide pretraining; 72B decentralized run | [one-covenant/templar](https://github.com/one-covenant/templar) |
| SWARM ★ | Yandex | communication-efficient model parallelism on unreliable devices | [yandex-research/swarm](https://github.com/yandex-research/swarm) |
| hivemind | learning-at-home | decentralized training primitives | [learning-at-home/hivemind](https://github.com/learning-at-home/hivemind) |
| DeDLOC | Yandex | collaborative training (encoder-era, sahajBERT) | [yandex-research/DeDLOC](https://github.com/yandex-research/DeDLOC) |
| DualPipe / DeepEP / DeepGEMM | DeepSeek | training infrastructure components (pipeline scheduling, MoE comms, GEMM kernels) | [deepseek-ai](https://github.com/deepseek-ai) |
| Muon | Keller Jordan | optimizer used in the nanogpt speedruns | [KellerJordan/Muon](https://github.com/KellerJordan/Muon) |

---

## Data Pipeline Companions

The other half of a training recipe. Listed only where they back a published pretraining run.

- [allenai/dolma](https://github.com/allenai/dolma) — OLMo's corpus toolkit
- [huggingface/datatrove](https://github.com/huggingface/datatrove) — FineWeb / SmolLM data processing
- [togethercomputer/RedPajama-Data](https://github.com/togethercomputer/RedPajama-Data) — RedPajama recipe
- [r-three/common-pile](https://github.com/r-three/common-pile) — openly-licensed 8TB corpus + collection code
- [m-a-p/Matrix](https://github.com/m-a-p/Matrix) — MAP-Neo data pipeline
- [EleutherAI/the-pile](https://github.com/EleutherAI/the-pile) — the original Pile replication code
- [LLM360/k2-data-prep](https://github.com/LLM360/k2-data-prep) — K2-65B data preparation
- [swiss-ai/pretrain-data](https://github.com/swiss-ai/pretrain-data) — Apertus corpus reconstruction scripts

---

## Borderline Cases

Real projects that don't cleanly satisfy the scope rule. Listed for honesty, not counted as from-scratch pretraining.

| Project | Why it's flagged | Source |
|---|---|---|
| SteptronOSS | StepFun's open trainer, but public docs emphasize SFT/RLVR workflows — verify pretrain support before relying on it | [stepfun-ai/SteptronOss](https://github.com/stepfun-ai/SteptronOss) |
| Qwen-AgentWorld | language *world model* with a 3-stage training pipeline; repo is largely an information hub | [QwenLM/Qwen-AgentWorld](https://github.com/QwenLM/Qwen-AgentWorld) |
| hackable_diffusion / DiffusionGemma | JAX diffusion toolbox; the released Gemma path is fine-tuning, not from-scratch LLM pretraining | [google/hackable_diffusion](https://github.com/google/hackable_diffusion) |
| Kolibri-1 | Apache-2.0 weight drop (78B-A3.46B). The trainer is **Scaling**, listed under frameworks | [Aleph-Alpha/Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1) |
| Zamba2 | model implementation; training guidance lives in **zcookbook** | [Zyphra/Zamba2](https://github.com/Zyphra/Zamba2) |

---

## Excluded by Scope

Recorded so it's clear these were evaluated and rejected, not overlooked.

**Inference / weights only — no pretraining code**
DeepSeek V1–V4 · Qwen* · Kimi K2* · GLM-4.5+ · MiniMax* · MiMo* · Step-3* · gpt-oss · Grok-1 · DBRX · Arctic · Falcon* · Mistral* · Phi* · Llama* · Gemma* · Hunyuan* · openPangu* · MiniCPM* · StableLM · Skywork · Baichuan · Yi · EuroLLM (weights + tech report; only an eval repo is public)

**Finetuning / post-training / RL**
Unsloth · Axolotl · LLaMA-Factory · TRL · torchtune · OpenRLHF · verl · slime · AReaL · open-r1 · alignment-handbook · LMFlow · ArcticTraining

**Serving / eval / agent / CLI**
vLLM · SGLang · TGI · llama.cpp · Ollama · Petals · GPT-Fast · lm-evaluation-harness · microsoft/BitNet (inference framework)

**Special case:** the official `openai/gpt-2` repo is inference-only. Use [nanoGPT](https://github.com/karpathy/nanoGPT) to train it.

---

## License Caveats

This list is CC0. The projects are not. Notable restrictions:

- **Scaling** (Aleph Alpha) — Open Aleph License, **non-commercial**
- **BLOOM** — RAIL (use restrictions); training code itself is open
- **OPT / metaseq** — non-commercial model license; training code is open
- **MPT / DBRX / Falcon / Hunyuan / MiniCPM** — mixed or community licenses; check before commercial use

Open training code ≠ open-source weights ≠ OSI-approved license. Verify each independently.

---

## Contributing

PRs are welcome, but the scope rule is enforced strictly.

**An entry must satisfy at least one:**

1. It contains a runnable from-scratch pretraining loop for a language model, **or**
2. It is a framework/infrastructure used to pretrain language models, **or**
3. It is the data pipeline for a specific published pretraining run.

**Every PR must include:**

- Direct link to the repo (not a blog post, not a model card)
- Path to the training entry point (`train.py`, `pretrain_gpt.py`, config dir, etc.)
- Parameter scale, or "framework / any"
- What is actually public: code / data / recipe / checkpoints
- License, if non-standard
- A ★ nomination with one sentence on why it's underrated, if applicable

**Instant rejections:**

- "But the weights are Apache-2.0" — not a criterion
- Finetuning, SFT, DPO, RLHF, RLVR, distillation tooling
- Inference engines, quantization-for-serving, eval harnesses, agent frameworks
- Architecture-only repos with no training path and no named trainer pairing
- Dead links or repos where the training code was promised but never shipped

If a listed repo's training code turns out to be gated, partial, or removed, open an issue with the specific path — entries get moved to **Borderline** or **Excluded**, not silently kept.

---

## Maintenance

- Snapshot date: **2026-11-13**
- Links verified against live repositories at snapshot time
- Sizes are total params, with `-A` denoting active params for MoE
- Entries marked ⚠️ ship architecture/kernels only and require a framework from this list to pretrain
- Repository activity changes weekly; treat dates and tiers as a snapshot, not a live feed

---

## License

[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, the contributors have waived all copyright and related rights to this work. Listed projects retain their own licenses.
