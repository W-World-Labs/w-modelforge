# Attribution and third-party notices

## Project code

W-ModelForge includes a downstream fork of **LookAtMyQuant** in `methods/lookatmyquant/`, copyright (c) 2026 **Shin Megami Boson**, distributed under the MIT License. Source: https://github.com/ShinMegamiBoson/lookatmyquant . Baseline: `4716ffee3da205a8d1df94e4abc629040893fa53`.

W-World-Labs modifications are copyright (c) 2026 W-World-Labs contributors and are also distributed under MIT. The original notice and full permission/warranty text are preserved in [LICENSE](LICENSE). Attribution does not imply endorsement.

The inherited repository includes upstream code, cached codebooks, experiment records and documentation. Its historical results belong to the upstream authors. Our modifications are listed in [UPSTREAM.md](UPSTREAM.md).

## Methods and external software

The trellis, Viterbi, LDLQ, Hadamard and EMA codebook implementations used here come from the upstream project. Its code cites QTIP/QuIP-style methods and the e-means paper, [arXiv:2611.09501](https://arxiv.org/abs/2611.09501). **This pilot uses upstream's EMA codebooks and implementation; it does not import or vendor FAISS/e-means source code.** FAISS is a separate [MIT-licensed project](https://github.com/facebookresearch/faiss/blob/main/LICENSE).

Runtime dependencies are installed separately, retain their own licenses and are not relicensed by this repository: [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE) (BSD-style, with third-party notices), [Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE), [Datasets](https://github.com/huggingface/datasets/blob/main/LICENSE), [Safetensors](https://github.com/huggingface/safetensors/blob/main/LICENSE) (Apache-2.0), [NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt), [pandas](https://github.com/pandas-dev/pandas/blob/main/LICENSE) (BSD-3-Clause), and [Modal](https://github.com/modal-labs/modal-client/blob/main/LICENSE) (Apache-2.0). The Colab workflow separately uses [rclone](https://github.com/rclone/rclone/blob/master/COPYING) (MIT), Google OAuth libraries and the official Colab CLI's installed OAuth client resources. Preserve dependency notices when redistributing their binaries.

Optional inference scripts call separately installed [vLLM](https://github.com/vllm-project/vllm/blob/main/LICENSE) (Apache-2.0) and [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE) (MIT). Their sources and binaries are not vendored. Standard HF/GGUF conversion does not add native support for this project's packed format; engine-specific dtype changes or further quantization require separate evaluation.

The adaptive standard-GGUF path reuses upstream LookAtMyQuant's `gguf_compiler.py`, `public_eval.py` and `compiler.py` sensitivity measurement/size allocation. It uses separately installed llama.cpp for importance matrices, standard Q/IQ quantization and native evaluation. Downstream Qwen3.5 GDN layout restoration follows the MIT-licensed [pinned converter](https://github.com/ggml-org/llama.cpp/blob/93cf736a8d1d46aa922e27eb64aa712bcd5ebc30/conversion/qwen.py). These artifacts are quantized afresh from original BF16 source weights; they do not preserve the earlier trellis codes and require independent results.

## Model weights and generated artifacts

The repository does **not** contain model weights. The pilot downloads:

| Model | Fixed revision | License/source |
| --- | --- | --- |
| Qwen/Qwen3.5-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | [Apache-2.0 model license](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/LICENSE) |
| Qwen/Qwen3.8-27B | `37561816e66b5b9153b1b64c10968c18c1e6f4a0` | [Apache-2.0 model card](https://huggingface.co/Qwen/Qwen3.8-27B/blob/37561816e66b5b9153b1b64c10968c18c1e6f4a0/README.md) |

Model licenses are separate from this repository's MIT code license. Quantized artifacts are modified model weights; retain the source model's LICENSE/NOTICE files, model identity/revision, and a statement of the quantization changes. Only language weights are quantized. Optional original BF16 visual encoder/merger weights and processor assets are preserved separately without quantization or dtype conversion; actual multimodal quality requires its own validation. The language packed format is the upstream custom format.

## Calibration and evaluation data

Sources are downloaded separately: [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math), [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), [CodeParrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean), and [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext). Their dataset cards and underlying source terms remain applicable. This fork grants no new rights to the underlying text, model-generated calibration solutions or externally downloaded datasets. Private run caches and raw data are excluded from new commits.

Redistributed download/launch helper copies carry the full project MIT copyright/permission text separately from the source model Apache-2.0 LICENSE. Public GGUF model packages contain no external engine binaries; they link the pinned llama.cpp source/license and record the actual engine/library fingerprints.
