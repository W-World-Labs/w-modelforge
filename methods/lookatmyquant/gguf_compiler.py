"""The compression compiler for llama.cpp: the menu is llama.cpp's own tensor types,
measured as llama.cpp makes them, and the plan is a llama-quantize command, so the result runs in llama.cpp, Ollama and
LM Studio. Runs inside modal_gguf.py's image (llama.cpp built with CUDA, at the commit in /llama.cpp/COMMIT).

    prepare   HF → GGUF (bf16) with llama.cpp's convert_hf_to_gguf.py; an importance matrix (llama-imatrix) on the text mix's
              calibration split; every menu type everywhere (--pure, with the importance matrix; embedding and output head
              forced to the type too); llama.cpp's stock mixes for comparison; evaluation text (WikiText-2 test, the mix's
              held-out split)
    measure   for each type, every tensor of the GGUF dequantized (transformers' GGUF loader, which undoes llama.cpp's q/k
              permutation) and swapped alone into the Hugging Face model: the KL to the original on held-out windows, the
              same as compiler.py measure; bits = the tensor's bytes in the GGUF. Output: a measurement compiler.py
              allocate reads (items named as compiler.py's, options = llama.cpp types)
    build     a plan → llama-quantize with a --tensor-type override per tensor (and the embedding and output types)
    evaluate  llama-perplexity: perplexity and KL to the bf16 GGUF's logits, on WikiText-2 test and on the mix

    python gguf_compiler.py prepare --model unsloth/Llama-3.2-3B
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HF = Path("/root/.cache/huggingface")
LLAMA = Path("/llama.cpp")
BIN = LLAMA / "build" / "bin"
MENU = ["Q2_K", "IQ3_XXS", "Q3_K", "IQ4_XS", "Q4_K", "Q5_K", "Q6_K", "Q8_0", "BF16"]
STOCK = ["Q2_K", "Q3_K_M", "IQ4_XS", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]  # llama.cpp's mixes, the baselines (IQ4_XS: the type's own mix)


def log(msg):
    print(msg, flush=True)


def sh(cmd, **kw):
    cmd = [str(c) for c in cmd]
    log("$ " + " ".join(cmd)[:600])
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode:
        log(p.stdout[-3000:] + p.stderr[-3000:])
        raise SystemExit(f"failed: {cmd[0]}")
    log(f"  ({time.time() - t0:.0f}s)")
    return p.stdout + p.stderr


def model_dir(repo, local=""):
    if local:
        return str(Path(local).resolve(strict=True))
    snapshots = glob.glob(str(HF / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / "*/"))
    if len(snapshots) != 1:
        raise ValueError("specify --model-path for an exact cached revision")
    return snapshots[0]


def work(repo, directory=""):
    w = Path(directory) if directory else HF / "gguf" / repo.split("/")[-1]
    w.mkdir(parents=True, exist_ok=True)
    return w


def tag_of(repo, cfg):
    return repo.split("/")[-1].lower()


def candidate(w, kind):
    return w / ("bf16.gguf" if kind == "BF16" else f"pure_{kind}.gguf")


def prepare(args):
    from transformers import AutoConfig, AutoTokenizer
    import compiler_arch as A
    w = work(args.model, args.work_dir)
    src = model_dir(args.model, args.model_path)
    cfg = AutoConfig.from_pretrained(src, local_files_only=True)
    base = w / "bf16.gguf"
    if not base.exists():
        sh([args.converter_python, LLAMA / "convert_hf_to_gguf.py", src, "--outtype", "bf16", "--outfile", base,
            *(["--no-mtp"] if "qwen3_5" in cfg.model_type else [])])
    tok = AutoTokenizer.from_pretrained(src, local_files_only=True)
    mix = A.TextMix(tok, tag_of(args.model, cfg))
    calib = w / "calib.txt"  # the text mix's calibration split, as text: every source equally (64 windows of 512 tokens)
    calib.write_text("\n\n".join(tok.decode(x) for x in mix.calibration(args.calib, args.context)))
    imat = w / "imatrix.gguf"
    if not imat.exists():
        sh([BIN / "llama-imatrix", "-m", base, "-f", calib, "-o", imat, "-c", args.context, "--chunks", args.calib, "-ngl", "999"])
    for t in ([] if args.skip_pure else args.menu.split(",")):
        f = candidate(w, t)
        if not f.exists():
            try:  # a type llama.cpp refuses for this model (IQ3_XXS needs importance data for every tensor; Gemma 4's
                # KV-shared layers keep key/value tensors no token reaches) is left off the menu
                sh([BIN / "llama-quantize", "--imatrix", imat, "--pure", "--token-embedding-type", t, "--output-tensor-type", t, base, f, t])
            except SystemExit:
                log(f"  {t}: refused by llama-quantize for this model; not on the menu")
                if f.exists():
                    f.rename(f.with_suffix(".gguf.broken"))  # set aside, not deleted
    for t in filter(None, args.stock.split(",")):
        f = w / f"stock_{t}.gguf"
        if not f.exists():
            sh([BIN / "llama-quantize", "--imatrix", imat, base, f, t])
        (w / f"stock_{t}.json").write_text(json.dumps(gguf_tensors(f)))  # each tensor's bytes, for the budgets
    from datasets import load_dataset
    (w / "wikitext_test.txt").write_text("\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]))
    ev = mix.windows("eval", args.eval_chunks, args.context, seed=2)
    (w / "mix_eval.txt").write_text("\n\n".join(tok.decode(ev[s][k]) for k in range(args.eval_chunks) for s in mix.sources))
    from row_allocation import file_sha256
    (w / "manifest.json").write_text(json.dumps({"model": args.model, "model_path": src,
        "commit": subprocess.check_output(["git", "-C", str(LLAMA), "rev-parse", "HEAD"], text=True).strip(),
        "config_sha256": file_sha256(Path(src) / "config.json"), "bf16_sha256": file_sha256(base),
        "calibration_sha256": file_sha256(calib), "imatrix_sha256": file_sha256(imat),
        "context": args.context, "calibration_chunks": args.calib,
        "menu": args.menu.split(","), "stock": list(filter(None, args.stock.split(",")))}, indent=1))
    log(f"prepared {w}")


def gguf_tensors(path):
    """{gguf tensor name: (bytes, shape)} from a GGUF file."""
    from gguf import GGUFReader
    r = GGUFReader(str(path))
    return {t.name: (int(t.n_bytes), tuple(int(x) for x in t.shape)) for t in r.tensors}


def unpermute(W, n):
    """Undo convert_hf_to_gguf.py's rotary permutation of a Llama q or k matrix (n = its heads)."""
    rows, cols = W.shape
    return W.reshape(n, rows // n // 2, 2, cols).swapaxes(1, 2).reshape(rows, cols)


def qwen35_layout(name, cfg):
    """Inverse of pinned llama.cpp qwen.py's grouped-to-tiled GDN V-head order."""
    if "qwen3_5" not in cfg.model_type or ".linear_attn." not in name:
        return None
    keys, values = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    if keys == values:
        return None
    axis, prefix, width = 0, 0, cfg.linear_value_head_dim
    if ".in_proj_qkv." in name or ".conv1d." in name:
        prefix = 2 * keys * cfg.linear_key_head_dim
    elif ".in_proj_a." in name or ".in_proj_b." in name:
        width = 1
    elif ".out_proj." in name:
        axis = 1
    elif ".in_proj_z." not in name:
        return None
    return f"v_heads:{axis}:{prefix}:{keys}:{values // keys}:{width}"


def restore_layout(X, transform):
    if transform == "T":
        return X.T
    if transform.startswith("unpermute:"):
        return unpermute(X, int(transform.split(":")[1]))
    if transform.startswith("v_heads:"):
        axis, prefix, keys, per_key, width = map(int, transform.split(":")[1:])
        tail = X.narrow(axis, prefix, X.shape[axis] - prefix)
        shape = list(tail.shape)
        expanded = shape[:axis] + [per_key, keys, width] + shape[axis + 1:]
        tail = tail.reshape(expanded).transpose(axis, axis + 1).reshape(shape)
        return torch.cat((X.narrow(axis, 0, prefix), tail), dim=axis) if prefix else tail
    if transform != "id":
        raise ValueError(f"unknown GGUF layout {transform}")
    return X


def dequantized(path, names, cfg):
    """{gguf tensor name: float32 (out, in) tensor} for `names`, in Hugging Face's layout."""
    import numpy as np
    from gguf import GGUFReader, dequantize
    r = GGUFReader(str(path))
    arch = str(bytes(r.fields["general.architecture"].parts[-1]), "utf-8")
    out = {}
    for t in r.tensors:
        if t.name not in names:
            continue
        a = dequantize(t.data, t.tensor_type).astype(np.float32)
        W = torch.from_numpy(a.reshape(int(t.shape[1]), int(t.shape[0])) if len(t.shape) == 2 else a)
        if arch == "llama" and t.name.endswith("attn_q.weight"):
            W = unpermute(W, cfg.num_attention_heads)
        if arch == "llama" and t.name.endswith("attn_k.weight"):
            W = unpermute(W, cfg.num_key_value_heads)
        out[t.name] = W
    return out


def measure(args):
    """Every llama.cpp type, tensor by tensor, swapped alone into the Hugging Face model: KL to the original."""
    import compiler_arch as A
    from transformers import AutoTokenizer
    w = work(args.model, args.work_dir)
    src = model_dir(args.model, args.model_path)
    tok = AutoTokenizer.from_pretrained(src)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    from row_allocation import load_original_model, _write_json, file_sha256
    model = load_original_model(src, dtype=args.dtype, device=dev)
    for p in model.parameters():
        p.requires_grad_(False)
    mix = A.TextMix(tok, tag_of(args.model, model.config))
    per = max(1, -(-args.windows // len(mix.sources)))
    wd = mix.windows("calib", per, args.context, seed=1)
    held = A.with_bos(torch.cat([wd[s] for s in mix.sources])[: args.windows], tok, model)
    bs = args.kl_batch
    with torch.no_grad():
        ref_lp = [F.log_softmax(model(held[b : b + bs].to(dev), use_cache=False).logits.float(), -1) for b in range(0, len(held), bs)]

    @torch.no_grad()
    def kl():
        tot = n = 0.0
        for b, lr in zip(range(0, len(held), bs), ref_lp):
            lp = F.log_softmax(model(held[b : b + bs].to(dev), use_cache=False).logits.float(), -1)
            tot += (lr.exp() * (lr - lp)).sum().item()
            n += lp.shape[0] * lp.shape[1]
        return tot / n
    floor = kl()
    import public_eval as P
    lpath, layers = A.decoder_layers(model)
    emb, head, tied = A.head_and_embedding(model)
    ple = A.per_layer_embedding(model)
    params = dict(model.named_parameters())
    if not (w / "bf16.gguf").exists():
        raise ValueError("run prepare before measure")
    mapping = P.gguf_map(model, w / "bf16.gguf", w / "gguf_map.json")  # gguf tensor → (parameter, layout, scale), by content
    mapped = json.loads((w / "gguf_map.json").read_text())
    if mapped["unmatched gguf tensors"] or mapped["parameters without a gguf tensor"]:
        raise ValueError("incomplete GGUF/HF mapping; refusing a partial sensitivity plan")

    def item_of(pname):
        """The compiler's item for a parameter: a matrix of decoder layer i, the embedding, head or per-layer table, or
        another matrix of the text model (Gemma 4's per-layer projection)."""
        P_ = params[pname]
        if P_ is emb.weight:
            return "embed.0", "embed", "embed", None
        if head is not None and not tied and P_ is head.weight:
            return "head.0", "head", "head", None
        if ple is not None and P_ is ple.weight:
            return "ple.0", "ple", "ple", None
        pre = lpath + "."
        if pname.startswith(pre):
            i, rel = pname[len(pre):].split(".", 1)
            return f"{i}.{rel[: -len('.weight')]}", "linear", rel[: -len(".weight")], int(i)
        return f"other.{pname[: -len('.weight')]}", "linear", "other", None
    out = {"model": args.model, "method": "gguf", "data": "mix", "dtype": args.dtype,
           "held-out windows": [len(held), args.context],
           "held_ids_sha256": hashlib.sha256(held.cpu().numpy().tobytes()).hexdigest(),
           "source_sha256": {n: file_sha256(Path(__file__).parent / n) for n in ("gguf_compiler.py", "public_eval.py", "compiler.py", "compiler_arch.py")},
           "kl floor (nothing quantized)": floor,
           "tied": tied, "embed buckets": [emb.weight.shape[0]], "embed order": "frequency in the text mix", "device": dev,
           "manifest": json.loads((w / "manifest.json").read_text()), "records": []}
    recs = out["records"]
    from gguf import GGUFReader, dequantize

    def readable(path):
        try:
            GGUFReader(str(path))
            return True
        except Exception:
            path.rename(path.with_suffix(".gguf.broken"))  # a half-written file from a refused type: set aside
            log(f"  {path.name}: unreadable, set aside")
            return False
    for t in [t for t in args.menu.split(",") if candidate(w, t).exists() and readable(candidate(w, t))]:
        t0 = time.time()
        path = candidate(w, t)
        n_hit = 0
        for gt in GGUFReader(str(path)).tensors:
            if gt.name not in mapping:
                continue
            pname, tr, c = mapping[gt.name]
            P_ = params[pname]
            X = torch.from_numpy(dequantize(gt.data, gt.tensor_type).astype(np.float32))
            X = X.reshape(int(gt.shape[1]), int(gt.shape[0])) if len(gt.shape) >= 2 else X
            X = restore_layout(X, tr)
            orig = P_.data.clone()
            P_.data.copy_((c * X).reshape(P_.shape).to(P_.dtype))
            d = kl() - floor
            P_.data.copy_(orig)
            item, kind, role, layer = item_of(pname)
            rec = {"item": item, "kind": kind, "role": role, "option": t, "bits": 8.0 * int(gt.n_bytes), "weights": P_.numel(), "dkl": d,
                   "gguf tensor": gt.name, "stored type": gt.tensor_type.name, "shape": list(P_.shape)}
            if layer is not None:
                rec["layer"] = layer
            if kind != "linear":
                rec |= {"bucket": 0, "rows": P_.shape[0]}
            recs.append(rec)
            n_hit += 1
            _write_json(args.out, out)
        log(f"{t}: {n_hit} tensors in {time.time() - t0:.0f}s")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
    log(f"done: {len(recs)} records")


def gguf_suffix(hf_suffix):
    """The tail of llama.cpp's tensor name for a Hugging Face parameter (gguf-py's TensorNameMap, without the arch)."""
    m = re.match(r"layers\.(\d+)\.(.+)\.weight$", hf_suffix)
    if not m:
        return {"embed_tokens.weight": "token_embd.weight", "lm_head.weight": "output.weight"}[hf_suffix]
    i, name = m.groups()
    tail = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
            "self_attn.qkv_proj": "attn_qkv", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
            "mlp.gate_up_proj": "ffn_up"}.get(name, name)
    return f"blk.{i}.{tail}.weight"


def build(args):
    """A plan (compiler.py allocate over a gguf measurement, --sens) → llama-quantize with a --tensor-type override for
    every tensor the plan places (the measurement says which GGUF tensor each item is)."""
    w = work(args.model, args.work_dir)
    plan = json.loads(Path(args.plan).read_text())
    sens = json.loads(Path(args.sens).read_text())
    gname = {r["item"]: r["gguf tensor"] for r in sens["records"]}
    stored = {(r["item"], r["option"]): r["stored type"] for r in sens["records"]}
    choice = dict(plan["linears"])
    for kind in ("embed", "head", "ple"):
        if plan.get(kind):
            choice[f"{kind}.0"] = plan[kind]["programs"][0]
    over, want = [], {}
    for item, t in sorted(choice.items()):
        if item in gname:
            want[gname[item]] = stored[item, t]
            over += ["--tensor-type", f"^{re.escape(gname[item])}$={stored[item, t].lower()}"]
    out = w / f"plan_{Path(args.plan).stem}.gguf"
    if out.exists():
        raise ValueError(f"output already exists: {out}")
    sh([BIN / "llama-quantize", "--pure", "--imatrix", w / "imatrix.gguf", *over, w / "bf16.gguf", out, args.default])
    from gguf import GGUFReader  # the check: every tensor got the plan's type (llama.cpp falls back where a type cannot apply)
    got = {t.name: t.tensor_type.name for t in GGUFReader(str(out)).tensors}
    off = {n: (want[n], got.get(n)) for n in want if got.get(n) != want[n]}
    log(f"built {out} ({out.stat().st_size / 1e9:.3f} GB): {len(want) - len(off)} of {len(want)} tensors as planned; "
        f"different: {list(off.items())[:8]}")
    if off:
        raise ValueError(f"GGUF stored types differ from plan: {off}")
    expected_bits = sum(r["bits"] for r in sens["records"] if choice.get(r["item"]) == r["option"])
    actual_bits = 8 * sum(int(t.n_bytes) for t in GGUFReader(str(out)).tensors if t.name in want)
    if actual_bits != expected_bits:
        raise ValueError("GGUF controlled tensor sizes differ from measured plan")
    return out


def evaluate(args):
    """llama-perplexity on WikiText-2 test and the mix: perplexity, and KL to the bf16 GGUF's logits."""
    w = work(args.model, args.work_dir)
    res = {}
    tmp = w / "kld"
    tmp.mkdir(exist_ok=True)
    for text in ("wikitext_test", "mix_eval"):
        base_kld = tmp / f"base_{text}.kld"
        common = ["-c", str(args.context), "--chunks", str(args.eval_chunks), "-ngl", "999"]
        if not base_kld.exists():
            sh([BIN / "llama-perplexity", "-m", w / "bf16.gguf", "-f", w / f"{text}.txt", *common, "--kl-divergence-base", base_kld])
        for f in args.files.split(","):
            g = w / f
            o = sh([BIN / "llama-perplexity", "-m", g, "-f", w / f"{text}.txt", *common, "--kl-divergence", "--kl-divergence-base", base_kld])
            (w / f"{g.stem}-{text}-native-ppl.log").write_text(o)
            ppl = re.search(r"Mean PPL\(Q\)\s*:\s*([\d.]+)", o) or re.search(r"Final estimate: PPL = ([\d.]+)", o)
            original_ppl = re.search(r"Mean PPL\(base\)\s*:\s*([\d.]+)", o)
            kld = re.search(r"Mean\s+KLD:\s*([\d.]+)", o)
            if not (ppl and original_ppl and kld):
                raise ValueError("native PPL/KL output is incomplete; inspect saved command log")
            values = [float(hit.group(1)) for hit in (ppl, original_ppl, kld)]
            if not all(math.isfinite(value) for value in values) or min(values[:2]) <= 0:
                raise ValueError("native PPL/KL is nonfinite or invalid")
            from row_allocation import file_sha256
            res.setdefault(f, {"bytes": g.stat().st_size})[text] = {
                "ppl": values[0], "original_bf16_ppl": values[1], "ppl_ratio": values[0] / values[1],
                "kl": values[2], "context": args.context, "chunks": args.eval_chunks,
                "text_sha256": file_sha256(w / f"{text}.txt")}
            log(f"{f} {text}: ppl {res[f][text]['ppl']}, KL {res[f][text]['kl']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


def inspect(args):
    """A GGUF's architecture, tensor count, bytes, and its first tensors' names, types and shapes."""
    from gguf import GGUFReader
    for f in args.files.split(","):
        path = Path(f if f.startswith("/") else sorted(glob.glob(str(HF / "hub" / f)))[0])
        r = GGUFReader(str(path))
        arch = str(bytes(r.fields["general.architecture"].parts[-1]), "utf-8")
        log(f"{path.name}: {arch}, {len(r.tensors)} tensors, {sum(int(t.n_bytes) for t in r.tensors) / 1e9:.3f} GB of tensors, file {path.stat().st_size / 1e9:.3f} GB")
        for t in r.tensors[:12] + [t for t in r.tensors if "per_layer" in t.name or "token_embd" in t.name][:6]:
            log(f"   {t.name} {t.tensor_type.name} {tuple(int(x) for x in t.shape)} {int(t.n_bytes)}")


def main():
    global LLAMA, BIN
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "measure", "build", "evaluate", "inspect"):
        s = sub.add_parser(name)
        s.add_argument("--model", required=True)
        s.add_argument("--model-path", default="", help="exact local HF snapshot")
        s.add_argument("--work-dir", default="")
        s.add_argument("--llama-dir", default=str(LLAMA))
        s.add_argument("--converter-python", default=sys.executable, help="isolated GGUF converter Python")
        s.add_argument("--menu", default=",".join(MENU))
        s.add_argument("--stock", default=",".join(STOCK), help="comma-separated baselines; empty skips them")
        s.add_argument("--calib", type=int, default=64)
        s.add_argument("--context", type=int, default=512)
        s.add_argument("--eval-chunks", type=int, default=64)
        s.add_argument("--kl-batch", type=int, default=4)
        s.add_argument("--dtype", default="float32")
        s.add_argument("--windows", type=int, default=16)
        s.add_argument("--plan", default="")
        s.add_argument("--sens", default="", help="build: the gguf measurement the plan was allocated from")
        s.add_argument("--default", default="Q4_K", help="build: the type of tensors the plan does not name")
        s.add_argument("--files", default="", help="evaluate: GGUF files in the model's work directory")
        s.add_argument("--out", default="")
        s.add_argument("--skip-pure", action="store_true", help="prepare: only the stock mixes (no type-everywhere files)")
    args = ap.parse_args()
    LLAMA = Path(args.llama_dir).resolve()
    BIN = LLAMA / "build/bin"
    {"prepare": prepare, "measure": measure, "build": build, "evaluate": evaluate, "inspect": inspect}[args.cmd](args)


if __name__ == "__main__":
    main()
