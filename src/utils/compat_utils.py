"""Convert outer-loop OLoop slow weights to ordinary Llama safetensors.

See docs/getting-started.md#evaluation. Run this module with a Python environment
containing torch, safetensors, and huggingface_hub; JAX evaluation needs no torch.
Source names follow outer-loop/src/models/__init__.py and its scanned layers.
Only original Llama parameters survive; adaptation parameters are discarded.
"""

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

WRAPPER_COMPONENTS = frozenset(("_orig_mod", "_module"))


def llama_parameter_shapes(config: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    """Allowlist the bias-free Llama architecture used by both repositories."""
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    vocab = config["vocab_size"]
    head_dim = config.get("head_dim", hidden // config["num_attention_heads"])
    shapes = {
        "model.embed_tokens.weight": (vocab, hidden),
        "model.norm.weight": (hidden,),
        "lm_head.weight": (vocab, hidden),
    }
    layer_shapes = {
        "input_layernorm.weight": (hidden,),
        "post_attention_layernorm.weight": (hidden,),
        "self_attn.q_proj.weight": (config["num_attention_heads"] * head_dim, hidden),
        "self_attn.k_proj.weight": (config["num_key_value_heads"] * head_dim, hidden),
        "self_attn.v_proj.weight": (config["num_key_value_heads"] * head_dim, hidden),
        "self_attn.o_proj.weight": (hidden, config["num_attention_heads"] * head_dim),
        "mlp.gate_proj.weight": (intermediate, hidden),
        "mlp.up_proj.weight": (intermediate, hidden),
        "mlp.down_proj.weight": (hidden, intermediate),
    }
    for layer in range(config["num_hidden_layers"]):
        shapes.update(
            {
                f"model.layers.{layer}.{key}": shape
                for key, shape in layer_shapes.items()
            }
        )
    return shapes


def llama_state_dict(
    state: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Strip wrappers, discard extra weights, and require a complete backbone.

    Tensor values are left untouched so this validation is usable without torch.
    Never silently fill a missing backbone parameter with a random initialization.
    """
    shapes = llama_parameter_shapes(config)
    selected = {}
    for name, value in state.items():
        name = ".".join(p for p in name.split(".") if p not in WRAPPER_COMPONENTS)
        name = name.replace("model.layers.layers.", "model.layers.")
        if name not in shapes:
            continue
        if name in selected:
            raise ValueError(
                f"Duplicate Llama parameter after removing wrappers: {name}"
            )
        if tuple(value.shape) != shapes[name]:
            raise ValueError(
                f"{name}: expected shape {shapes[name]}, got {value.shape}"
            )
        selected[name] = value
    missing = shapes.keys() - selected.keys()
    if config.get("tie_word_embeddings", False):
        missing.discard("lm_head.weight")
    if missing:
        raise ValueError(f"Missing Llama parameters: {sorted(missing)}")
    return selected


def convert_checkpoint(
    source: str,
    output: str | Path,
    step: int,
    revision: str | None = None,
) -> Path:
    """Convert one local checkpoint root or Hub repository, retaining its step."""
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file

    if step < 0:
        raise ValueError("step must be nonnegative")
    step_name = f"{step:012d}"
    target = Path(output).expanduser().resolve() / step_name
    # Refuse to accidentally replace a previously converted artifact.
    if target.exists():
        raise FileExistsError(f"Output already exists: {target}")
    root = Path(source).expanduser()
    if root.exists():
        root = root.resolve()
    else:
        root = Path(
            snapshot_download(
                source,
                revision=revision,
                allow_patterns=[f"{step_name}/config.json", f"{step_name}/model.pt"],
            )
        )
    config = json.loads((root / step_name / "config.json").read_text())
    # Restricted deserialization and mmap avoid executing checkpoint code or
    # materializing the discarded OLoop tensors. Official API:
    # https://pytorch.org/docs/stable/generated/torch.load.html
    state = torch.load(
        root / step_name / "model.pt", map_location="cpu", weights_only=True, mmap=True
    )
    selected = llama_state_dict(state, config)
    # Clone retained tensors to detach mmap storage and handle tied weights.
    selected = {k: v.detach().contiguous().clone() for k, v in selected.items()}
    target.mkdir(parents=True)
    save_file(selected, target / "model.safetensors")
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {
        "source": source,
        "resolved_source": str(root),
        "revision": revision,
        "step": step,
        "retained_tensors": len(selected),
        "discarded_tensors": len(state) - len(selected),
        "conversion": "Llama backbone only; OLoop adaptation parameters discarded",
    }
    (target / "conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(selected)} Llama tensors to {target}", flush=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", help="Hugging Face repository or local checkpoint root"
    )
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="Converted checkpoint root"
    )
    parser.add_argument("--revision", help="Optional pinned Hugging Face revision")
    args = parser.parse_args()
    convert_checkpoint(args.source, args.output, args.step, args.revision)


if __name__ == "__main__":
    main()
