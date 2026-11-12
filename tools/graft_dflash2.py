#!/usr/bin/env python3
"""Add a DFlash2 drafter to an existing NInfer v3 Flash Next file set without reconverting it.

    python3 graft_dflash2.py [--q8] NINFER_CHECKOUT SOURCE.ninfer DRAFTER_DIR OUT_DIR

Writes OUT_DIR/<entry name>: a new entry file (the old payload under a longer directory), hard
links to the old continuation files, and one new continuation file holding the drafter's tensors
from DRAFTER_DIR/model.safetensors. The directory gains a `dflash2` component, its bindings and
its uses, named as tools/convert/dflash.py names them. The source files are only read.

Without --q8 every tensor is stored as its BF16 words, one object per tensor.

With --q8 (the DGX Spark layout; needs numpy) the large matrices become q8_g32_fp16 row-split
objects, quantized as tools/convert/quantization/groupwise.py does (group max / 127, binary16
scale, round to nearest even), and the projections that read the same input share one object:
    per layer  [key; value; query]  [6144, H]   context K/V is its first 2048 rows
    per layer  [gate; up]           [15360, H]
    per layer  output [H, 4096], down [H, 7680];  feature_projection [H, 5H]
Norms, convolution kernels and projections, the selector projection and both codebooks stay BF16.
"""

import json
import os
import struct
import sys
from pathlib import Path

HEADER = struct.Struct("<8sQ16s")
MAGIC, PART_MAGIC, ALIGN = b"NINFER\x00\x03", b"NINPRT\x00\x03", 4096
OBJECT_ALIGN = 256  # contiguous_le_v1 and row_split_k128_v1
CHUNK = 64 << 20
Q8, GROUP, QMAX = "q8_g32_fp16", 32, 127


def align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def draft_config(raw, target):
    def expect(key, value, scope=raw):
        if scope.get(key) != value:
            raise SystemExit(f"drafter config: {key} is {scope.get(key)!r}, expected {value!r}")

    expect("architectures", ["DFlash2DraftModel"])
    expect("model_type", "qwen3")
    expect("hidden_act", "silu")
    expect("attention_bias", False)
    expect("is_causal", False)
    expect("hidden_size", target["hidden_size"])
    expect("vocab_size", target["vocab_size"])
    expect("num_target_layers", target["num_hidden_layers"])
    draft = raw["dflash_config"]
    out = {"architectures": ["DFlash2DraftModel"], "model_type": "qwen3"}
    for key in ("intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim",
                "num_hidden_layers", "max_position_embeddings"):
        out[key] = int(raw[key])
    out["rms_norm_eps"] = float(raw.get("rms_norm_eps", 1e-6))
    out["rope_parameters"] = {"rope_theta": float(raw["rope_parameters"]["rope_theta"])}
    layers = raw["layer_types"]
    if len(layers) != out["num_hidden_layers"] or set(layers) != {"sliding_attention"}:
        raise SystemExit("drafter config: expected sliding attention in every layer")
    out["layer_types"] = list(layers)
    out["sliding_window"] = int(raw["sliding_window"])
    out["dflash_config"] = {
        "target_layer_ids": list(draft["target_layer_ids"]),
        "mask_token_id": int(draft["mask_token_id"]),
        "block_size": int(draft["block_size"]),
        **{k: int(draft[k]) for k in ("conv_kernel_size", "conv_group_size", "selector_rank",
                                      "selector_top_k")},
    }
    return out


def parameters(config, target):
    """(logical name, source tensor, shape, inputs, group) in tools/convert/dflash.py's order.

    `group` names the shared q8 object of --q8 and the parameter's place in it, or is None."""
    h, d = target["hidden_size"], config["head_dim"]
    q, k = config["num_attention_heads"] * d, config["num_key_value_heads"] * d
    extra, inter = config["dflash_config"], config["intermediate_size"]
    taps, groups = extra["conv_kernel_size"], h // extra["conv_group_size"]
    out = [("dflash2/feature_projection", "fc.weight", (h, len(extra["target_layer_ids"]) * h),
            ("dflash2/target_features",), ("fc", 0)),
           ("dflash2/context_norm", "hidden_norm.weight", (h,), (), None),
           ("dflash2/final_norm", "norm.weight", (h,), (), None)]
    for i in range(config["num_hidden_layers"]):
        p, sp = f"dflash2/layers/{i}/", f"layers.{i}."
        out += [(p + "input_norm", sp + "input_layernorm.weight", (h,), (), None),
                (p + "post_attention_norm", sp + "post_attention_layernorm.weight", (h,), (), None)]
        for role, field, rows, place in (("query", "q_proj", q, 2), ("key", "k_proj", k, 0),
                                         ("value", "v_proj", k, 1)):
            source = sp + "self_attn." + field + ".weight"
            out.append((p + "attention/" + role, source, (rows, h),
                        (p + "query_projection_input",), (f"kvq{i}", place)))
            if role != "query":
                out.append((p + "attention/context_" + role, source, (rows, h),
                            ("dflash2/context_input",), (f"kvq{i}", place)))
        out += [(p + "attention/query_norm", sp + "self_attn.q_norm.weight", (d,), (), None),
                (p + "attention/key_norm", sp + "self_attn.k_norm.weight", (d,), (), None),
                (p + "attention/output", sp + "self_attn.o_proj.weight", (h, q),
                 (p + "attention_output",), (f"output{i}", 0)),
                (p + "mlp/gate", sp + "mlp.gate_proj.weight", (inter, h), (p + "mlp_input",),
                 (f"gate_up{i}", 0)),
                (p + "mlp/up", sp + "mlp.up_proj.weight", (inter, h), (p + "mlp_input",),
                 (f"gate_up{i}", 1)),
                (p + "mlp/down", sp + "mlp.down_proj.weight", (h, inter), (p + "mlp_product",),
                 (f"down{i}", 0))]
        for branch in ("attention_conv", "mlp_conv"):
            out += [(p + branch + "/base_kernel", sp + branch + ".base_kernel", (2, taps, h), (),
                     None),
                    (p + branch + "/kernel_projection", sp + branch + ".kernel_projection.weight",
                     (2 * taps * groups, h), (p + branch + "_input",), None)]
    rank = extra["selector_rank"]
    out.append(("dflash2/candidate_selector/hidden_projection",
                "candidate_selector.hidden_projection.weight", (rank, h),
                ("dflash2/final_hidden",), None))
    for role in ("predecessor_codebook", "successor_codebook"):
        out.append(("dflash2/candidate_selector/" + role, "candidate_selector." + role,
                    (target["vocab_size"], rank), (), None))
    return out


def copy_range(src, dst, src_offset, dst_offset, count):
    while count:
        n = os.copy_file_range(src, dst, min(count, CHUNK), src_offset, dst_offset)
        if n <= 0:
            raise OSError("short copy")
        src_offset, dst_offset, count = src_offset + n, dst_offset + n, count - n


def quantize_q8(matrix):
    """float32 [N,K] -> row-split payload bytes and the relative RMS error."""
    import numpy as np

    n, k = matrix.shape
    if k % 128:
        raise SystemExit("q8: K must be a multiple of 128")
    grouped = matrix.reshape(n, k // GROUP, GROUP)
    max_abs = np.abs(grouped).max(axis=2)
    scale = (max_abs.astype(np.float64) / QMAX).astype(np.float32).astype(np.float16)
    scale[(scale == 0) & (max_abs > 0)] = np.float16(2.0 ** -24)
    if not np.isfinite(scale).all():
        raise SystemExit("q8: scale is not finite")
    reciprocal = np.zeros(scale.shape, np.float32)
    positive = scale > 0
    reciprocal[positive] = (1.0 / scale[positive].astype(np.float64)).astype(np.float32)
    codes = np.clip(np.rint(grouped * reciprocal[:, :, None]), -QMAX, QMAX).astype(np.int8)
    restored = codes.astype(np.float32) * scale.astype(np.float32)[:, :, None]
    error = float(np.sqrt(((restored - grouped) ** 2).sum() / max((grouped ** 2).sum(), 1e-30)))
    base = codes.tobytes()
    payload = base + b"\0" * (align(len(base), OBJECT_ALIGN) - len(base)) + scale.tobytes()
    return payload, error


def main():
    args = sys.argv[1:]
    q8 = "--q8" in args
    args = [a for a in args if a != "--q8"]
    checkout, source, drafter, out_dir = (Path(a) for a in args[:4])
    sys.path.insert(0, str(checkout))
    from tools.artifact.schema import encode_directory, parse_directory

    with source.open("rb") as f:
        magic, json_bytes, artifact_id = HEADER.unpack(f.read(HEADER.size))
        if magic != MAGIC:
            raise SystemExit("source is not a NInfer v3 entry")
        directory = json.loads(f.read(json_bytes))
    old_start = align(HEADER.size + json_bytes, ALIGN)
    if "dflash2" in directory["components"]:
        raise SystemExit("source already has a dflash2 component")
    target = directory["components"]["text"]["config"]
    config = draft_config(json.loads((drafter / "config.json").read_text()), target)

    weights = drafter / "model.safetensors"
    with weights.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        tensors = json.loads(f.read(n))
    tensors.pop("__metadata__", None)
    data_start = 8 + n

    def floats(name):
        import numpy as np

        begin, end = tensors[name]["data_offsets"]
        words = np.fromfile(weights, dtype="<u2", count=(end - begin) // 2,
                            offset=data_start + begin)
        return (words.astype(np.uint32) << 16).view(np.float32).reshape(tensors[name]["shape"])

    old_payload = sum(f["payload_bytes"] for f in directory["files"])
    state = {"offset": old_payload}
    objects, writes, placed, used, errors = [], [], {}, set(), {}
    taken = {o["id"] for o in directory["objects"]}

    def add_object(shape, format, layout, size):
        state["offset"] = align(state["offset"], OBJECT_ALIGN)
        object_id = f"weight/dflash2_{len(objects):03d}"
        if object_id in taken:
            raise SystemExit("object id collision")
        objects.append({"id": object_id, "shape": list(shape), "format": format, "layout": layout,
                        "offset": state["offset"], "bytes": size, "kind": "tensor"})
        state["offset"] += size
        return object_id, state["offset"] - size

    rows = parameters(config, target)
    groups = {}
    for name, source_name, shape, inputs, group in rows:
        t = tensors[source_name]
        if t["dtype"] != "BF16" or tuple(t["shape"]) != tuple(shape):
            raise SystemExit(f"{source_name}: {t['dtype']} {t['shape']}, expected BF16 {shape}")
        used.add(source_name)
        if q8 and group:
            groups.setdefault(group[0], {})[group[1]] = (source_name, shape)
    if used != set(tensors):
        raise SystemExit(f"unused drafter tensors: {sorted(set(tensors) - used)}")

    ranges = {}  # source tensor -> (object id, first element, end element) in --q8 groups
    for key, members in groups.items():
        import numpy as np

        parts = [members[i] for i in sorted(members)]
        width = parts[0][1][1]
        matrix = np.concatenate([floats(name) for name, _ in parts], axis=0)
        payload, errors[key] = quantize_q8(np.ascontiguousarray(matrix))
        object_id, where = add_object(matrix.shape, Q8, "row_split_k128_v1", len(payload))
        writes.append(("bytes", where, payload))
        first = 0
        for name, shape in parts:
            ranges[name] = (object_id, first * width, (first + shape[0]) * width,
                            len(parts) == 1)
            first += shape[0]

    for name, source_name, shape, inputs, group in rows:
        if source_name in ranges:
            object_id, begin, end, whole = ranges[source_name]
            directory["bindings"][name] = ({"object": object_id} if whole else
                                           {"parts": [{"object": object_id,
                                                       "range": [begin, end]}]})
        else:
            if source_name not in placed:
                begin, end = tensors[source_name]["data_offsets"]
                object_id, where = add_object(shape, "bf16", "contiguous_le_v1", end - begin)
                placed[source_name] = object_id
                writes.append(("copy", where, (data_start + begin, end - begin)))
            directory["bindings"][name] = {"object": placed[source_name]}
        for value in inputs:
            directory["uses"].append({"parameter": name, "input": value,
                                      "activation_policy": "A16Only"})
    directory["objects"] += objects
    directory["components"]["dflash2"] = {"config": config, "target": "text"}
    part_index = len(directory["files"])
    part_name = f"{source.name}.part-{part_index:04d}"
    part_bytes = state["offset"] - old_payload
    directory["files"].append({"path": part_name, "payload_bytes": part_bytes})
    directory.setdefault("provenance", {})["dflash2_graft"] = {
        "drafter": "incoai/Qwen3.8-Flash-Next-DFlash2",
        "representation": "q8_g32_fp16 shared projections, bf16 rest" if q8
        else "bf16 words unchanged"}

    parse_directory(directory, entry_name=source.name)
    encoded = encode_directory(directory)
    new_start = align(HEADER.size + len(encoded), ALIGN)
    encoded += b" " * (new_start - HEADER.size - len(encoded))
    parse_directory(json.loads(encoded), entry_name=source.name)

    out_dir.mkdir(parents=True, exist_ok=False)
    entry_payload = directory["files"][0]["payload_bytes"]
    with source.open("rb") as src, (out_dir / source.name).open("wb") as dst:
        dst.write(HEADER.pack(MAGIC, len(encoded), artifact_id))
        dst.write(encoded)
        dst.truncate(new_start + entry_payload)
        copy_range(src.fileno(), dst.fileno(), old_start, new_start, entry_payload)
    for record in directory["files"][1:part_index]:
        os.link(source.parent / record["path"], out_dir / record["path"])
    with weights.open("rb") as src, (out_dir / part_name).open("wb") as dst:
        dst.write(HEADER.pack(PART_MAGIC, part_index, artifact_id))
        dst.truncate(ALIGN + part_bytes)
        for kind, logical, what in writes:
            position = ALIGN + logical - old_payload
            if kind == "copy":
                copy_range(src.fileno(), dst.fileno(), what[0], position, what[1])
            else:
                os.pwrite(dst.fileno(), what, position)
    print(json.dumps({"entry": str(out_dir / source.name), "new_part": part_name,
                      "drafter_objects": len(objects), "drafter_bytes": part_bytes,
                      "directory_bytes": len(encoded),
                      "q8_relative_rms_error": {k: round(v, 5) for k, v in errors.items()}},
                     indent=1))


if __name__ == "__main__":
    main()
