"""Decoder input-name bindings for the SAM2 Hiera decoder.

The decoder MBLT has several inputs with identical shapes, so calibration must
never rely on array position alone. Every MBLT input name is mapped to a
semantic role, and the generated calibration manifest records both the names
and the roles so `model_compile.py` can reject a mismatched pair.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

# MBLT input name -> semantic role, in the order the decoder MBLT reports.
DEFAULT_DECODER_MBLT_BINDINGS: dict[str, str] = {
    "image_embeddings": "image_embeddings",
    "dense_prompt_embeddings": "dense_prompt_embeddings",
    "image_pe": "image_pe",
    "sparse_prompt_embeddings_0": "sparse_prompt_embeddings",
    "high_res_features0_0": "hrf0_nhwc",
    "high_res_features1_0": "hrf1_nhwc",
}

DECODER_ROLES = frozenset(DEFAULT_DECODER_MBLT_BINDINGS.values())


def load_binding_map(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return dict(DEFAULT_DECODER_MBLT_BINDINGS)
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise ValueError(f"decoder input binding file must be a string map: {path}")
    return data


def read_mblt_input_names(path: str | Path) -> list[str]:
    """Read top-level MBLT input names without loading weight buffers."""
    from mblt.serialize import SerializeMeta as MbltSerializeMeta

    path = str(path)
    header = MbltSerializeMeta.read_header(path)
    if header.is_legacy:
        from qbcompiler.model_dict.serialize import SerializeMeta

        legacy_header = SerializeMeta.get_header(path)
        model_dict = SerializeMeta.get_model_dict(path, header=legacy_header)
    else:
        model_dict = MbltSerializeMeta.get_model_dict(path)
    return list(model_dict.inputs)


def resolve_decoder_bindings(input_names: Sequence[str], mapping: Mapping[str, str] | None = None) -> list[str]:
    """Resolve every MBLT input to a semantic role, in MBLT input order."""
    mapping = DEFAULT_DECODER_MBLT_BINDINGS if mapping is None else mapping
    missing = [name for name in input_names if name not in mapping]
    if missing:
        raise ValueError(
            f"unsupported SAM2 decoder input names: {missing}. Provide an explicit --decoder-input-bindings JSON map."
        )
    roles = [mapping[name] for name in input_names]
    unknown_roles = sorted(set(roles) - DECODER_ROLES)
    if unknown_roles:
        raise ValueError(f"unknown decoder roles in binding map: {unknown_roles}")
    if len(roles) != len(set(roles)):
        raise ValueError(f"decoder roles must be one-to-one, got {roles}")
    missing_roles = sorted(DECODER_ROLES - set(roles))
    if missing_roles:
        raise ValueError(f"decoder binding map is missing roles: {missing_roles}")
    return roles
