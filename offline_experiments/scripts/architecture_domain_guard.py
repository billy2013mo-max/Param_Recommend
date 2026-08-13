#!/usr/bin/env python
"""Shared architecture predicates for the fail-closed domain gates.

Both predictors need to answer the same two questions -- "is this a
vision-language checkpoint?" and "is this a hybrid-attention stack?" -- and the
answers must not drift between them.  Before this module the memory side owned
the only VL predicate (``h800_physical_v4b_predictor._is_vision_language``) and
the throughput side had none at all, so a VL request was refused by one model
and silently priced by the other.

Two properties this module exists to guarantee:

* **One definition, two callers.**  A model that is out of domain is out of
  domain on both sides, by construction rather than by matching edits.
* **Uncertainty resolves to "out of domain".**  Every predicate here returns
  True when the config cannot be interpreted, because the caller uses the
  answer to *refuse* work.  A parse failure must never read as "supported".

Deliberately not covered: whether a supported-looking checkpoint is actually
*calibrated*.  These predicates describe architecture only.  Model-id
whitelists and evidence tiers stay with each predictor.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SCHEMA = "sft_architecture_domain_guard/v1"
IMPLEMENTATION_VERSION = "sft_architecture_domain_guard_impl/2026-08-06.v1"

VISION_LANGUAGE_REASON = "vision_language_outside_supported_domain"
HYBRID_ATTENTION_REASON = "hybrid_attention_outside_supported_domain"


def _text(value: Any) -> str:
    return str(value or "").strip().lower()


def _family(model: Mapping[str, Any]) -> str:
    """Family-ish text for substring probes.

    Kept identical in spirit to the memory-side predicate it replaces: the
    ``model_id`` is included because several registry entries carry the
    architecture only in their id (``qwen3_vl_8b``).
    """
    return " ".join(
        _text(model.get(key))
        for key in ("family", "model_family", "model_id", "id", "model_type")
    )


def _architectures(model: Mapping[str, Any]) -> str:
    return " ".join(_text(value) for value in (model.get("architectures") or []))


def is_vision_language(model: Mapping[str, Any]) -> bool:
    """True when the checkpoint carries a vision tower.

    Preserves the behaviour of the memory-side predicate this replaces,
    including the deliberately loose ``"vl" in family`` substring probe: a
    false positive costs one needless refusal, a false negative costs a
    silently mispriced multimodal job.
    """
    if not isinstance(model, Mapping):
        return True
    if model.get("vision_config"):
        return True
    for key in ("image_token_id", "video_token_id", "vision_start_token_id"):
        if model.get(key) is not None:
            return True
    haystack = f"{_family(model)} {_architectures(model)}"
    return "vision" in haystack or "vl" in haystack


def is_hybrid_attention(model: Mapping[str, Any]) -> bool:
    """True when the decoder stack is not uniformly softmax attention.

    ``layer_types`` is the authoritative signal and is read through
    :func:`structural_linear_work.layer_type_counts`, so the two modules cannot
    disagree about what a layer schedule means.  That function raises on a layer
    kind it does not know; here such a config is *out of domain* rather than an
    error, which is the whole point of a fail-closed predicate.
    """
    if not isinstance(model, Mapping):
        return True

    from structural_linear_work import LINEAR_ATTENTION, layer_type_counts

    try:
        counts = layer_type_counts(model)
    except Exception:
        # Unreadable or unrecognised layer schedule: refuse, do not guess.
        return True
    if counts.get(LINEAR_ATTENTION, 0) > 0:
        return True

    # A uniform stack by layer_types can still be a generation whose hybrid
    # schedule lives under a key we do not parse.  These two markers are only
    # emitted by such stacks, so their presence contradicts the uniform read.
    geometry = model.get("text_config")
    if not isinstance(geometry, Mapping) or geometry.get("hidden_size") is None:
        geometry = model
    for key in ("linear_attention", "full_attention_interval", "linear_conv_kernel_dim"):
        if geometry.get(key) is not None:
            return True
    return False


def architecture_domain_refusals(model: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return ``(reason_code, detail)`` for every architecture-level refusal.

    Empty means the architecture is inside the calibrated domain.  Both
    predictors call this and raise their own ``unsupported`` label from it, so
    the two never disagree about which architectures are out of scope.
    """
    refusals: list[tuple[str, str]] = []
    if is_vision_language(model):
        refusals.append(
            (
                VISION_LANGUAGE_REASON,
                "Vision-language checkpoints are outside the calibrated domain: "
                "vision-encoder activation, multimodal workspace, ZeRO gather of "
                "vision parameters and image-token sequence inflation are not "
                "modelled.",
            )
        )
    if is_hybrid_attention(model):
        refusals.append(
            (
                HYBRID_ATTENTION_REASON,
                "Hybrid-attention checkpoints are outside the calibrated domain: "
                "the analytic basis assumes a uniform softmax-attention stack, so "
                "both the per-layer activation width and the attention workspace "
                "are wrong for linear-attention layers.",
            )
        )
    return refusals


_CONFIG_CACHE: dict[str, dict[str, Any]] = {}


def _checkpoint_config(model_path: str) -> dict[str, Any]:
    """Read and cache a checkpoint ``config.json``.

    An unreadable path yields ``{}``, which the predicates treat as out of
    domain -- the fail-closed direction.  Cached because the fit loop asks about
    the same handful of checkpoints thousands of times.
    """
    if model_path in _CONFIG_CACHE:
        return _CONFIG_CACHE[model_path]
    from pathlib import Path

    config: dict[str, Any] = {}
    try:
        import json

        raw = (Path(model_path) / "config.json").read_text(encoding="utf-8")
        loaded = json.loads(raw)
        if isinstance(loaded, dict):
            config = loaded
    except Exception:
        config = {}
    _CONFIG_CACHE[model_path] = config
    return config


_REGISTRY_CACHE: dict[str, dict[str, Any]] | None = None


def _registry() -> dict[str, dict[str, Any]]:
    """``model_id`` -> registry entry, merged over the known registries.

    Needed because the record shapes that reach the predictors do not all carry
    a checkpoint path.  A native calibration record states only
    ``scenario.model_id``; the path lives in the registry, and without this
    lookup there is no way to reach the ``config.json`` that names the
    architecture.  Missing or malformed registries yield ``{}``, which leaves the
    caller with no path -- handled as absent evidence, not as "supported".
    """
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is not None:
        return _REGISTRY_CACHE
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    project = root.parent
    merged: dict[str, dict[str, Any]] = {}
    sources = [
        root / "config" / "models.json",
        root / "artifacts" / "model_inventory.json",
    ]
    # Separate-campaign inventories.  Qwen3.5 in particular is registered only
    # here (its campaign uses its own venv), so without this the guard would
    # refuse ``qwen3p5_4b`` merely for lack of evidence rather than on its
    # actual layer schedule -- the same verdict, but unfalsifiable.
    sources.extend(
        sorted(project.glob("offline_experiments_*/artifacts/model_inventory.json"))
    )
    for source in sources:
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except Exception:
            continue
        entries = raw.get("models") if isinstance(raw, Mapping) else raw
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, Mapping) and entry.get("id") is not None:
                merged.setdefault(str(entry["id"]), dict(entry))
    _REGISTRY_CACHE = merged
    return merged


def architecture_view_of_observation(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Build an architecture description from a record the predictors see.

    Three record shapes reach here and they do not agree on where the model is
    named, so all three are read:

    * canonical observation rows -- ``configuration.job`` with a checkpoint path;
    * native calibration records (:func:`h800_native_memory_calibration.
      build_native_record`) -- ``scenario.model_id`` only, *no* ``configuration``
      key at all and a ``model_basis`` of plain geometry integers.  This is the
      shape V5's fit loop actually predicts on, so a view that read only
      ``configuration.job`` left the gate inert exactly where it has to work;
    * bare registry/config mappings, used by callers holding a request rather
      than an observation.

    Whichever names the model, the identifying fields are merged *over* the
    checkpoint config, because a registry id such as ``qwen3_vl_8b`` is
    sometimes the only VL evidence present.

    Returns ``None`` when the record carries *no* architecture evidence
    whatsoever.  That is deliberately distinct from "evidence says out of
    domain": a mechanism-only record (as the mechanism-routing unit tests use)
    must not be refused for architecture, because there is no architecture claim
    in it to refuse.  Callers that hold a real prediction request should reject a
    ``None`` view themselves -- see :func:`observation_architecture_refusals`,
    which treats it as in-domain, versus a request path, which should demand
    evidence.
    """
    if not isinstance(record, Mapping):
        return None
    job = (record.get("configuration") or {}).get("job") or {}
    if not isinstance(job, Mapping):
        job = {}
    scenario = record.get("scenario")
    if not isinstance(scenario, Mapping):
        scenario = {}

    model_id = (
        job.get("model_id")
        or scenario.get("model_id")
        or record.get("model_id")
    )
    family = job.get("model_family") or record.get("model_family")
    model_path = job.get("runtime_model_path") or job.get("model_path")
    if not model_path and model_id is not None:
        entry = _registry().get(str(model_id)) or {}
        model_path = entry.get("path")
        family = family or entry.get("family")

    identifiers = {
        key: value
        for key, value in (("model_id", model_id), ("family", family))
        if value is not None
    }
    if not model_path and not identifiers:
        return None
    view: dict[str, Any] = dict(
        _checkpoint_config(str(model_path)) if model_path else {}
    )
    view.update(identifiers)
    return view


def observation_architecture_refusals(
    record: Mapping[str, Any]
) -> list[tuple[str, str]]:
    """:func:`architecture_domain_refusals` for a canonical observation row.

    A row with no architecture evidence at all yields no refusals: absence of a
    claim is not a claim.  Refusing it would fail closed on the wrong axis --
    every mechanism-only record would become out-of-domain, including the ones
    the calibrated fleet is fitted on.
    """
    view = architecture_view_of_observation(record)
    if view is None:
        return []
    return architecture_domain_refusals(view)


