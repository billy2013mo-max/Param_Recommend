"""Compose the platform CCE integration with benchmark-only no-save behavior.

Only the first ``sitecustomize`` on ``PYTHONPATH`` is imported.  This benchmark
directory must be first so its no-save guard is isolated from normal jobs, but
that also means it must preserve the launcher's ``ENABLE_CCE`` integration.

Both patches are installed lazily after ``llamafactory.train.tuner`` has fully
loaded.  Importing Transformers or LlamaFactory directly at interpreter startup
is deliberately avoided.
"""

from __future__ import annotations

import os
import sys


_TRUE = ("1", "true", "yes", "on")
_CCE_ENABLED = os.environ.get("ENABLE_CCE", "").strip().lower() in _TRUE
_NO_SAVE_ENABLED = os.environ.get("PARAM_RECOMMEND_BENCHMARK_NO_SAVE") == "1"

if _CCE_ENABLED:
    _LAUNCHER_DIR = "/fine-tuning-launcher"
    if _LAUNCHER_DIR not in sys.path:
        sys.path.insert(0, _LAUNCHER_DIR)


def _install_cce() -> None:
    if not _CCE_ENABLED:
        return
    try:
        import cce_llamafactory

        cce_llamafactory.install()
    except Exception as exc:  # noqa: BLE001 - platform integration is best effort
        sys.stderr.write(
            f"[ParamRecommendBenchmark] CCE integration failed: {exc!r}\n"
        )


def _install_no_save() -> None:
    if not _NO_SAVE_ENABLED:
        return

    from transformers import Trainer

    def _benchmark_save_model(self, output_dir=None, _internal_call=False):
        if self.is_world_process_zero():
            target = output_dir or self.args.output_dir
            print(
                "[ParamRecommendBenchmark] skipped model-weight save to "
                f"{target}",
                flush=True,
            )

    Trainer.save_model = _benchmark_save_model


if _CCE_ENABLED or _NO_SAVE_ENABLED:
    _state = {"done": False}
    _builtins = (
        __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
    )
    _original_import = _builtins["__import__"]

    def _hooked_import(name, *args, **kwargs):
        module = _original_import(name, *args, **kwargs)
        if not _state["done"]:
            tuner = sys.modules.get("llamafactory.train.tuner")
            if tuner is not None and hasattr(tuner, "run_exp"):
                _state["done"] = True
                _builtins["__import__"] = _original_import
                _install_cce()
                _install_no_save()
        return module

    _builtins["__import__"] = _hooked_import
