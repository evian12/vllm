# Native async scheduling audit
Actual1096679a source:
- vllm/config/vllm.py:568-601 explicitasync supports EAGLE/MTP;draft_model hardfails or auto-disables.
- gpu_model_runner.py:356-366 CDB varlen currently requires synchronous eager draft_model.
- TargetFullGraph.validate_scope and new staging scope explicitly reject async.
- async_scheduler.py:31-36 increments output placeholders and sets spec_token_ids global placeholder list;
  CDB formal context fingerprints/commit certificates/Ready/Carry require authenticated committed prefix.
Async CPU/GPU scheduling does not inherently require overlapping Draft/Target GPU forwards.
User now chooses all paper comparisons async OFF;do not implement/remove guards just to enable it.
Keep fixedH/HF.8 and continue bottom-level work;any future Async development is separate scope.
