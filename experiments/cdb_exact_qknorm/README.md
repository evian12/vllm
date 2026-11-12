# Exact BF16 QK Norm/RoPE candidate
Isolated experimental CUDA operator, not installed in any model path.
Preserves native vector8/CUB1024-valid16 RMSNorm reduction and BF16 rounding,
including each multiply/add/subtract in RoPE. Head128 full NeoX only.
Separate contiguous Q/K outputs; QKV/V/weights/cache/positions are readonly.
API checks shape/dtype/aliasing; invalid GPU position triggers a device assertion.
Build with existing CUDA12.4/GCC11, Torch2.9.1+CUDA12.9. No dependency changes.
CPU compile and CPU-input guard are checked before the first GPU experiment.
GPU acceptance requires raw-byte equality against the existing native operators;
empty intervals are inapplicable. Accuracy failure prohibits promotion.
This is not a production feature and not an end-to-end speedup claim.
