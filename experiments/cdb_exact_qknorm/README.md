# Exact BF16 QK Norm/RoPE candidate
Isolated experimental CUDA operator; default dispatch is unchanged.
The explicit opt-in model path below is validated only in the recorded static scope.
Preserves native vector8/CUB1024-valid16 RMSNorm reduction and BF16 rounding,
including each multiply/add/subtract in RoPE. Head128 full NeoX only.
Separate contiguous Q/K outputs; QKV/V/weights/cache/positions are readonly.
API checks shape/dtype/aliasing; invalid GPU position triggers a device assertion.
Build with existing CUDA12.4/GCC11, Torch2.9.1+CUDA12.9. No dependency changes.
CPU compile and CPU-input guard are checked before the first GPU experiment.
GPU acceptance requires raw-byte equality against the existing native operators;
empty intervals are inapplicable. Accuracy failure prohibits promotion.
This is not a production feature and not an end-to-end speedup claim.

## Experimental model dispatch
CDB_AV_EXACT_QKNORM=1 enables the candidate (default0).
CDB_AV_EXACT_QKNORM_MODE=both|draft|target (defaultboth).
CDB_AV_EXACT_QK_LIB must point to the built .so with a matching .json receipt.
Only static eager BF16 SM89 CDB/capacity4/TP1 basic head128 NeoX is accepted.
Batch-invariant mode,other custom-op dispatch,quantization,LoRA and prefix caching are rejected.
Rows>1024 or unsupported layout use the complete original path.
CDB_AV_EXACT_QK_VALIDATE=1 compares actual model Q/K bytes against native outside capture;
SAMPLES defaults8 per layer. Existing Graph/KV validation remains independent.
No scheduling,verification,lease or weight logic is modified.

## Raw API alignment guard
Vector8 QKV and learned-norm-weight pointers must be16-byte aligned.
Contiguous offset tensors are explicitly rejected before launch;model allocations are aligned.
The guard adds no valid-input GPU operations. Keep build receipts from each source version.
Negative API tests require CDB_EXACT_NEGATIVE_GPU=1 and a separately reserved GPU3.
