# CDB-AV 512-request experiment (2026-11-03)

## Result

Three runs of 512 requests each completed successfully. Each run generated 65,536 output tokens.

| Run | Wall time (s) | Output TPS | E2E P50 (s) | E2E P95 (s) | E2E P99 (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 475.902 | 137.709 | 15.009 | 21.423 | 23.981 |
| 2 | 478.022 | 137.098 | 14.908 | 21.891 | 24.172 |
| 3 | 468.954 | 139.749 | 14.647 | 21.199 | 23.512 |
| **Mean TPS** | | **138.186** | | | |

Sample standard deviation of output throughput: **1.388 tok/s**.

## Version provenance

- Runtime base commit: `4a8500ab6799ae3f8f10931df9596e7fa78f8118`
- Runner bundle commit: `0d00d4de4bad3d4aad9e5a141d37e43fde855dfe`
- Runner local patch: `isolated_import_and_response_validation`
- Runtime source comparison: 1,779/1,779 tracked files under `vllm/` byte-identical to the runtime base commit.
- Runtime source fingerprint-list SHA-256: `d2e9faeff00cd70a32648b3190d26dfcbc999916c6aec5bb86b55c714f26de80`
- Runner patch SHA-256: `8a877a46c45174fa069ecf6c283e99527895eb406e519c1bc4c9c2b47f67cd1f`

## Configuration

- Target: Qwen3-8B
- Draft: Qwen3-1.7B
- GPU index: 1
- Port: 18083
- vLLM: 0.15.1
- Attention backend: FLASH_ATTN
- Client concurrency: 16
- Requests/run: 512
- Max output tokens/request: 128
- H: 8
- Initial admission: 3
- K policy: hf_confidence
- K cap: 8
- HF threshold: 0.4
- KV reuse: enabled

## Important measurement limits

The supplied `run_config.txt` does not record per-request submission timestamps or the server's `max_num_seqs`. Therefore this record does not classify arrivals as constant-rate, Poisson, or burst arrivals, and does not claim a specific server sequence limit. For a strict native-vLLM comparison, use an identical request arrival trace and matching server/model settings.

The 1,779-file equality check covers Git-tracked files under `vllm/` only. The full fingerprint list, full runner patch, and raw server logs remain on the experiment server; only their recorded digests and summarized results are included here.
