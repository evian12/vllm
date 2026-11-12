# SPDX-License-Identifier: Apache-2.0
"""Single-copy Draft metadata; fresh storage, no asynchronous host reuse."""
import torch


def packed_metadata_tensors(blocks, seqs, qlens, slots, device):
    """Preserve int32 attention fields and int64 physical slot IDs.

    Align metadata fields to 16 bytes in the allocation only. There are
    no extra token rows, sequence rows, query positions or visible table columns.
    Every call owns new CPU/GPU storage. The blocking .to() preserves the
    lifetime of the packed pageable source without a reusable pinned-buffer
    protocol or changes to KV page ownership.
    """
    nr = len(blocks)
    if not nr or len(seqs) != nr or len(qlens) != nr:
        raise ValueError("Packed Draft metadata request counts differ")
    if any(not row for row in blocks) or any(q < 1 for q in qlens):
        raise ValueError("Packed Draft metadata requires nonempty requests")
    if len(slots) != sum(qlens):
        raise ValueError("Packed Draft metadata slot count differs from query tokens")
    width = max(map(len, blocks))
    offsets = [0]
    for q in qlens:
        offsets.append(offsets[-1] + q)
    table = [item for row in blocks for item in list(row) + [0] * (width - len(row))]
    ints = list(offsets)
    ints.extend([0] * (-len(ints) % 4))
    seq_start = len(ints)
    ints.extend(seqs)
    ints.extend([0] * (-len(ints) % 4))
    table_start = len(ints)
    ints.extend(table)
    ints.extend([0] * (-len(ints) % 4))
    host32 = torch.tensor(ints, dtype=torch.int32, device="cpu")
    prefix64 = host32.numel() // 2
    host64 = torch.cat((host32.view(torch.int64),
                        torch.tensor(slots, dtype=torch.int64, device="cpu")))
    # Exactly one H2D call; all device fields below are dtype-preserving views.
    device64 = host64.to(device)
    device32 = device64[:prefix64].view(torch.int32)
    return dict(
        query_start_loc=device32[:nr + 1],
        query_start_loc_cpu=host32[:nr + 1],
        seq_lens=device32[seq_start:seq_start + nr],
        block_table_tensor=device32[table_start:table_start + nr * width].view(nr, width),
        slot_mapping=device64[prefix64:],
    )
