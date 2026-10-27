
def validate_and_normalize_dynamic_sd_schedule(schedule):
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("Dynamic K schedule must be a non-empty list")

    ranges = []

    for entry in schedule:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise ValueError("Each rule must be [start_bs, end_bs, K]")

        start, end, k = entry

        if not all(type(v) is int for v in (start, end, k)):
            raise ValueError("Schedule values must be integers")

        if start < 1 or end < start or k < 0:
            raise ValueError("Invalid batch-size range or K")

        ranges.append((start, end, k))

    ranges.sort(key=lambda item: item[0])

    if ranges[0][0] != 1:
        raise ValueError("The first range must start at BS=1")

    previous_end = 0
    for start, end, _ in ranges:
        if start <= previous_end:
            raise ValueError("Overlapping batch-size ranges")
        previous_end = end

    return ranges


def build_dynamic_sd_schedule_lookup(
    num_speculative_tokens_per_batch_size,
    vllm_max_batch_size,
    vllm_num_speculative_tokens,
):
    max_bs = vllm_max_batch_size
    max_k = vllm_num_speculative_tokens

    if max_bs <= 0 or max_k <= 0:
        raise ValueError("max_bs and max_k must be positive")

    ranges = validate_and_normalize_dynamic_sd_schedule(
        num_speculative_tokens_per_batch_size
    )

    lookup = [0] * (max_bs + 1)

    # Expand the schedule; gaps inherit the preceding K.
    last_k = ranges[0][2]
    next_bs = 1

    for start, end, k in ranges:
        for bs in range(next_bs, min(start, max_bs + 1)):
            lookup[bs] = min(last_k, max_k)

        for bs in range(max(start, next_bs), min(end, max_bs) + 1):
            lookup[bs] = min(k, max_k)

        next_bs = max(next_bs, end + 1)
        last_k = k

        if next_bs > max_bs:
            break

    for bs in range(next_bs, max_bs + 1):
        lookup[bs] = min(last_k, max_k)

    return lookup
