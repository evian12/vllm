# SPDX-License-Identifier: Apache-2.0
"""CDB-AV provisional Draft step session.

Supports fixed membership or shrinking varlen membership; no new admission.
Not a cross-target-forward Carry object. Generator captures mutable model buffers.
"""

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CDBDraftSession:
    generator: Generator[dict[str, Any], None, Any]
    steps_completed: int = 0
    done: bool = False
    result: Any = None
    last_report: dict[str, Any] | None = field(default=None, repr=False)
    request_ids: tuple[str, ...] = ()
    request_draft_lengths: tuple[int, ...] = ()
    draft_progress: tuple[int, ...] = ()
    active_indices: tuple[int, ...] = ()
    ready_indices: tuple[int, ...] = ()
    _cdb_state_store: Any | None = field(default=None, repr=False)
