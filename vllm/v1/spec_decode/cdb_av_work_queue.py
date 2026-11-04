"""CDB-AV request-level Draft Work Queue.

Control-plane bookkeeping only.
This module does not allocate KV or execute GPU forwards.
"""

from dataclasses import dataclass
from enum import Enum


class DraftWorkStatus(str, Enum):
    PREPARE_REQUIRED = "PREPARE_REQUIRED"
    ELIGIBLE = "ELIGIBLE"
    ACTIVE = "ACTIVE"
    READY = "READY"
    CARRY = "CARRY"
    STALE = "STALE"


@dataclass
class DraftWorkItem:
    req_id: str
    context_tokens: int
    context_version: int
    k_cap: int
    progress: int = 0
    status: DraftWorkStatus = DraftWorkStatus.PREPARE_REQUIRED
    source: str = "waiting"
    kv_validated: bool = False
    fingerprint: str | None = None


class CDBDraftWorkQueue:
    def __init__(self):
        self.items: dict[str, DraftWorkItem] = {}
        self.window_id: int | None = None

    def begin_window(self, window_id: int) -> None:
        """Begin a new Draft generation; no cross-window Carry yet."""
        if window_id < 1:
            raise ValueError("Invalid CDB window ID")

        if self.window_id == window_id:
            return

        if self.window_id is not None and window_id <= self.window_id:
            raise RuntimeError("CDB window ID moved backwards")

        # Window v1 retires every old work item.
        # Future Carry support must replace this with explicit preservation.
        self.items.clear()
        self.window_id = window_id

    def register(
        self,
        req_id: str,
        context_tokens: int,
        context_version: int,
        k_cap: int,
        source: str,
    ) -> DraftWorkItem:
        if not req_id:
            raise ValueError("Empty request ID")
        if context_tokens < 0 or context_version < 0:
            raise ValueError("Invalid context provenance")
        if k_cap < 1:
            raise ValueError("Invalid Draft K")

        existing = self.items.get(req_id)

        if existing is not None:
            context_changed = (
                existing.context_tokens != context_tokens
                or existing.context_version != context_version
            )

            if context_changed:
                if existing.status in (
                    DraftWorkStatus.ACTIVE,
                    DraftWorkStatus.CARRY,
                ):
                    raise RuntimeError(f"Context changed during active Draft: {req_id}")

                existing.status = DraftWorkStatus.STALE
                existing.kv_validated = False
                return existing

            return existing

        item = DraftWorkItem(
            req_id=req_id,
            context_tokens=context_tokens,
            context_version=context_version,
            k_cap=k_cap,
            source=source,
        )
        self.items[req_id] = item
        return item

    def sync_step(
        self,
        step_result: dict,
        context_versions: dict[str, tuple[int, int]],
    ) -> None:
        """Ingest a real Worker step result.

        context_versions maps req_id -> (context_tokens, version).
        """
        request_ids = list(step_result["request_ids"])
        active = set(step_result["active_req_ids"])
        ready = set(step_result["ready_req_ids"])
        progress = step_result["request_progress"]

        if len(request_ids) != len(set(request_ids)):
            raise RuntimeError("Duplicate Draft request IDs")
        if active & ready or active | ready != set(request_ids):
            raise RuntimeError("Invalid Draft step partition")
        if set(progress) != set(request_ids):
            raise RuntimeError("Draft progress mapping mismatch")

        for req_id in request_ids:
            if req_id not in context_versions:
                raise RuntimeError(f"Missing context version: {req_id}")

            context_tokens, context_version = context_versions[req_id]
            state = progress[req_id]
            current = int(state["progress"])
            cap = int(state["k_cap"])

            if not 0 <= current <= cap:
                raise RuntimeError("Invalid Draft progress")

            item = self.register(
                req_id=req_id,
                context_tokens=context_tokens,
                context_version=context_version,
                k_cap=cap,
                source="running",
            )
            if item.status == DraftWorkStatus.STALE:
                raise RuntimeError(f"Cannot sync stale Draft work item: {req_id}")

            if item.k_cap != cap:
                raise RuntimeError("Draft K changed inside a window")
            if current < item.progress:
                raise RuntimeError("Draft progress moved backwards")

            item.progress = current
            item.status = (
                DraftWorkStatus.READY if req_id in ready else DraftWorkStatus.ACTIVE
            )

            # Real GPU execution occurred, but independent Draft KV
            # validity has not yet been proven.
            item.kv_validated = False

    def retire(self, req_id: str) -> None:
        self.items.pop(req_id, None)

    def snapshot(self) -> dict:
        counts = {status.value: 0 for status in DraftWorkStatus}

        for item in self.items.values():
            counts[item.status.value] += 1

        return {
            "total": len(self.items),
            "counts": counts,
            "ready_ids": [
                item.req_id
                for item in self.items.values()
                if item.status == DraftWorkStatus.READY
            ],
            "prepare_ids": [
                item.req_id
                for item in self.items.values()
                if item.status == DraftWorkStatus.PREPARE_REQUIRED
            ],
        }
