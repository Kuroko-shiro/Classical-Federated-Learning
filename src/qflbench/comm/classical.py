"""Classical communication channel + cost accounting.

Passes payloads through unchanged and records how much was sent. The quantum
channel will implement the same interface (see quantum_stub.py) but size things in
qubits/ebits and optionally perturb payloads.
"""

from __future__ import annotations

from typing import List, Optional

from ..core.interfaces import CommunicationChannel
from ..core.registry import CHANNELS
from ..core.types import Direction, Payload, TransmissionRecord


@CHANNELS.register("classical")
class ClassicalChannel(CommunicationChannel):
    def __init__(self, dtype_bytes: int = 4) -> None:
        self.dtype_bytes = dtype_bytes
        self._records: List[TransmissionRecord] = []

    def transmit(
        self, payload: Payload, direction: Direction, round_idx: int,
        client_id: Optional[int] = None,
    ) -> Payload:
        self._records.append(
            TransmissionRecord(
                round_idx=round_idx,
                client_id=client_id,
                direction=direction,
                kind=payload.kind,
                num_scalars=payload.num_scalars(),
                nbytes=payload.nbytes(self.dtype_bytes),
            )
        )
        # classical channel is lossless and identity
        return payload

    def records(self) -> List[TransmissionRecord]:
        return list(self._records)

    def reset(self) -> None:
        self._records.clear()
