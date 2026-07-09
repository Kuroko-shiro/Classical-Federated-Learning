"""Quantum-internet communication channel — STUB (the QFL integration point).

This file exists to make the swap point concrete and to document, in code, exactly
where the team's quantum-internet work (roadmap #3/#4) plugs in. It is NOT used by
the classical benchmark; it shows the intended shape.

The same `transmit` signature is reused. A real implementation would:
  - choose an encoding (e.g. amplitude encoding) and size it in QUBITS from
    payload.num_scalars();
  - model entanglement distribution: each transmission consumes EBITS; generation
    can FAIL with some probability -> retry/buffering -> extra latency/rounds;
  - model decoherence: memory decay perturbs the payload tensors (noise) as a
    function of storage/wait time;
  - fill TransmissionRecord.qubits / .ebits so the SAME accounting + plots compare
    classical bytes vs quantum qubits/ebits head to head.

By keeping all of this behind CommunicationChannel, none of the Strategy/Engine
code changes when we go quantum.
"""

from __future__ import annotations

from typing import List, Optional

from ..core.interfaces import CommunicationChannel
from ..core.registry import CHANNELS
from ..core.types import Direction, Payload, TransmissionRecord


@CHANNELS.register("quantum_stub")
class QuantumChannelStub(CommunicationChannel):
    def __init__(
        self,
        ent_success_prob: float = 0.5,   # entanglement generation success per attempt
        decoherence: float = 0.0,        # memory decay -> payload noise std
        qubits_per_scalar: float = 1.0,  # placeholder encoding cost
    ) -> None:
        self.ent_success_prob = ent_success_prob
        self.decoherence = decoherence
        self.qubits_per_scalar = qubits_per_scalar
        self._records: List[TransmissionRecord] = []

    def transmit(self, payload, direction, round_idx, client_id=None) -> Payload:
        raise NotImplementedError(
            "QuantumChannelStub documents the integration point. Implement "
            "qubit/ebit accounting, entanglement-failure retries, and decoherence "
            "noise here (roadmap #3/#4). Interface matches ClassicalChannel."
        )

    def records(self) -> List[TransmissionRecord]:
        return list(self._records)

    def reset(self) -> None:
        self._records.clear()
