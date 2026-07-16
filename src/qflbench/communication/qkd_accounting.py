"""Convert encrypted application payloads into OTP/QKD key budgets."""

from __future__ import annotations

from typing import Iterable


DEFAULT_KEY_RATES_BPS = (10_000, 50_000, 100_000, 1_000_000)


def qkd_key_budget(
    encrypted_payload_bytes: int,
    *,
    key_rates_bps: Iterable[int] = DEFAULT_KEY_RATES_BPS,
) -> dict:
    required_bits = 8 * int(encrypted_payload_bytes)
    rates = [int(rate) for rate in key_rates_bps]
    if any(rate <= 0 for rate in rates):
        raise ValueError("key rates must be positive")
    return {
        "assumption": "one-time pad; one key bit per encrypted payload bit",
        "encrypted_payload_bytes": int(encrypted_payload_bytes),
        "required_key_bits": required_bits,
        "key_generation_seconds": {
            str(rate): required_bits / rate for rate in rates
        },
    }
