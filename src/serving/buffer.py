"""Serving-side rolling buffer: append-and-predict is one atomic operation
(DECISIONS.md "Phase 6: serving registration, 2026-09-25", sections 3-4).

required_raw_history is defined in src/serving/bundle.py (the bundle
schema needs it at training time, before this module exists at runtime)
and re-exported here so serving code can import it from buffer.py.
"""

import pandas as pd

from config.features import SPLIT_VAL_END
from src.serving.bundle import required_raw_history

__all__ = [
    "required_raw_history",
    "SEQUENCE_MODEL_RAW_HISTORY",
    "DuplicateTimestampError",
    "NonSuccessorTimestampError",
    "InsufficientHistoryError",
    "RollingBuffer",
    "seed_buffer",
]

STEP = pd.Timedelta(minutes=10)

# The production serving buffer's capacity. A sequence model (LSTM/GRU/
# CNN-LSTM) consumes a window of L=18 consecutive feature rows
# (required_history_length = L - 1 = 17 preceding rows, per
# SequenceForecaster), and the OLDEST of those 18 feature rows itself
# needs required_raw_history() raw rows to be finite (its own lag_144
# lookback). So the raw span needed is required_history_length + raw
# rows for one feature row = 17 + 145 = 162 (verified both algebraically
# and empirically: 162 raw rows -> exactly one fully-finite L=18 window;
# 161 -> zero). LR/RF only ever need required_raw_history() (145) for a
# single feature row; the buffer is sized for the largest current
# requirement across model families so any registered family can be
# served from the same shared buffer.
#
# This is a hardcoded constant, not derived per-bundle: it assumes the
# currently-registered sequence-model config (L=18). If a future
# sequence model is ever registered with a different L, this constant
# would need recomputing (required_history_length + required_raw_history()
# for that L) — nothing here enforces that link automatically.
SEQUENCE_MODEL_RAW_HISTORY = required_raw_history() + 17


class DuplicateTimestampError(Exception):
    def __init__(self, timestamp: pd.Timestamp):
        self.timestamp = timestamp
        super().__init__(f"duplicate timestamp: {timestamp}")


class NonSuccessorTimestampError(Exception):
    def __init__(self, timestamp: pd.Timestamp, expected_next: pd.Timestamp):
        self.timestamp = timestamp
        self.expected_next = expected_next
        super().__init__(
            f"timestamp {timestamp} is not the expected successor {expected_next}"
        )


class InsufficientHistoryError(Exception):
    def __init__(self, have: int, need: int):
        self.have = have
        self.need = need
        super().__init__(f"insufficient history: have {have}, need {need}")


class RollingBuffer:
    """Fixed-capacity FIFO of (timestamp, Appliances) rows on the 10-minute grid."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._rows: list[tuple[pd.Timestamp, float]] = []

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def last_timestamp(self) -> pd.Timestamp | None:
        return self._rows[-1][0] if self._rows else None

    @property
    def is_ready(self) -> bool:
        return len(self._rows) >= self.capacity

    def check_next(self, ts: pd.Timestamp) -> None:
        last = self.last_timestamp
        if last is None:
            return
        if ts == last:
            raise DuplicateTimestampError(ts)
        expected = last + STEP
        if ts != expected:
            raise NonSuccessorTimestampError(ts, expected)

    def tentative_frame(self, ts: pd.Timestamp, value: float) -> pd.DataFrame:
        n_history = self.capacity - 1
        history = self._rows[-n_history:] if n_history > 0 else []
        dates = [row[0] for row in history] + [ts]
        values = [row[1] for row in history] + [value]
        return pd.DataFrame({"date": dates, "Appliances": values})

    def committed_frame(self) -> pd.DataFrame:
        """The currently committed rows only, no tentative new row appended.
        Read-only: unlike tentative_frame, this never represents a row that
        has not yet been committed."""
        dates = [row[0] for row in self._rows]
        values = [row[1] for row in self._rows]
        return pd.DataFrame({"date": dates, "Appliances": values})

    def commit(self, ts: pd.Timestamp, value: float) -> None:
        self._rows.append((ts, value))
        if len(self._rows) > self.capacity:
            self._rows = self._rows[-self.capacity :]


def seed_buffer(
    df_raw: pd.DataFrame,
    capacity: int,
    seed_end: str | pd.Timestamp,
    date_column: str = "date",
) -> RollingBuffer:
    boundary = pd.Timestamp(SPLIT_VAL_END)
    seed_end = pd.Timestamp(seed_end)

    pre = (
        df_raw[df_raw[date_column] < boundary]
        .sort_values(date_column)
        .reset_index(drop=True)
    )
    tail = pre.tail(capacity)

    if len(tail) < capacity:
        raise InsufficientHistoryError(have=len(tail), need=capacity)

    last_seeded = tail[date_column].iloc[-1]
    if last_seeded != seed_end:
        raise ValueError(
            f"seed_buffer: last seeded date {last_seeded} does not match seed_end {seed_end}"
        )

    diffs = tail[date_column].diff().dropna()
    if not (diffs == STEP).all():
        raise ValueError("seed_buffer: seeded rows are not contiguous on the 10-minute grid")

    buffer = RollingBuffer(capacity)
    for _, row in tail.iterrows():
        buffer.commit(row[date_column], row["Appliances"])
    return buffer
