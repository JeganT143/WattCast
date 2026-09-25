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
    "DuplicateTimestampError",
    "NonSuccessorTimestampError",
    "InsufficientHistoryError",
    "RollingBuffer",
    "seed_buffer",
]

STEP = pd.Timedelta(minutes=10)


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
