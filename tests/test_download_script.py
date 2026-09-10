from __future__ import annotations

import pytest

from scripts.download_dataset import evenly_spaced_indices


def test_evenly_spaced_indices_cover_both_ends() -> None:
    indices = evenly_spaced_indices(total=1581, count=10)
    assert len(indices) == 10
    assert len(set(indices)) == 10
    assert indices[0] == 0
    assert indices[-1] == 1580


@pytest.mark.parametrize(
    ("total", "count"), [(0, 1), (1, 0), (1, 2)]
)
def test_evenly_spaced_indices_reject_invalid_request(total: int, count: int) -> None:
    with pytest.raises(ValueError):
        evenly_spaced_indices(total=total, count=count)
