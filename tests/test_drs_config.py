from __future__ import annotations

import pytest

from drsgui.drs.config import DRSConfig


def test_paper_specified_defaults_are_locked() -> None:
    config = DRSConfig()
    assert config.focus_top_fraction == 0.15
    assert config.scatter_external_fraction == 0.10
    assert config.scatter_max_area_scale == 1.5
    assert config.shift_external_fraction == 0.15
    assert config.shift_max_iou == 0.3
    assert config.rollout_budget == 8
    assert config.max_depth == 3
    assert config.uct_exploration_constant == 1
    assert (config.reward_alpha, config.reward_beta, config.reward_gamma) == (
        0.4,
        0.4,
        0.2,
    )


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        DRSConfig(reward_alpha=0.5)
    with pytest.raises(ValueError, match="shift_max_iou"):
        DRSConfig(shift_max_iou=1.1)
