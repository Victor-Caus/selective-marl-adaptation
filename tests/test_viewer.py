from pathlib import Path

import pytest

from selective_marl.cli import build_parser
from selective_marl.evaluation.viewer import watch_checkpoint


def test_watch_command_defaults_to_semantic_before_after_session() -> None:
    args = build_parser().parse_args(["watch", "checkpoint.pt"])
    assert args.command == "watch"
    assert args.checkpoint == Path("checkpoint.pt")
    assert args.condition == "semantic"
    assert args.episodes == 2
    assert args.change_episode == 1


def test_viewer_rejects_invalid_session_before_loading_checkpoint() -> None:
    with pytest.raises(ValueError, match="episodes must be positive"):
        watch_checkpoint(Path("missing.pt"), episodes=0)
