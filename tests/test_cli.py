from selective_marl.cli import build_parser


def test_train_accepts_explicit_no_intervention_schedule() -> None:
    args = build_parser().parse_args(
        ["train", "--algorithm", "rmappo", "--training-conditions", "none"]
    )

    assert args.training_conditions == ["none"]


def test_pipeline_accepts_all_intervention_conditions() -> None:
    args = build_parser().parse_args(
        [
            "pipeline",
            "--algorithms",
            "rmappo",
            "--training-conditions",
            "none",
            "semantic",
            "behavioral",
            "both",
        ]
    )

    assert args.training_conditions == ["none", "semantic", "behavioral", "both"]
