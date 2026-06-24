from sysml_gpt.config import TrainConfig
from sysml_gpt.train import apply_args, parse_args


def test_cli_overrides_eval_settings(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--eval-interval",
            "7",
            "--eval-iters",
            "3",
        ],
    )

    args = parse_args()
    config = apply_args(TrainConfig(), args)

    assert config.eval_interval == 7
    assert config.eval_iters == 3
