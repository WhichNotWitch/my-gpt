from sysml_gpt.config import TrainConfig
from sysml_gpt.train import apply_args, apply_checkpoint_config, parse_args


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


def test_checkpoint_config_restores_tokenizer_metadata():
    config = TrainConfig(tokenizer="char", vocab_size=1000)
    checkpoint = {
        "block_size": 128,
        "n_embed": 128,
        "n_layer": 4,
        "num_heads": 4,
        "dropout": 0.2,
        "vocab_size": 500,
        "tokenizer": {"type": "byte_bpe"},
    }

    apply_checkpoint_config(config, checkpoint)

    assert config.tokenizer == "byte-bpe"
    assert config.vocab_size == 500
