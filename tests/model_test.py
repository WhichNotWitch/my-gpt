import torch

from sysml_gpt.model import TinyGPTLanguageModel


def test_tiny_gpt_forward_backward_smoke():
    torch.manual_seed(0)

    vocab_size = 17
    block_size = 8

    model = TinyGPTLanguageModel(
        vocab_size=vocab_size,
        n_embed=16,
        block_size=block_size,
        n_layer=1,
        num_heads=2,
        dropout=0.0,
    )

    x = torch.randint(0, vocab_size, (2, block_size))
    y = torch.randint(0, vocab_size, (2, block_size))

    logits, loss = model(x, y)

    assert logits.shape == (2 * block_size, vocab_size)
    assert loss is not None

    loss.backward()