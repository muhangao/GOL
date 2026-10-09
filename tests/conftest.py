import pytest
import torch
from gol.config import Config, ModelConfig, TrainConfig, LifeConfig
from gol.data import prepare


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


@pytest.fixture
def config():
    return Config(ModelConfig(layers=2, dim=32, heads=4, ffn_dim=64, seq_len=64),
                  TrainConfig(global_batch_size=4, micro_batch_size=2, text_steps=3,
                              warmup_reference_steps=2, lr=0.001, lr_warmup_steps=1,
                              dtype="float32", eval_every=2, eval_sequences=5, checkpoint_every=2),
                  LifeConfig(board_size=4, frames=3))


@pytest.fixture
def corpus(tmp_path):
    inputs = {}
    for split, offset in (("valid", 2000), ("warmup", 0), ("train", 1000)):
        path = tmp_path / f"{split}.txt"
        path.write_text("\n".join(f"Document {offset+i} contains a separate sequence of words for testing the loss pipeline."
                                  for i in range(40)))
        inputs[split] = path
    output = tmp_path / "corpus"
    prepare(output, inputs)
    return output
