"""Explicit experiment configuration and approximate training-compute accounting."""
from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    layers: int = 12
    dim: int = 768
    heads: int = 12
    ffn_dim: int = 2048
    seq_len: int = 1024
    rope_theta: float = 10000.0
    tied_embeddings: bool = True

    def __post_init__(self):
        for name in ("layers", "dim", "heads", "ffn_dim", "seq_len"):
            if getattr(self, name) <= 0:
                raise ValueError(f"model.{name} must be positive")
        if self.dim % self.heads or (self.dim // self.heads) % 2:
            raise ValueError("dim must be divisible by heads and head dimension must be even")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")


@dataclass(frozen=True)
class TrainConfig:
    global_batch_size: int = 128
    micro_batch_size: int = 4
    text_steps: int = 2048
    warmup_reference_steps: int = 128
    budget_match: str = "estimated_flops"
    lr: float = 0.0006
    min_lr_ratio: float = 0.1
    lr_warmup_steps: int = 20
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    dtype: str = "bfloat16"
    eval_every: int = 64
    eval_sequences: int = 128
    checkpoint_every: int = 128
    data_seed: int = 1729

    def __post_init__(self):
        for name in ("global_batch_size", "micro_batch_size", "text_steps",
                     "warmup_reference_steps", "eval_every", "eval_sequences", "checkpoint_every"):
            if getattr(self, name) <= 0:
                raise ValueError(f"train.{name} must be positive")
        if self.lr <= 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("Invalid optimizer configuration")
        if not 0 <= self.min_lr_ratio <= 1 or self.lr_warmup_steps < 0 or self.data_seed < 0:
            raise ValueError("Invalid learning-rate schedule or data seed")
        if self.budget_match not in {"tokens", "estimated_flops"}:
            raise ValueError("budget_match must be tokens or estimated_flops")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("Only float32 and bfloat16 are supported (no FP16 loss scaling)")


@dataclass(frozen=True)
class WarmupConfig:
    """Optional source-stage overrides for BOTH reset arms, never text_continue.

    Omitted fields inherit TrainConfig. The reference budget is always defined
    using train.global_batch_size, so reducing source batch increases updates,
    not the prescribed warmup token/FLOP budget.
    """
    global_batch_size: int | None = None
    micro_batch_size: int | None = None
    lr: float | None = None
    min_lr_ratio: float | None = None
    lr_warmup_steps: int | None = None
    weight_decay: float | None = None
    grad_clip: float | None = None

    def resolve(self, train: TrainConfig) -> TrainConfig:
        overrides = {k: v for k, v in asdict(self).items() if v is not None}
        for key in ("global_batch_size", "micro_batch_size", "lr_warmup_steps"):
            if key in overrides and (isinstance(overrides[key], bool) or not isinstance(overrides[key], int)):
                raise ValueError(f"warmup.{key} must be an integer")
        for key, value in overrides.items():
            if not math.isfinite(value):
                raise ValueError(f"warmup.{key} must be finite")
        return replace(train, **overrides)


@dataclass(frozen=True)
class LifeConfig:
    board_size: int = 12
    frames: int = 6
    density_min: float = 0.2
    density_max: float = 0.6
    boundary: str = "toroidal"

    def __post_init__(self):
        if self.board_size < 3 or self.frames < 2:
            raise ValueError("GoL needs board_size >= 3 and frames >= 2")
        if not 0 < self.density_min <= self.density_max < 1:
            raise ValueError("Invalid initial-density interval")
        if self.boundary not in {"toroidal", "dead"}:
            raise ValueError("boundary must be toroidal or dead")


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    life: LifeConfig = field(default_factory=LifeConfig)
    warmup: WarmupConfig | None = None

    def __post_init__(self):
        if self.model.seq_len < 2 * (self.life.board_size ** 2 + 1):
            raise ValueError("seq_len must expose at least two complete GoL frames")

        if self.warmup is not None:
            self.warmup.resolve(self.train)  # Validate before any training starts.

    def to_dict(self):
        result = asdict(self)
        if self.warmup is None:
            result.pop("warmup")  # Keep legacy config dictionaries unchanged.
        return result

    @classmethod
    def from_dict(cls, obj):
        unknown = set(obj) - {"model", "train", "life", "warmup"}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(ModelConfig(**obj.get("model", {})),
                   TrainConfig(**obj.get("train", {})),
                   LifeConfig(**obj.get("life", {})),
                   None if obj.get("warmup") is None else WarmupConfig(**obj["warmup"]))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))


def flops_per_token(model: ModelConfig, vocab_size: int) -> int:
    """Dense matmul forward+backward estimate, not measured hardware FLOPs.

    6x linear weights includes the output projection, even when embeddings are
    tied. 12*L*S*d counts dense QK and AV forward/backward. Causal kernels may
    skip part of that work. Norms, softmax, loss, optimizer, input lookup,
    validation, data generation and communication are NOT included.
    """
    d = model.dim
    linear = model.layers * (4 * d * d + 3 * d * model.ffn_dim) + d * vocab_size
    return 6 * linear + 12 * model.layers * model.seq_len * d


def phase_train_config(config: Config, arm: str, phase: str) -> TrainConfig:
    if phase == "warmup" and arm != "text_continue" and config.warmup is not None:
        return config.warmup.resolve(config.train)
    return config.train


def warmup_steps(config: Config, text_vocab: int, source_vocab: int,
                 source_batch_size: int | None = None) -> int:
    """Ceiling-match the original reference budget, not the source step count."""
    if source_batch_size is None:
        source_batch_size = phase_train_config(config, "gol_reset", "warmup").global_batch_size
    if source_batch_size <= 0:
        raise ValueError("source_batch_size must be positive")
    target = config.train.warmup_reference_steps * config.train.global_batch_size
    cost = source_batch_size
    if config.train.budget_match == "estimated_flops":
        target *= flops_per_token(config.model, text_vocab)
        cost *= flops_per_token(config.model, source_vocab)
    return (target + cost - 1) // cost


def learning_rate(step: int, total: int, config: TrainConfig) -> float:
    """1-based update index; warmup is capped for very short smoke runs."""
    warm = min(config.lr_warmup_steps, max(0, total - 1))
    if step <= warm:
        return config.lr * step / warm
    progress = (step - warm - 1) / max(1, total - warm - 1)
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    return config.lr * (config.min_lr_ratio + (1 - config.min_lr_ratio) * cosine)
