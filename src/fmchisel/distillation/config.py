from dataclasses import dataclass, field
from typing import Literal

from dataclasses_json import Undefined, dataclass_json

from fmchisel.distillation.losses import DISTILL_LOSS_MAP


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class DistillTrainingConfig:
    """
    Configuration class for knowledge distillation training.

    Supported distillation loss functions:
    - "forward_kl": Forward KL divergence (student → teacher)
    - "reverse_kl": Reverse KL divergence (teacher → student)
    - "combined_kl": Combined forward and reverse KL divergence
    - "js": Jensen-Shannon divergence
    - "skl": Skew KL divergence
    - "tvd": Total Variation Distance

    Example:
        ```python
        config = DistillTrainingConfig(
            enable_distill=True,
            teacher_model_path="/path/to/teacher_model",
            temperature=1.0,
            distillation_loss_ratio=0.9,
            distill_loss="forward_kl",
            sample_method="on-policy",
            sample_fraction=0.5,
            max_new_tokens=50
        )
        ```
    """

    enable_distill: bool = field(default=False)
    teacher_model_path: str = field(default="")
    temperature: float = field(default=1.0)
    distillation_loss_ratio: float = field(default=0.9)
    distill_loss: str = field(default="forward_kl")
    compile_distill_loss: bool = field(
        default=False,
        metadata={
            "help": "Whether to `torch.compile` the per-token distillation loss. This is especially "
            "helpful in reducing GPU memory footprint. Default to False."
        },
    )
    forward_ratio: float = field(
        default=0.5,
        metadata={"help": "Forward loss ratio for combined KL. Ignored when using other divergence metrics."},
    )

    # [begin] Divergence loss specific configurations
    js_beta: float = field(
        default=0.5,
        metadata={"help": "Beta for Jensen-Shannon divergence. Ignored when using other divergence metrics."},
    )
    skl_alpha: float = field(
        default=0.1,
        metadata={"help": "Alpha in Skew KL divergence. Ignored when using other divergence metrics."},
    )
    tvd_log_scale: bool = field(
        default=False,
        metadata={"help": "Whether to use log scale for the input to the TVD distillation loss."},
    )
    # [end] Divergence loss specific configurations

    val_include_distill_loss: bool = field(default=False)

    # [begin] sampling and generation configs
    sample_method: Literal["supervised", "on-policy", "sequence-level"] = field(default="supervised")
    sample_fraction: float = field(
        default=1.0,
        metadata={
            "help": "Fraction of batches whose responses are sampled from student (on-policy) distribution \
                or teacher (sequence-evel)  distribution rather than using the original responses, \
                same as the huggingface GKD trainer (parameter self.lmbda). https://huggingface.co/docs/trl/gkd_trainer#trl.GKDConfig \
                e.g., 0.4 means 40% of batches are using the responses sampled from student/teacher model, with 60% using original data \
                Ignored when using supervised methods (ground-truth tokens)."
        },
    )
    max_new_tokens: int = field(
        default=100,
        metadata={
            "help": "Maximum number of tokens to generate for each response during on-policy or sequence-level sampling."
        },
    )
    sample_temperature: float = field(
        default=0.8,
        metadata={
            "help": "Sample temperature used for on-policy or sequence-level response token generation. \
                The higher the temperature, the more random the completions."
        },
    )
    # [end] sampling and generation configs
    include_prompt_loss: bool = field(
        default=False,
        metadata={"help": "Whether to include prompt token loss in the distillation loss."},
    )

    def __post_init__(self):
        if self.enable_distill:
            assert self.teacher_model_path, "Teacher model path is required for distillation training."

            supported_losses = DISTILL_LOSS_MAP.keys()
            assert (
                self.distill_loss in supported_losses
            ), f"Only {', '.join(supported_losses)} are supported for distillation training."

            assert 0.0 <= self.distillation_loss_ratio <= 1.0, "Distillation loss ratio for KL should be in [0, 1]."
            assert self.temperature > 0.0, "Temperature should be positive."

            if self.distill_loss == "combined_kl":
                assert 0.0 < self.forward_ratio < 1.0, "Forward loss ratio for combined KL should be in (0, 1)."
            if self.distill_loss == "js":
                assert 0.0 <= self.js_beta <= 1.0, "Beta for Jensen-Shannon divergence should be in [0, 1]."
            assert self.sample_method in {
                "supervised",
                "on-policy",
                "sequence-level",
            }, f"Unsupported sample method: {self.sample_method}"

            assert (
                0.0 <= self.sample_fraction <= 1.0
            ), f"Sample ratio should be in [0, 1], got sample_fraction={self.sample_fraction}."
            assert self.max_new_tokens > 0, f"max_new_tokens to generate should be positive, got {self.max_new_tokens}."
            assert (
                self.sample_temperature > 0.0
            ), f"Sample temperature should be positive, got sample_temperature={self.sample_temperature}."

    @property
    def distill_loss_kwargs(self):
        return {
            "temperature": self.temperature,
            "distillation_loss_ratio": self.distillation_loss_ratio,
            "forward_ratio": self.forward_ratio,
            "beta": self.js_beta,
            "alpha": self.skl_alpha,
            "log_scale": self.tvd_log_scale,
        }
