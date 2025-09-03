from dataclasses import dataclass, field
from typing import List, Literal, Union


@dataclass
class TrainingArgs:
    model_path: str
    output_dir: str
    lr: float = field(default=5e-6)
    num_epoch: int = field(default=None)
    warmup_ratio: float = field(default=0.1)
    weight_decay: float = field(default=0.1)
    val_check_interval: int = field(default=10)
    keep_sparse: bool = field(default=False)
    optimizer: str = field(default="adamw")
    enable_gradient_checkpointing: bool = field(default=True)
    gradient_accumulation_steps: int = field(default=1)
    save_on_best_validation: bool = field(default=True)
    cpu_offload: bool = field(default=False)
    use_liger: bool = field(
        default=False,
        metadata={
            "help": "Whether to use `liger-kernel` for distillation. With this flag set, we support "
            "liger chunked losses for computing distillation loss and liger flce for computing the hard loss. "
            "Currently we support FKL, RKL and JSD. Defaults to False."
        },
    )
    # LoRA Args
    lora_rank: int = field(default=None)
    use_lora: bool = field(default=False)
    lora_target_modules: Union[List[str], str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_alpha_to_rank_ratio: float = field(default=2.0)
    verify_lora_saving_correctness: bool = field(
        default=False,
        metadata={"help": "Check if the LoRA saved model is properly merged and saved? Only use for testing."},
    )

    def __post_init__(self):
        if self.optimizer not in {"adamw", "adamw_schedulefree"}:
            raise ValueError(
                f"Optimizer {self.optimizer} is not supported. Please use `adamw` or `adamw_schedulefree`."
            )
        if self.use_lora:
            assert (
                not self.keep_sparse
            ), "LoRA does not update the base weights, so they remain sparse. But the merged weights will not be sparse."


@dataclass
class DataLoadingConfig:
    data_path: str
    dataset: str = field(default="cnn_dailymail")
    max_length: int = field(default=4096)
    batch_size: int = field(default=8)
    n_train: int = field(default=16000)
    n_val: int = field(default=5000)
    return_prompt_input_ids: bool = field(default=False)


@dataclass
class CalibrationDataConfig:
    dataset: str = field(
        metadata={"help": "Dataset name from HuggingFace (e.g., allenai/c4)."},
    )
    data_split: str = field(
        metadata={"help": "What split of data to use (e.g., train, validation, etc)."},
    )
    data_field: str = field(
        metadata={"help": "What field of the data to use (e.g., text, question, etc)."},
    )
    data_dir: str = field(
        default=None,
        metadata={"help": "If applicable, the data directory from Huggingface."},
    )
    num_calibration_samples: int = 1024


@dataclass
class QuantizationConfig:

    model: str
    output_dir: str
    quantization_recipe: str = (
        field(
            metadata={
                "help": "Use W4A16, W8A8, or enter a path to a yaml recipe. Example recipes can be found at flows/inference/quantization/src/recipes."
            },
        ),
    )
    model_max_length: int = field(
        default=2048,
    )


@dataclass
class StructuredPruningConfig:
    model: str
    output_dir: str
    num_drop_mlp_neuron: int = field(
        default=0,
        metadata={"help": "Number of hidden MLP neurons to be pruned."},
    )
    num_drop_attn_group: int = field(
        default=0,
        metadata={"help": "Number of attention KV groups to be pruned."},
    )
    model_max_length: int = field(
        default=2048,
    )
    save_compressed: bool = field(
        default=True,
        metadata={
            "help": "Save the compressed smaller model on disk. If set to False, the saved model will have occupy same disk space with zero paddings in the MLP/attention layers for pruned weights."
        },
    )

    def __post_init__(self):
        if self.num_drop_attn_group < 0 or self.num_drop_mlp_neuron < 0:
            raise ValueError("num_drop_attn_group and num_drop_mlp_neuron must be non-negative integers.")
        if self.num_drop_attn_group + self.num_drop_mlp_neuron == 0:
            raise ValueError(
                "At least one mlp neuron or attn group has to be removed. got num_drop_attn_group + num_drop_mlp_neuron = 0."
            )


@dataclass
class PruningConfig:
    model: str
    output_dir: str
    pruning_yaml_recipe: str = field(
        default=None,
        metadata={
            "help": "The yaml recipe that can be used for pruning. If a valid yaml file is passed, the values of pruning_strategy, sparsity, prunen and prunem WILL BE IGNORED. Alternatively, leave this field empty and pass in pruning_strategy, sparsity, prunen and prunem."
        },
    )
    pruning_strategy: Literal["ALPS", "SparseGPT", "wanda"] = field(
        default=None,
        metadata={"help": "Method to be used for pruning. WILL BE IGNORED if pruning_yaml_recipe is passed."},
    )
    model_max_length: int = field(
        default=2048,
    )
    sparsity: float = field(
        default=0.5,
        metadata={
            "help": "The unstructured sparsity ratio. WILL BE IGNORED if pruning_yaml_recipe is passed. WILL BE IGNORED if prunen is not set to zero."
        },
    )
    prunen: int = field(
        default=2,
        metadata={"help": "The value of N in N:M sparsity. WILL BE IGNORED if pruning_yaml_recipe is passed."},
    )
    prunem: int = field(
        default=4,
        metadata={"help": "The value of M in N:M sparsity. WILL BE IGNORED if pruning_yaml_recipe is passed."},
    )
    save_compressed: bool = field(
        default=False,
        metadata={"help": "save the pruned model in the compressed format? It is recommended to be set to False."},
    )
