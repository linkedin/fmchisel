from abc import ABC, abstractmethod

import datasets
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from fmchisel.config import DataLoadingConfig
from fmchisel.data.collator import DataCollatorForCompletionOnlyLM

_RETAIN_COLUMNS = {"input_ids", "attention_mask", "labels"}


CNN_RESPONSE_TEMPLATE = " <Highlight> "


class DataModule(pl.LightningDataModule, ABC):
    def __init__(self, tokenizer: AutoTokenizer, data_load_config: DataLoadingConfig):
        super().__init__()
        self.data_name = data_load_config.dataset
        self.tokenizer = tokenizer
        self.data_path = data_load_config.data_path
        self.max_length = data_load_config.max_length
        self.batch_size = data_load_config.batch_size
        self.n_train = data_load_config.n_train
        self.n_val = data_load_config.n_val
        self.return_prompt_input_ids = data_load_config.return_prompt_input_ids

    @abstractmethod
    def formatting_func(self, example):
        pass

    def tokenize(self, example):
        outputs = self.tokenizer(
            self.formatting_func(example),
            truncation=True,
            padding=False,
            max_length=self.max_length,
        )
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    @abstractmethod
    def setup(self, stage) -> None:
        self.train_dataset = self.dataset["train"].map(
            self.tokenize,
            remove_columns=list(set(self.dataset["train"].column_names) - _RETAIN_COLUMNS),
            batched=True,
            batch_size=1,
        )
        self.val_dataset = self.dataset["test"].map(
            self.tokenize,
            remove_columns=list(set(self.dataset["test"].column_names) - _RETAIN_COLUMNS),
            batched=True,
            batch_size=1,
        )
        self.dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
        )


class CNNModule(DataModule):
    def __init__(self, tokenizer: AutoTokenizer, data_load_config: DataLoadingConfig):
        super().__init__(tokenizer, data_load_config)
        response_prompt = tokenizer.encode(CNN_RESPONSE_TEMPLATE, add_special_tokens=False)
        self.collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            response_template=response_prompt,
            pad_to_multiple_of=16,
            return_prompt_input_ids=self.return_prompt_input_ids,
        )

    def formatting_func(self, example):
        output = "Given a text, please give highlights.\n\n"
        output += f"TEXT: {example['article']}\n"
        output += f" {CNN_RESPONSE_TEMPLATE} "
        output += f"{example['highlights']} "
        return [output]

    def setup(self, stage) -> None:
        self.dataset = (
            datasets.load_dataset(path=self.data_path)
            if self.data_path
            else datasets.load_dataset("cnn_dailymail", "3.0.0")
        )
        self.dataset["train"] = self.dataset["train"].select(range(self.n_train))
        self.dataset["test"] = self.dataset["test"].select(range(self.n_val))
        super().setup(stage)
