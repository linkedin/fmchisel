from abc import ABC, abstractmethod
from typing import List

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

# The following can be used as a reference for a few common datasets.
C4_DATA_PATH = "allenai/c4"
CNN_MAIL_DATA_PATH = "abisee/cnn_dailymail"
WIKITEXT_DATA_PATH = "Salesforce/wikitext"

DATASETS_DICT = {
    #     (split,    field,    dataset,   data_dir)
    "c4": {"split": "train", "field": "text", "dataset": C4_DATA_PATH, "dir": "en"},
    "cnn_dailymail": {"split": "train", "field": "article", "dataset": CNN_MAIL_DATA_PATH, "dir": "1.0.0"},
    "wikitext": {"split": "train", "field": "text", "dataset": WIKITEXT_DATA_PATH, "dir": "wikitext-103-raw-v1"},
}
#


class CalibrationDataLoader(ABC):

    def __init__(
        self,
        nsamples: int,
        tokenizer: AutoTokenizer,
        max_seq_length: int,
        padding: bool = False,
        truncation: bool = True,
        add_special_tokens: bool = False,
        **kwargs,
    ):

        self.nsamples = nsamples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.padding = padding
        self.truncation = truncation
        self.add_special_tokens = add_special_tokens

    @abstractmethod
    def _get_calibration_data(self) -> List[str]:
        pass

    def get_tokenized_calibration(self):

        calibration_dataset = self._get_calibration_data()

        assert isinstance(calibration_dataset, List), "Calibration dataset must be a list of strings."

        assert len(calibration_dataset) == self.nsamples, "Length of calibration data should be the same as nsamples."

        tokenized_ids = self.tokenizer.batch_encode_plus(
            calibration_dataset,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_seq_length,
            add_special_tokens=self.add_special_tokens,
        )  # {"input_ids": [[..], ..], "attention_mask": [[..], ..]}
        return Dataset.from_dict(tokenized_ids)


class HFCalibrationDataLoader(CalibrationDataLoader):

    def __init__(
        self,
        nsamples: int,
        tokenizer: AutoTokenizer,
        max_seq_length: int,
        padding: bool = False,
        truncation: bool = True,
        add_special_tokens: bool = False,
        **kwargs,
    ):

        super().__init__(
            nsamples,
            tokenizer,
            max_seq_length,
            padding,
            truncation,
            add_special_tokens,
            **kwargs,
        )
        self.dataset = kwargs.get("dataset")
        self.data_split = kwargs.get("data_split")
        self.data_field = kwargs.get("data_field")
        self.data_dir = kwargs.get("data_dir", None)

    def _get_calibration_data(self) -> List[str]:

        if self.data_dir is not None:
            ds = load_dataset(self.dataset, self.data_dir, streaming=True, split=self.data_split)[self.data_field]
        else:
            ds = load_dataset(self.dataset, streaming=True, split=self.data_split)[self.data_field]
        text_data = []
        for item in ds:
            if len(item) > 0:
                text_data.append(item)
            if len(text_data) == self.nsamples:
                break
        return text_data
