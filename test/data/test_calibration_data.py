import unittest

from transformers import AutoTokenizer

from fmchisel.data.calibration_datautil import DATASETS_DICT, HFCalibrationDataLoader


class TestCalibrationDataLoader(unittest.TestCase):

    def setUp(self):
        model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.nsamples = 50
        self.length = 128

        self.common_kwargs = {
            "tokenizer": self.tokenizer,
            "nsamples": self.nsamples,
            "max_seq_length": self.length,
        }

    def single_dataset_test(self, kwargs):
        loader = HFCalibrationDataLoader(**kwargs)

        ids = loader.get_tokenized_calibration()
        self.assertEqual(len(ids), self.nsamples)
        for id in ids:
            assert len(id["input_ids"]) <= self.length
        return True

    def form_kwargs(self, dataset_name):
        data_info = DATASETS_DICT[dataset_name]
        data_split = data_info["split"]
        data_field = data_info["field"]
        dataset = data_info["dataset"]
        data_dir = data_info["dir"]
        kwargs = {"dataset": dataset, "data_field": data_field, "data_dir": data_dir, "data_split": data_split}
        return {**self.common_kwargs, **kwargs}

    def test_c4(self):

        kwargs = self.form_kwargs("c4")
        assert self.single_dataset_test(kwargs)

    def test_cnn_dailymail(self):
        kwargs = self.form_kwargs("cnn_dailymail")
        assert self.single_dataset_test(kwargs)

    def test_wikitext(self):
        kwargs = self.form_kwargs("wikitext")
        assert self.single_dataset_test(kwargs)


if __name__ == "__main__":
    unittest.main()
