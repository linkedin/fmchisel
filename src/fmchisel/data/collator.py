import warnings
from typing import Any, Dict, List, Union

import numpy as np
import torch
from transformers.data.data_collator import DataCollatorForLanguageModeling


# Code taken from yudai_linkedin's e2e flow
# copied from OSS trl repo https://github.com/huggingface/trl/blob/a7dc892717a1503d5f68f94af870b523fe14bc94/trl/trainer/utils.py#L75
# for avoiding additional dependency
class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
        return_prompt_input_ids (`bool`, *optional*, defaults to `False`): Whether or not to return the prompt only input ids and corresponding prompt attention mask
    """

    def __init__(
        self,
        response_template: Union[str, List[int]],
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        return_prompt_input_ids=False,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)
        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template
        self.ignore_index = ignore_index
        self.return_prompt_input_ids = return_prompt_input_ids

    def torch_call(self, examples: List[Union[List, Any, Dict]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        if self.return_prompt_input_ids:
            # create empty container
            batch["prompt_input_ids"] = torch.full(
                batch["input_ids"].shape,
                self.tokenizer.pad_token_id,
                dtype=batch["input_ids"].dtype,
                device=batch["input_ids"].device,
            )
            batch["prompt_attention_mask"] = torch.zeros_like(batch["attention_mask"])

        for i in range(len(examples)):
            response_token_ids_start_idx = None

            for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                if (
                    self.response_token_ids == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist()
                ):  # noqa: E203
                    response_token_ids_start_idx = idx

            if response_token_ids_start_idx is None:
                warnings.warn(
                    f"Could not find response key `{self.response_template}` in the instance. "
                    f"This instance will be ignored in loss calculation. "
                    f"Note, if this happens often, consider increasing the `max_seq_length`."
                )
                batch["labels"][i, :] = self.ignore_index
                if self.return_prompt_input_ids:
                    # no response token found, all ids in this row are prompt ids
                    batch["prompt_input_ids"][i, :] = batch["input_ids"][i, :]
                    batch["prompt_attention_mask"][i, :] = 1
            else:
                response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

                # Make pytorch loss function ignore all tokens up through the end of the response key
                batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index
                if self.return_prompt_input_ids:
                    batch["prompt_input_ids"][i, -response_token_ids_end_idx:] = batch["input_ids"][
                        i, :response_token_ids_end_idx
                    ]
                    batch["prompt_attention_mask"][i, -response_token_ids_end_idx:] = 1
        return batch
