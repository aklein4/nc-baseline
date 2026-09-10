"""Transformers tokenizer collator for ordinary text rows.

References:
- docs/components.md#data-and-collators
- https://huggingface.co/docs/transformers/main_classes/tokenizer
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from transformers import AutoTokenizer


class TokenizeCollator:
    def __init__(
        self,
        tokenizer_url: str,
        sequence_length: int,
        text_key: str = "text",
        pad_token_id: int = -100,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_url)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.sequence_length = sequence_length
        self.text_key = text_key
        self.pad_token_id = pad_token_id

    def __call__(
        self, rows: Sequence[Mapping[str, Any]]
    ) -> dict[str, NDArray[np.int32]]:
        encoded = self.tokenizer(
            [row[self.text_key] for row in rows],
            truncation=True,
            padding="max_length",
            max_length=self.sequence_length,
            return_tensors="np",
        )
        ids = np.where(
            encoded["attention_mask"], encoded["input_ids"], self.pad_token_id
        )
        return {"input_ids": ids.astype(np.int32)}
