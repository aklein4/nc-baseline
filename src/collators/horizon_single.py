"""Collator for rows from ``aklein4/300k-horizons-single``."""

from collections.abc import Mapping, Sequence
from typing import Any

from numpy.typing import NDArray

from collators.horizon import HorizonCollator


class SingleHorizonCollator(HorizonCollator):
    """Tokenize one chat episode per row and return batch-leading 2D arrays."""

    def __init__(self, tokenizer_url: str, max_length: int) -> None:
        super().__init__(tokenizer_url, max_length, cluster_length=1)

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, NDArray[Any]]:
        missing = [index for index, row in enumerate(rows) if "messages" not in row]
        if missing:
            raise ValueError(
                f"SingleHorizonCollator requires a messages column; missing in rows {missing}"
            )
        messages = [
            [
                {key: value for key, value in message.items() if value is not None}
                for message in row["messages"]
            ]
            for row in rows
        ]
        return self.collate_messages(messages, (len(rows), self.max_length))
