from __future__ import annotations

from typing import Optional, Self

from datasets import Dataset, DatasetDict, load_dataset
from pydantic import BaseModel


PREFIX = "Price is EGP"
QUESTION = "What is the price of this car?"


class CarItem(BaseModel):
    """One Egyptian-market listing in Week 7 prompt/completion form."""

    brand: str
    model: str
    year: int
    mileage: int
    fuel: str
    transmission: str
    description: str
    price: float
    prompt: str
    completion: str
    gen_suspect: bool = False
    drop_reason: Optional[str] = None
    id: Optional[int] = None

    @property
    def title(self) -> str:
        return f"{self.brand} {self.model} {self.year}"

    def test_prompt(self) -> str:
        """Prompt for inference: everything up to and including PREFIX, no number."""
        if PREFIX not in self.prompt:
            return self.prompt.rstrip() + "\n" + PREFIX
        return self.prompt.split(PREFIX)[0] + PREFIX

    def __repr__(self) -> str:
        return f"<{self.title} = EGP {self.price:,.0f}>"

    def count_prompt_tokens(self, tokenizer) -> int:
        full = self.prompt + self.completion
        return len(tokenizer.encode(full, add_special_tokens=False))

    def to_datapoint(self) -> dict:
        return {"prompt": self.prompt, "completion": self.completion}

    def to_eval_row(self) -> dict:
        return {**self.to_datapoint(), "price": self.price, "title": self.title}

    @staticmethod
    def push_prompts_to_hub(
        dataset_name: str, train: list[Self], val: list[Self], test: list[Self]
    ):
        """Push prompt/completion splits. Do not call unless explicitly asked."""
        DatasetDict(
            {
                "train": Dataset.from_list([item.to_datapoint() for item in train]),
                "val": Dataset.from_list([item.to_datapoint() for item in val]),
                "test": Dataset.from_list([item.to_datapoint() for item in test]),
            }
        ).push_to_hub(dataset_name)

    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list[Self], list[Self], list[Self]]:
        ds = load_dataset(dataset_name)
        return (
            [cls.model_validate(row) for row in ds["train"]],
            [cls.model_validate(row) for row in ds["validation"]],
            [cls.model_validate(row) for row in ds["test"]],
        )
