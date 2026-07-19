from __future__ import annotations
from typing import Any, Callable
from datasets import Dataset, DatasetDict
from .hf_executor import HuggingFaceExecutor
from .operation import FilterOperation, MapOperation, Operation

class Pipeline:
    def __init__(
        self,
        dataset: Dataset | DatasetDict,
        operations: tuple[Operation, ...] = (),
    ) -> None:
        if not isinstance(dataset, (Dataset, DatasetDict)):
            raise TypeError(f"Expected a Hugging Face datasets compatible Dataset or DatasetDict, got {type(dataset)} instead.")
        self._dataset = dataset
        self._operations = operations

    def map(
        self,
        function: Callable[..., dict[str, Any]],
        *,
        batched: bool | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        remove_columns: list[str] | None = None,
    ) -> Pipeline:
        operation = MapOperation(
            function=function,
            batched=batched,
            batch_size=batch_size,
            num_workers=num_workers,
            remove_columns=tuple(remove_columns or ()),
        )
        return Pipeline(self._dataset, self._operations + (operation,))

    def filter(
        self,
        function: Callable[..., bool],
        *,
        batched: bool | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
    ) -> Pipeline:
        operation = FilterOperation(
            function=function,
            batched=batched,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        return Pipeline(self._dataset, self._operations + (operation,))

    def run(self, *, mode: str = "auto") -> Dataset | DatasetDict:
        return HuggingFaceExecutor(mode=mode).execute(self._dataset, self._operations,)
