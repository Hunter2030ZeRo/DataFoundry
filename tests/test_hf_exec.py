from datasets import Dataset
from dataforge.hf_executor import HuggingFaceExecutor
from dataforge.operation import MapOperation

def test_executor_applies_map_op() -> None:
    dataset = Dataset.from_dict({"a": [1, 2, 3]})

    operations = (
        MapOperation(
            function=lambda row: {"a": row["a"] * 10},
        ),
    )

    result = HuggingFaceExecutor(mode="auto").execute(
        dataset,
        operations,
    )

    assert result["a"] == [10, 20, 30]
