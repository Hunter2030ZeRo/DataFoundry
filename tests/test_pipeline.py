from datasets import Dataset, DatasetDict

from dataforge.pipeline import Pipeline

def test_dataset_pipeline() -> None:
    dataset = Dataset.from_dict({"a": [1, 2, 3]})

    result = (
    Pipeline(dataset)
        .map(lambda row: {"a": row["a"] * 2})
        .filter(lambda row: row["a"] > 2)
        .run()
    )

    assert isinstance(result, Dataset)
    assert result["a"] == [4, 6]

def test_datasetdict_pipeline() -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"a": [1, 2, 3]}),
            "validation": Dataset.from_dict({"a": [4, 5]}),
        }
    )

    result = (
        Pipeline(dataset)
        .map(lambda row: {"a": row["a"] + 1})
        .filter(lambda row: row["a"] % 2 == 0)
        .run()
    )

    assert isinstance(result, DatasetDict)
    assert result["train"]["a"] == [2, 4]
    assert result["validation"]["a"] == [6]


def test_map_remove_columns() -> None:
    dataset = Dataset.from_dict(
        {
            "text": ["hello", "world"],
            "unused": [1, 2],
        }
    )

    result = (
        Pipeline(dataset)
        .map(
            lambda row: {"length": len(row["text"])},
            remove_columns=["text", "unused"],
        )
        .run()
    )

    assert result.column_names == ["length"]
    assert result["length"] == [5, 5]
