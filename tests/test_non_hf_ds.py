from dataforge.pipeline import Pipeline

import pytest

def test_pipeline_rejects_non_hf_dataset() -> None:
    with pytest.raises(TypeError, match=f"Expected a Hugging Face datasets compatible Dataset or DatasetDict, got {type([{'a': 1}])} instead."):
        Pipeline([{"a": 1}])
