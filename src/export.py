"""Export a trained model as a small JSON file.

The web app cannot run Spark (free hosting has no Java), so we save everything
needed to score a comment with plain Python: vocabulary, IDF weights,
coefficients, intercepts, labels and stop words.
"""
import json

from pyspark.ml import PipelineModel

from src import config
from src.modeling import IDX_COUNTVEC, IDX_IDF, IDX_INDEXER, IDX_LR, IDX_STOPWORDS


def export_lightweight(model: PipelineModel, task: str):
    lr = model.stages[IDX_LR]
    payload = {
        "task": task,
        "labels": list(model.stages[IDX_INDEXER].labelsArray[0]),
        "stop_words": sorted(model.stages[IDX_STOPWORDS].getStopWords()),
        "vocabulary": list(model.stages[IDX_COUNTVEC].vocabulary),
        "idf": [round(float(x), 5) for x in model.stages[IDX_IDF].idf.toArray()],
        "coefficients": [[round(float(x), 5) for x in row] for row in lr.coefficientMatrix.toArray()],
        "intercepts": [round(float(x), 5) for x in lr.interceptVector.toArray()],
    }
    config.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.EXPORT_DIR / f"{task}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return path
    