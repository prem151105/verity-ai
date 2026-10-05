"""Optional batched local inference. Imports/downloads occur only on explicit use."""
from functools import lru_cache


class LocalNLI:
    def __init__(self, backend="torch"):
        if backend not in {"torch", "onnx"}:
            raise ValueError("backend must be torch or onnx")
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(
            "cross-encoder/nli-deberta-v3-small", backend=backend,
            revision="e9890682d9e4279b7ae6d0fcfb435a43206280ec",
            max_length=512, trust_remote_code=False,
        )

    def predict(self, pairs):
        import torch
        return self.model.predict(pairs, batch_size=16, show_progress_bar=False,
                                  activation_fn=torch.nn.Softmax(dim=-1)).tolist()


@lru_cache(maxsize=2)
def get_scorer(backend="torch"):
    return LocalNLI(backend)
