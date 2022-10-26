from typing import cast
import torch
from torch import nn, Tensor
import nltk

from transformers import T5EncoderModel

def ensure_framenet_downloaded() -> None:
    try:
        nltk.data.find("corpora/framenet_v17")
    except LookupError:
        nltk.download("framenet_v17")

class T5Encoder(nn.Module):
    _model: T5EncoderModel
    model_path: str
    model_revision: str
    device: torch.device
    max_batch_size: int
    predictions_per_sample: int
    max_length: int

    def __init__(self,
                 model_name:str = 'base',
                 device: torch.device = "cpu",
                 max_batch_size: int = 16,
                 predictions_per_sample: int = 5,
                 max_length = 100,
        ):
        super(T5Encoder, self).__init__()
        self.model_path = f"chanind/frame-semantic-transformer-{model_name}"
        self.model_revision = "v0.1.0"
        self.device = device
        self.max_batch_size = max_batch_size
        self.predictions_per_sample = predictions_per_sample
        self.max_length = max_length

        self.setup()


    def setup(self) -> None:
        """
        Initialize the model and tokenizer, and download models / files as needed
        If this is not called explicitly it will be lazily called before inference
        """

        self._model = T5EncoderModel.from_pretrained(
            self.model_path, revision=self.model_revision
        ).to(self.device)

        ensure_framenet_downloaded()

    @property
    def model(self) -> T5EncoderModel:
        if not self._model:
            self.setup()
        return cast(T5EncoderModel, self._model)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
    ) -> torch.FloatTensor:
        outputs = self._model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
        )

        return outputs.last_hidden_state