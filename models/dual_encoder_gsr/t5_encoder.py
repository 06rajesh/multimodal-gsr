from typing import cast
import torch
from torch import nn, Tensor
import nltk

from transformers import T5ForConditionalGeneration, T5Tokenizer

def ensure_framenet_downloaded() -> None:
    try:
        nltk.data.find("corpora/framenet_v17")
    except LookupError:
        nltk.download("framenet_v17")

class T5Encoder(nn.Module):
    _model: T5ForConditionalGeneration
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


    def setup(self) -> None:
        """
        Initialize the model and tokenizer, and download models / files as needed
        If this is not called explicitly it will be lazily called before inference
        """

        self._model = T5ForConditionalGeneration.from_pretrained(
            self.model_path, revision=self.model_revision
        ).to(self.device)
        # self._tokenizer = T5Tokenizer.from_pretrained(
        #     self.model_path,
        #     revision=self.model_revision,
        #     model_max_length=self.max_length,
        # )

        ensure_framenet_downloaded()

    # @property
    # def model(self) -> T5ForConditionalGeneration:
    #     if not self._model:
    #         self.setup()
    #     return cast(T5ForConditionalGeneration, self._model)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            num_return_sequences: int = 1,
            num_beams: int = 5,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 2.5,
            length_penalty: float = 1.0,
            early_stopping: bool = True,
    ) -> torch.FloatTensor:
        outputs = self._model.generate(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            num_beams=num_beams,
            max_length=self.max_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            output_hidden_states=True,
            output_attentions=True,
            return_dict_in_generate=True,
        )

        return outputs.encoder_hidden_states[-1], outputs.encoder_attentions[-1]