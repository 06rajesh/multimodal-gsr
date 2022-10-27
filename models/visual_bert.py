import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import warnings

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, KLDivLoss, LogSoftmax
from transformers import BertModel

from models.dual_encoder_gsr.t5_encoder import T5Encoder

class VisualBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings and visual embeddings."""

    def __init__(self, config, bert_model_name="bert-base-uncased"): #word_embedding_weight=None
        super().__init__()

        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.bert_model.requires_grad_(False)

        # self.word_embeddings = nn.Embedding(config.vocab_size, config.word_embedding_dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids",
                             torch.arange(config.max_position_embeddings)
                             .expand((1, -1)).repeat(config.batch_size, 1)
                             )

        # if word_embedding_weight != None:
        #     self.word_embeddings.weight.data.copy_(word_embedding_weight)
        #     self.word_embeddings.requires_grad_(False)

        # For Visual Features
        # Token type and position embedding for image features
        self.visual_token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.visual_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        if config.special_visual_initialize:
            self.visual_token_type_embeddings.weight.data = nn.Parameter(
                self.token_type_embeddings.weight.data.clone(), requires_grad=True
            )
            self.visual_position_embeddings.weight.data = nn.Parameter(
                self.position_embeddings.weight.data.clone(), requires_grad=True
            )

        self.word_embedding_projection = nn.Linear(config.word_embedding_dim, config.hidden_size)
        self.visual_projection = nn.Linear(config.visual_embedding_dim, config.hidden_size)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        visual_embeds=None,
        visual_token_type_ids=None,
        image_text_alignment=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        n_inputs = input_shape[0]
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:n_inputs, :seq_length]

        if inputs_embeds is None:
            bert_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            inputs_embeds = bert_outputs.last_hidden_state
            # inputs_embeds = self.word_embeddings(input_ids)
            inputs_embeds = self.word_embedding_projection(inputs_embeds)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        token_type_embeds = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeds

        # Absolute Position Embeddings
        position_embeddings = self.position_embeddings(position_ids)
        final_pos_embeddings = position_embeddings

        if visual_embeds is not None:
            if visual_token_type_ids is None:
                visual_token_type_ids = torch.ones(
                    visual_embeds.size()[:-1], dtype=torch.long, device=self.position_ids.device
                )

            visual_embeds = self.visual_projection(visual_embeds)
            visual_token_type_embeddings = self.visual_token_type_embeddings(visual_token_type_ids)

            if image_text_alignment is not None:
                # image_text_alignment = Batch x image_length x alignment_number.
                # Each element denotes the position of the word corresponding to the image feature. -1 is the padding value.

                dtype = token_type_embeds.dtype
                image_text_alignment_mask = (image_text_alignment != -1).long()
                # Get rid of the -1.
                image_text_alignment = image_text_alignment_mask * image_text_alignment

                # Batch x image_length x alignment length x dim
                visual_position_embeddings = self.position_embeddings(image_text_alignment)
                visual_position_embeddings *= image_text_alignment_mask.to(dtype=dtype).unsqueeze(-1)
                visual_position_embeddings = visual_position_embeddings.sum(2)

                # We want to averge along the alignment_number dimension.
                image_text_alignment_mask = image_text_alignment_mask.to(dtype=dtype).sum(2)

                if (image_text_alignment_mask == 0).sum() != 0:
                    image_text_alignment_mask[image_text_alignment_mask == 0] = 1  # Avoid divide by zero error
                    warnings.warn(
                        "Found 0 values in `image_text_alignment_mask`. Setting them to 1 to avoid divide-by-zero"
                        " error."
                    )
                visual_position_embeddings = visual_position_embeddings / image_text_alignment_mask.unsqueeze(-1)

                visual_position_ids = torch.zeros(
                    *visual_embeds.size()[:-1], dtype=torch.long, device=visual_embeds.device
                )

                # When fine-tuning the detector , the image_text_alignment is sometimes padded too long.
                if visual_position_embeddings.size(1) != visual_embeds.size(1):
                    if visual_position_embeddings.size(1) < visual_embeds.size(1):
                        raise ValueError(
                            f"Visual position embeddings length: {visual_position_embeddings.size(1)} "
                            f"should be the same as `visual_embeds` length: {visual_embeds.size(1)}"
                        )
                    visual_position_embeddings = visual_position_embeddings[:, : visual_embeds.size(1), :]

                visual_position_embeddings = visual_position_embeddings + self.visual_position_embeddings(
                    visual_position_ids
                )
            else:
                visual_position_ids = torch.zeros(
                    *visual_embeds.size()[:-1], dtype=torch.long, device=visual_embeds.device
                )
                visual_position_embeddings = self.visual_position_embeddings(visual_position_ids)

            visual_embeddings = visual_embeds + visual_token_type_embeddings

            embeddings = torch.cat((embeddings, visual_embeddings), dim=1)

            final_pos_embeddings = torch.cat((position_embeddings, visual_position_embeddings), dim=1)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, final_pos_embeddings



class VisualT5Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings and visual embeddings."""

    def __init__(self, config, t5_model_name="base"): #word_embedding_weight=None
        super().__init__()

        self.t5_model = T5Encoder(
            model_name=t5_model_name,
            device=config.device,
            max_batch_size=config.batch_size,
            max_length=config.max_position_embeddings
        )
        self.t5_model.requires_grad_(False)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids",
                             torch.arange(config.max_position_embeddings)
                             .expand((1, -1)).repeat(config.batch_size, 1)
                             )


        # For Visual Features
        # Token type and position embedding for image features
        self.visual_token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.visual_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        if config.special_visual_initialize:
            # self.visual_token_type_embeddings.weight.data = nn.Parameter(
            #     self.token_type_embeddings.weight.data.clone(), requires_grad=True
            # )
            self.visual_position_embeddings.weight.data = nn.Parameter(
                self.position_embeddings.weight.data.clone(), requires_grad=True
            )

        self.word_embedding_projection = nn.Linear(config.word_embedding_dim, config.hidden_size)
        self.visual_projection = nn.Linear(config.visual_embedding_dim, config.hidden_size)

    @property
    def text_encoder(self):
        """I'm the 'x' property."""
        print("getter of x called")
        return self.t5_model

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        visual_embeds=None,
        visual_token_type_ids=None,
        image_text_alignment=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        n_inputs = input_shape[0]
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:n_inputs, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.t5_model(input_ids=input_ids, attention_mask=attention_mask)
            inputs_embeds = self.word_embedding_projection(inputs_embeds)


        embeddings = inputs_embeds

        # Absolute Position Embeddings
        position_embeddings = self.position_embeddings(position_ids)
        final_pos_embeddings = position_embeddings

        if visual_embeds is not None:
            if visual_token_type_ids is None:
                visual_token_type_ids = torch.ones(
                    visual_embeds.size()[:-1], dtype=torch.long, device=self.position_ids.device
                )

            visual_embeds = self.visual_projection(visual_embeds)
            visual_token_type_embeddings = self.visual_token_type_embeddings(visual_token_type_ids)

            if image_text_alignment is not None:
                # image_text_alignment = Batch x image_length x alignment_number.
                # Each element denotes the position of the word corresponding to the image feature. -1 is the padding value.

                dtype = inputs_embeds.dtype
                image_text_alignment_mask = (image_text_alignment != -1).long()
                # Get rid of the -1.
                image_text_alignment = image_text_alignment_mask * image_text_alignment

                # Batch x image_length x alignment length x dim
                visual_position_embeddings = self.position_embeddings(image_text_alignment)
                visual_position_embeddings *= image_text_alignment_mask.to(dtype=dtype).unsqueeze(-1)
                visual_position_embeddings = visual_position_embeddings.sum(2)

                # We want to averge along the alignment_number dimension.
                image_text_alignment_mask = image_text_alignment_mask.to(dtype=dtype).sum(2)

                if (image_text_alignment_mask == 0).sum() != 0:
                    image_text_alignment_mask[image_text_alignment_mask == 0] = 1  # Avoid divide by zero error
                    warnings.warn(
                        "Found 0 values in `image_text_alignment_mask`. Setting them to 1 to avoid divide-by-zero"
                        " error."
                    )
                visual_position_embeddings = visual_position_embeddings / image_text_alignment_mask.unsqueeze(-1)

                visual_position_ids = torch.zeros(
                    *visual_embeds.size()[:-1], dtype=torch.long, device=visual_embeds.device
                )

                # When fine-tuning the detector , the image_text_alignment is sometimes padded too long.
                if visual_position_embeddings.size(1) != visual_embeds.size(1):
                    if visual_position_embeddings.size(1) < visual_embeds.size(1):
                        raise ValueError(
                            f"Visual position embeddings length: {visual_position_embeddings.size(1)} "
                            f"should be the same as `visual_embeds` length: {visual_embeds.size(1)}"
                        )
                    visual_position_embeddings = visual_position_embeddings[:, : visual_embeds.size(1), :]

                visual_position_embeddings = visual_position_embeddings + self.visual_position_embeddings(
                    visual_position_ids
                )
            else:
                visual_position_ids = torch.zeros(
                    *visual_embeds.size()[:-1], dtype=torch.long, device=visual_embeds.device
                )
                visual_position_embeddings = self.visual_position_embeddings(visual_position_ids)

            visual_embeddings = visual_embeds + visual_token_type_embeddings

            embeddings = torch.cat((embeddings, visual_embeddings), dim=1)

            final_pos_embeddings = torch.cat((position_embeddings, visual_position_embeddings), dim=1)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, final_pos_embeddings