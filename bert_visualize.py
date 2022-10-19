
import matplotlib.pyplot as plt
import torch
import numpy as np

from sklearn.manifold import TSNE
from torch import nn
from transformers import BertTokenizer, VisualBertConfig, BertModel


class BertEmbeddingTest(nn.Module):

    def __init__(self, vocab_size, word_embedding_dim, hidden_size=512, word_embedding_weight=None):
        super(BertEmbeddingTest, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)

        if word_embedding_weight != None:
            self.word_embeddings.weight.data.copy_(word_embedding_weight)
            self.word_embeddings.requires_grad_(False)

        self.word_embedding_projection = nn.Linear(word_embedding_dim, hidden_size)

    def forward(self, input_ids=None):
        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)

        inputs_embeds = self.word_embeddings(input_ids)
        # inputs_embeds = self.word_embedding_projection(inputs_embeds)

        return inputs_embeds


def build_model():
    bertmodelname = "bert-base-uncased"
    bertmodel = BertModel.from_pretrained(bertmodelname)
    embedding_matrix = bertmodel.embeddings.word_embeddings.weight

    tokenizer = BertTokenizer.from_pretrained(bertmodelname)

    vbconfig = VisualBertConfig()
    vocab_size = vbconfig.vocab_size
    word_embedding_dim = embedding_matrix.shape[1]

    model = BertEmbeddingTest(vocab_size, word_embedding_dim, word_embedding_weight=embedding_matrix)

    return model, tokenizer


def tsne_plot(model, vocabs:dict):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in vocabs.keys():
        id = vocabs[word]
        output = model([id])[0]
        tokens.append(output.numpy())
        labels.append(word)

    tokens = np.array(tokens)

    tsne_model = TSNE(perplexity=20, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

if __name__ == '__main__':

    captions = [
        "The attacker is attacking with a gun.",
        "Mary and Samantha arrived at the bus station early but waited until noon for the bus.",
        "the people are skiing at the slope",
        "the group are skiing at the ski run",
        "the unit is skiing at the ski slope",
        "a causal agent is dialing the cellular telephone on the inside",
        "a man is dialing the telephone on the inside",
        "the skier is skiing at the slope",
        "a person is gluing with glue to decoration on the table",
        "the male child is gluing the tissue with glue to cartoon in the classroom",
        "the veterinarian is bandaging the horse in the stable",
        "a nurse is bandaging the equine on the leg",
        "a female is decorating the paint of a vase at the workshop",
        "a woman is decorating the paint of a vase in the room"
        "a man is sneezing in the room",
        "a adult is sneezing at the elbow",
        "an elephant is stampeding at the scrubland",
        "a man is clipping the hair with clipper from beard on the inside",
        "a man is clipping the beard with clipper from face at the barbershop",
        "a female is smearing the makeup with a hand over face on the inside",
        "the water scooter races in the ocean",
        "the worker operates a gauge with a hand at the outdoors",
        "the parent talks to a baby at the highchair",
        "the child talks to mother on the table",
        "the football player is tripping the real property at the football field",
        "a organism is disciplining the male child with a hand in the room",
        "the marching band is marching at the football field",
        "the group are congregating in the gallery",
        "a person is folding in paper in a triangle on the table",
    ]

    # model, tokenizer = build_model()
    #
    # vocabs = {}
    # for c in captions:
    #     inputs = tokenizer(c)
    #
    #     ids = inputs['input_ids']
    #     decoded = tokenizer.convert_ids_to_tokens(ids)
    #
    #     for i in range(len(decoded)):
    #         if decoded[i] != '[CLS]' and decoded[i] != '[SEP]':
    #             if decoded[i] not in vocabs:
    #                 vocabs[decoded[i]] = ids[i]
    #
    # print(vocabs)
    # tsne_plot(model, vocabs)

    bertmodelname = "bert-base-uncased"
    bertmodel = BertModel.from_pretrained(bertmodelname)

    tokenizer = BertTokenizer.from_pretrained(bertmodelname, model_max_length=25)

    # 1.Tokenize the sequence:
    tokens = tokenizer(
        captions[0:5],
        padding='max_length',
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )
    # print(tokens)

    outputs = bertmodel(**tokens)
    print(outputs.last_hidden_state.shape)
    print(outputs.pooler_output.shape)
