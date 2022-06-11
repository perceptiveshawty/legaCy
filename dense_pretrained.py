import tensorflow as tf
import tensorflow_hub as hub

from chainer.dataset import tabular
from chainer.iterators.serial_iterator import SerialIterator

import pickle
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer, models, datasets, losses
from simcse import SimCSE

from torch.utils.data import DataLoader

def use(instances):
    encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    dataset = tabular.from_data(instances)
    dataloader = SerialIterator(dataset, 10, shuffle=False)

    embeddings = []
    for batch in dataloader:
        embeddings.append(np.array(encoder(batch)))

    with open('embeddings/use.pkl', 'wb') as d:
        pickle.dump(np.vstack(embeddings), d)

def declutr(instances):
    encoder = SentenceTransformer('johngiorgi/declutr-base')
    dataset = tabular.from_data(instances)
    dataloader = SerialIterator(dataset, 10, shuffle=False)

    embeddings = []
    for batch in dataloader:
        embeddings.append(np.array(encoder(batch)))

    with open('embeddings/declutr.pkl', 'wb') as d:
        pickle.dump(np.vstack(embeddings), d)


def simcse(instances):
    encoder = SimCSE('princeton-nlp/sup-simcse-roberta-base')
    embeddings = encoder.encode(instances, return_numpy=True, batch_size=10)

    with open('embeddings/simcse.pkl', 'wb') as d:
        pickle.dump(np.vstack(embeddings), d)

def tsdae(instances, base_transformer='nlpaueb/legal-bert-base-uncased'):
    word_embedding_model = models.Transformer(base_transformer)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'cls')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    train_dataset = datasets.DenoisingAutoEncoderDataset(instances)
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=base_transformer, tie_encoder_decoder=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        weight_decay=0,
        scheduler='constantlr',
        optimizer_params={'lr': 3e-5},
        show_progress_bar=True
    )

    dataset = tabular.from_data(instances)
    dataloader = SerialIterator(dataset, 10, shuffle=False)

    embeddings = []
    for batch in dataloader:
        embeddings.append(np.array(model.encode(batch)))

    with open('embeddings/tsdae_' + base_transformer + '.pkl', 'wb') as d:
        pickle.dump(np.vstack(embeddings), d)

