#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : word2vec.py
# Author            : Wan Li
# Date              : 15.08.2019
# Last Modified Date: 15.08.2019
# Last Modified By  : Wan Li

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.word2vec import LineSentence


def learn(fn_sentences, num_dims, window_size):
    """
        Learn word2vec embedding model
        Params:
            fn_sentences: file name of file with format:
                one sentence = one line
                words already preprocessed and separated by whitespace.
            num_dims: embedding dimension number
            window_size: skip window size
    """
    model = Word2Vec(LineSentence(fn_sentences),
                     size=num_dims, window=window_size,
                     min_count=5, sg=1, workers=5, iter=5)
    return model


def search(wv, items, topk=10):
    """
        Search top-k most similar items in cosin space
        Params:
            wv: KeyedVectors instance (or Word2Vec.wv)
            items: array of string/int identifiers
            topk: number of most similar results
        Return:
            [(identifier, score), ..]
            score ranged (0, 1]
    """
    items = wv.most_similar(positive=items, topn=topk)
    return items


def save(model, fn_embedding, fn_vocab=None):
    """
        Saving a model using original Google’s word2vec C format
    """
    model.wv.save_word2vec_format(fn_embedding, fvocab=fn_vocab)


def load(fn_embedding):
    """
        Load file with Google’s word2vec C format as a KeyedVectors instance
    """
    word_vectors = KeyedVectors.load_word2vec_format(fn_embedding, binary=False)
    return word_vectors
