import os
import warnings
import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from sentence_transformers import SentenceTransformer

if "SENTENCE_TRANSFORMER_CACHE_DIR" in os.environ:
    CACHE_DIR = os.environ["SENTENCE_TRANSFORMER_CACHE_DIR"]
else:
    try:
        os.mkdir("./data")
    except FileExistsError:
        pass
    CACHE_DIR = "./data/"
    os.environ["SENTENCE_TRANSFORMER_CACHE_DIR"] = CACHE_DIR


class CachedSentenceTransformer:
    """A wrapper for a sentence transformers model that
    chaches each item passed to the model"""

    def __init__(self, model_name, model, cache):
        self.name = model_name
        self.model = model
        self.cache = cache

    def __call__(self, sentences, *args, **kwargs):
        cache = self.cache[self.name]
        new = pd.Series(index=np.unique(sentences))
        if cache is not None:
            new = new[new.index.difference(cache.index)]
        if len(new) > 0:
            encodings = self.model.encode(new.index.tolist(), *args, **kwargs)
            new = pd.DataFrame(encodings, index=new.index)
            cache = pd.concat([cache, new])

        self.cache[self.name] = cache

        return cache.loc[sentences]

    def __repr__(self):
        return self.name


class SentenceTransformerCache:

    def __init__(self, dir):
        dir = dir.rstrip("/") + "/"
        self.dir = dir + ".cache"
        try:
            os.mkdir(self.dir)
        except FileExistsError:
            pass
        self.models = {}
        self.cache = {}

    def __call__(self, model_name):
        model = self.models.get(model_name, SentenceTransformer(model_name))
        self.models[model_name] = model
        if model_name not in self.cache:
            file = f"{self.dir}/{model_name.replace('/', '-')}.parquet"
            try:
                cache = pd.read_parquet(file)
            except FileNotFoundError:
                cache = None
        else:
            cache = None
        self.cache[model_name] = cache

        return CachedSentenceTransformer(model_name, model, self)

    def __getitem__(self, model_name):
        return self.cache[model_name]

    def __setitem__(self, model_name, df):

        cache = self.cache[model_name]

        if cache is None:

            cache = df

        else:

            new = df.loc[df.index.difference(cache.index)]
            cache = pd.concat([cache, new])
            file = f"{self.dir}/{model_name.replace('/', '-')}.parquet"
            cache.to_parquet(file)

        self.cache[model_name] = cache


SENTENCE_TRANSFORMER_CACHE = SentenceTransformerCache(CACHE_DIR)
