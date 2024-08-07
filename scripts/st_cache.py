import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class CachedSentenceTransformer:
    def __init__(self, model_name, model, cache):
        self.name = model_name
        self.model = model
        self.cache = cache

    def __call__(self, sentences, *args, **kwargs):
        cache = self.cache[self.name]
        new = pd.Series(index=np.unique(sentences))
        if cache is not None:
            new = new[new.index.difference(cache.index)]
        if len(new)>0:
            encodings = self.model.encode(new.index.tolist(), *args, **kwargs)
            new = pd.DataFrame(encodings, index=new.index)
            cache = pd.concat([cache, new])
        self.cache[self.name] = cache
        return cache.loc[sentences]
    
class SentenceTransformerCache:
    def __init__(self):
        self.models = {}
        self.cache = {}

    def __call__(self, model_name):
        model = self.models.get(model_name, SentenceTransformer(model_name))
        self.models[model_name] = model
        self.cache[model_name] = self.cache.get(model_name)
        return CachedSentenceTransformer(model_name, model, self.cache)
    
SENTENCE_TRANSFORMER_CACHE = SentenceTransformerCache()