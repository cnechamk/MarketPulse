import numpy as np
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    MetaEstimatorMixin,
    OneToOneFeatureMixin,
    TransformerMixin,
)
from scripts.st_cache import SENTENCE_TRANSFORMER_CACHE


class EncoderTransformer(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
):

    def __init__(self, model_name):

        self.model_name = model_name

    def fit(self, X, y=None):

        self.encoder = SENTENCE_TRANSFORMER_CACHE(self.model_name)
        return self

    def transform(self, X):

        return self.encoder(X.tolist())


class SentenceSelector(
    OneToOneFeatureMixin, TransformerMixin, MetaEstimatorMixin, BaseEstimator
):
    """1. Splits documents using "splitter"
    2. Encodes the sentences using "encoder"
    3. Filters sentences using "condition"
    4. Marks sentences as positive based on "examples"
    5. Filters sentences based on the highest score from estimator"""

    def __init__(self, splitter, encoder, condition, examples, estimator):

        self.splitter = splitter
        self.encoder = encoder
        self.condition = condition
        self.examples = examples
        self.estimator = estimator

    def get_sents(self, X):
        try:
            X = pd.Series(X)
        except:
            X = X.squeeze()
        X = X.reset_index(drop=True).rename_axis("group")
        sents = self.splitter(X)
        sents = sents[self.condition(sents)]

        return sents

    def get_labels(self, sents):

        labels = self.examples(sents)
        labels = labels.where(labels.groupby("group").any(), np.nan)

        return labels

    def fit(self, X, y=None):

        sents = self.get_sents(X)
        labels = self.get_labels(sents)
        mask = labels.notna().values
        if callable(self.encoder):
            encodings = self.encoder(sents[mask].tolist())
        else:
            encodings = self.encoder.fit_transform(sents[mask])
        self.estimator.fit(encodings, labels[mask])

        return self

    def transform(self, X):

        sents = self.get_sents(X)
        if callable(self.encoder):
            encodings = self.encoder(sents.tolist())
        else:
            encodings = self.encoder.transform(sents)
        preds = self.estimator.predict(encodings)

        preds = pd.Series(preds, sents.index)
        idx = preds.reset_index().groupby("group").idxmax()
        out = sents.reset_index(drop=True)[idx.squeeze()]

        return out


# The following 3 classes are the ones used for their respective arguments of SentenceSelector


class Splitter:
    def __call__(self, ser):
        return (
            ser.str.replace(r"\r\n", " ", regex=True)
            .str.split(r"\n", regex=True)
            .explode()
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    def __repr__(self):
        return "Split by line"


class Condition:
    def __call__(self, ser):
        conds = [
            ser != "",
            ~ser.str.contains(r"^Voting for"),
            ~ser.str.contains("[email protected]", regex=False),
            ~ser.str.lower().str.contains(r"implementation note issued.*\d"),
            ser.str.split(" ").apply(len) > 15,
        ]
        return pd.concat(conds, axis=1).all(axis=1)

    def __repr__(self):
        return "Exclude voting, email, notes and sentences with less than 15 words"


class Examples:
    def __call__(self, ser):
        return (
            ser.str.lower().str.contains("the committee decided")
            & ser.str.contains(r"\d\spercent", regex=True)
        ) | ser.str.contains("Federal Reserve Actions")

    def __repr__(self):
        return "Contains ('the committee decided' & '{x}percent') | 'Federal Reserve Actions'"
