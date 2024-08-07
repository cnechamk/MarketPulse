import numpy as np
import pandas as pd
from sklearn.base import(
    BaseEstimator, 
    ClassifierMixin,
    ClassNamePrefixFeaturesOutMixin, 
    MetaEstimatorMixin, 
    OneToOneFeatureMixin, 
    TransformerMixin
)
from sklearn.metrics import balanced_accuracy_score, roc_curve
from scripts.st_cache import SENTENCE_TRANSFORMER_CACHE


class EncoderTransformer(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    
    def __init__(self, model_name):

        self.model_name = model_name
    
    def fit(self, X, y=None):

        self.encoder = SENTENCE_TRANSFORMER_CACHE(self.model_name)
        return self
    
    def transform(self, X):
        
        return self.encoder(X.tolist())
    
class SentenceSelector(OneToOneFeatureMixin, TransformerMixin, MetaEstimatorMixin, BaseEstimator):

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
        X = X.reset_index(drop=True).rename_axis('group')
        sents = self.splitter(X)
        sents = sents[self.condition(sents)]

        return sents
    
    def get_labels(self, sents):
        
        labels = self.examples(sents)
        labels = labels.where(labels.groupby('group').any(), np.nan)

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
        idx = preds.reset_index().groupby('group').idxmax()
        out = sents.reset_index(drop=True)[idx.squeeze()]
        
        return out

class ClassifiedRegressor(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    def __init__(self, estimator, threshold=True):
        self.estimator = estimator
        self.threshold = threshold

    def fit(self, X, y):
        X = self.to_numpy(X)
        y = self.to_numpy(y)
        self.estimator.fit(X, y)
        self.get_thresholds(X, y)
        return self
    
    def to_numpy(self, arr):
        try:
            return arr.values
        except AttributeError:
            return arr
        
    def get_thresholds(self, X, y):
        y = (y>0).T
        p = self.estimator.predict(X).T
        self.threshold_ = np.zeros(y.shape[0])
        if self.threshold:
            for i, values in enumerate(zip(y, p)):
                optimal = self.get_optimal_threshold(values)
                self.threshold_[i] = optimal

    def get_optimal_threshold(self, values):
        fpr, tpr, thresholds = roc_curve(*values)
        opt_idx = np.argmax(tpr-fpr)
        
        return thresholds[opt_idx]
    
    def predict(self, X):
        preds = self.decision_function(X)
        return preds>self.threshold_
    
    def decision_function(self, X):
        return self.estimator.predict(X)
    
    def score(self, X, y):
        preds = self.predict(X).ravel()
        y = self.to_numpy(y).ravel()>0
        return balanced_accuracy_score(y, preds)