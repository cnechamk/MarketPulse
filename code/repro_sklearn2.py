import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_curve
from sklearn.decomposition import PCA

ENCODE_NAME = 'philschmid/bge-base-financial-matryoshka'

def encode_docs(docs):
    encoder = SentenceTransformer(ENCODE_NAME)
    X = encoder.encode(docs.tolist(), show_progress_bar=True)
    return X

def multioutput_scorer(scorer, y_true, y_pred, **kwargs):
    def f(series, scorer, **kwargs):
        score = scorer(series['y_true'], series['y_pred'])
        try:
            ci = score.confidence_interval()
        except AttributeError:
            return {'Score':score}
        else:
            result = {'Score':score[0]}
            result.update(zip(['low', 'high'],ci))
            # print(result)
            return pd.Series(result)
    y_pred = pd.DataFrame(y_pred, index=y_true.index, columns=y_true.columns)
    df = pd.concat([y_true, y_pred], keys=['y_true', 'y_pred'])
    results = df.apply(f, scorer=scorer, result_type='expand', **kwargs)
    return results.T

def get_model():
    reg=LinearRegression()
    model = make_pipeline(
        PCA(
                16,
                # whiten=True
            ),
        reg
    )
    return model

def get_optimal_threshold(y, p):
    fpr, tpr, thresholds = roc_curve(y>0, p, drop_intermediate=True)
    opt_idx = np.argmax(tpr-fpr)
    # print(opt_idx)
    return thresholds[opt_idx]

def main(docs):
    X


if __name__ == "__main__":
    docs = pd.read_parquet('data/sts_gb.parquet').set_index('date').squeeze()#['statement']
    X = encode_docs(docs)

    y = pd.read_parquet('data/prices_min.parquet')
    y = y.asfreq('D').bfill().loc[docs.index]
    y = y.dropna(axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=False)
    model = get_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, index=y_test.index, columns=y_test.columns)

    y_pred.to_csv("data/y_pred.csv")
    y_test.to_csv("data/y_true.csv")


