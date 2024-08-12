import os
import argparse
import pickle
import pandas as pd
from sklearn.decomposition import PCA, NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline

from scripts.get_statements import get_statements
from scripts.get_prices import get_prices
from scripts.sklearn_classes import EncoderTransformer, SentenceSelector, Condition, Examples, Splitter
from scripts.st_cache import SENTENCE_TRANSFORMER_CACHE



parser = argparse.ArgumentParser('Fit Markepulse Models')
parser.add_argument('symbols', help='csv file with symbols and optionally their full names')
parser.add_argument('start', help='first date')
parser.add_argument('end', help='last date')
parser.add_argument('--seed', help='random seed', default=0)
parser.add_argument('--model', help='SentenceTransformer model name', default='philschmid/bge-base-financial-matryoshka')
parser.add_argument('--save', help='save datasets', default=True)

def mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


if __name__ == '__main__':

    args = parser.parse_args()

    if args.save:
        mkdir('./data')

    docs = get_statements(args.start, args.end)

    prices = get_prices(args.symbols, 40, 20, docs.index.min(), docs.index.max())

    selector = SentenceSelector(
        Splitter(), 
        SENTENCE_TRANSFORMER_CACHE(args.model), 
        Condition(),
        Examples(), 
        KNeighborsRegressor(5)
    )

    tfidf_model = make_pipeline(
        selector,
        TfidfVectorizer(min_df=.05, max_df=.5),
        NMF(6, max_iter=1000, random_state=args.seed),
        LinearRegression()
    )

    transformer_model = make_pipeline(
        selector,
        EncoderTransformer(args.model),
        PCA(24),
        LinearRegression()
    )

    y = prices.loc[docs.index]

    X_train, X_test, y_train, y_test = train_test_split(docs, y, test_size=.4, random_state=args.seed)

    if args.save:
        X_train.to_parquet('./data/train_statements.parquet')
        X_test.to_parquet('./data/test_statements.parquet')
        prices.to_parquet('./data/prices.parquet')

    tfidf_model.fit(X_train, y_train)

    transformer_model.fit(X_train, y_train)

    mkdir('./models')

    with open('./models/tfidf.pkl', 'wb') as f:
        pickle.dump(tfidf_model, f)

    with open('./models/transformer.pkl', 'wb') as f:
        pickle.dump(transformer_model, f)