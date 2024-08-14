import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import shap
from scipy.stats import pearsonr
from sklearn.metrics import balanced_accuracy_score

from scripts.metrics import multioutput_scorer

try:
    os.mkdir("/plots")
except FileExistsError:
    pass


def accuracy_plot(X, y, models, threshold=0):

    y = y > threshold

    scores = {}
    for name, model in models.items():
        preds = model.predict(X) > threshold
        score = multioutput_scorer(balanced_accuracy_score, y, preds)
        scores[name] = score

    results = pd.concat(scores, names=["Model", "Day", "Ticker"])
    results = results.reset_index().sort_values(["Model", "Day"])

    grid = sns.relplot(
        results, y="Score", x="Day", hue="Model", kind="line", col="Ticker", col_wrap=3
    )

    for ticker, grp in results.groupby("Ticker"):

        ax = grid.axes_dict[ticker]
        ax.hlines(0.5, grp["Day"].min(), grp["Day"].max(), color="black")

    grid.fig.suptitle("Balanced Accuracy Score")
    grid.fig.set_constrained_layout(True)

    grid.fig.savefig("accuracy_plot.png")

    return grid


def correlation_plot(X, y, models):
    scores = {}
    for name, model in models.items():
        preds = model.predict(X)
        score = multioutput_scorer(pearsonr, y, preds)
        scores[name] = score

    results = pd.concat(scores, names=["Model", "Day", "Ticker"])
    results = results.reset_index().sort_values(["Model", "Day"])

    grid = sns.relplot(
        results, y="Score", x="Day", hue="Model", kind="line", col="Ticker", col_wrap=3
    )

    for ticker, grp in results.groupby("Ticker"):

        ax = grid.axes_dict[ticker]
        ax.hlines(0, grp["Day"].min(), grp["Day"].max(), color="black")

        cmap = sns.color_palette()
        for i, (model_name, model_df) in enumerate(grp.groupby("Model")):
            ax.fill_between(
                "Day", "low", "high", data=model_df, color=cmap[i], alpha=0.1
            )
            ax.plot("Day", "high", data=model_df, color=cmap[i], alpha=0.15, ls="--")
            ax.plot("Day", "low", data=model_df, color=cmap[i], alpha=0.15, ls="--")

    grid.fig.suptitle("Spearman Correlation with Confidence Intervals")
    grid.fig.set_constrained_layout(True)

    grid.fig.savefig("correlation_plot.png")

    return grid


def wordclouds(vocab, features, coefs):

    vocab = sorted(vocab, key=lambda w: vocab[w])
    frequencies = pd.DataFrame(features.T, vocab)
    n_rows = int(np.ceil(len(features) / 3).item())
    fig, axs = plt.subplots(n_rows, 3, layout="constrained")
    axs = axs.ravel()

    for i in range(len(features)):

        ax = axs[i]
        wc = WordCloud(random_state=i * 10, colormap="Set2")
        topic_freqs = frequencies[i].sort_values()
        topic_freqs = topic_freqs[topic_freqs > 0]
        wc.generate_from_frequencies(topic_freqs)
        ax.imshow(wc)
        ax.set_title(f"Topic {i}")
        ax.set_axis_off()

    fig.set_figwidth(15)
    fig.suptitle("Topic Wordclouds")

    fig.savefig("wordclouds.png")

    return fig


def topic_bar_plot(coefs, tickers):

    df = pd.DataFrame(coefs, tickers).groupby("Ticker").mean().T
    df.index = [f"Topic {i}" for i in range(len(df))]
    df.index.name = "Topic"
    df = df[tickers.unique()]
    df = df.melt(ignore_index=False).reset_index()

    grid = sns.catplot(
        df,
        x="Topic",
        y="value",
        hue="Topic",
        col="Ticker",
        col_wrap=3,
        kind="bar",
        sharey=False,
        sharex=False,
    )

    grid.fig.suptitle("Coeficient of each Topic")
    grid.fig.set_constrained_layout(True)

    grid.fig.savefig("topic_bar_plot.png")

    return grid


def topic_plot(tfidf, nmf, lin_reg, prices):

    vocab = tfidf.vocabulary_
    features = nmf.components_
    coefs = lin_reg.coef_
    tickers = prices.columns.get_level_values(1)

    wordcloud = wordclouds(vocab, features, coefs)
    bar_plot = topic_bar_plot(coefs, tickers)

    return wordcloud, bar_plot


class ShapModel:
    def __init__(self, model, period, y):

        self.model = model
        self.period = period
        self.colummns = y.columns

    def __call__(self, X):

        preds = self.model[1:].predict(X)
        preds = pd.DataFrame(preds, columns=self.colummns)

        return preds[self.period]


def shap_plot(model, X, y, dates, period, mask_pat=r"\W"):

    masker = shap.maskers.Text(mask_pat)

    dates = sorted(pd.to_datetime(dates))
    examples = X[X.index.isin(dates)].squeeze()
    examples = model[0].transform(examples)

    ticker_names = y[y.columns.get_level_values(0)[0]].columns

    shap_model = ShapModel(model, period, y)
    explainer = shap.Explainer(shap_model, masker, output_names=ticker_names)
    shap_values = explainer(examples, silent=True)

    for i, date in enumerate(dates):
        print(date.strftime("%B %d, %Y"))
        shap.plots.text(shap_values[i])
        
        plot = shap.plots.text(shap_values[i], display=False)
        with open(f"./plots/{date}.html", "w") as f:
            f.write(plot)

    return shap_values
