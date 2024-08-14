import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import balanced_accuracy_score, r2_score


def multioutput_scorer(scorer, y_true, y_pred, **kwargs):
    """Scores multioutput predictions giving one score per column"""

    def f(series, scorer, **kwargs):

        score = scorer(series["y_true"], series["y_pred"])
        try:
            ci = score.confidence_interval()
        except AttributeError:
            return {"Score": score}
        else:
            result = {"Score": score[0]}
            result.update(zip(["low", "high"], ci))

            return pd.Series(result)

    y_pred = pd.DataFrame(y_pred, index=y_true.index, columns=y_true.columns)
    df = pd.concat([y_true, y_pred], keys=["y_true", "y_pred"])
    results = df.apply(f, scorer=scorer, result_type="expand", **kwargs)

    return results.T


def get_scores(y_true, y_pred, threshold=0):
    """Get balanced accuracy, pearson correlation,
    and scaled mean squared error for multioutput
    predictions"""

    acc = multioutput_scorer(balanced_accuracy_score, y_true > 0, y_pred > threshold)
    acc = acc.unstack().mean()["Score"].rename("acc")
    pearson = multioutput_scorer(pearsonr, y_true, y_pred)
    pearson = pearson.unstack().mean()["Score"].rename("pearson")
    r2 = multioutput_scorer(r2_score, y_true, y_pred)
    r2 = r2.unstack().mean()["Score"].rename("r2")

    return pd.concat([acc, r2, pearson], axis=1)
