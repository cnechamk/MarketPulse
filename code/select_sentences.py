# %%
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.neighbors import KNeighborsClassifier
from sentence_transformers import SentenceTransformer

# %%
def process_html(doc):
    return BeautifulSoup(doc).text

# %%
statements_df = pd.read_parquet('../data/Monetary_Policy.parquet').set_index('id', drop=False)
statements_df['date'] = pd.to_datetime(statements_df.date)

# %%
statements_df = statements_df[statements_df.description.str.contains('FOMC statement')]

# %%
statements = statements_df.doc.apply(
        process_html
    ).str.replace(
        r'\r\n', ' ', regex=True
    ).str.split(
        r'\n', regex=True
    ).explode(
    ).str.strip(
    ).str.replace(
        r'\s+', ' ', regex=True
    )
conds =[
    statements!='',
    ~statements.str.contains(r'^Voting for'),
    ~statements.str.contains('[email protected]', regex=False),
    ~statements.str.lower().str.contains(r'implementation note issued.*\d'),
    statements.str.split(' ').apply(len)>15,
]
statements = statements[pd.concat(conds, axis=1).all(axis=1)]

# %%
# statements.str.split(' ').apply(len).plot.hist(bins=30)

# %%
ix = (
    (statements.str.lower().str.contains('the committee decided')\
    & statements.str.contains(r'\d\spercent', regex=True))
    | statements.str.contains('Federal Reserve Actions')\
)

# %%
statements = pd.concat([statements,ix], axis=1, keys=['statement', 'y'])
statements['y'] = statements['y'].where(statements.groupby('id')['y'].any(), np.nan).astype(float)

# %%
model_name = 'philschmid/bge-base-financial-matryoshka'
encoder = SentenceTransformer(model_name)
encodings = encoder.encode(statements['statement'].tolist())

# %%
encodings = pd.read_parquet('../data/sts_enc.parquet')

# %%
statements=statements.reset_index()
mask = statements['y'].isna()
knn = KNeighborsClassifier(5)
knn.fit(encodings[~mask], statements[~mask]['y'])
statements.loc[mask, 'y'] = knn.predict_proba(encodings[mask])[:,1]
statements = statements[statements['y'] == statements.groupby('id')['y'].transform('max')]
statements = statements.groupby('id')['statement'].agg('\n'.join)

# %%
pd.concat([statements_df['date'], statements], axis=1).to_parquet('../data/sts_gb.parquet')

# %%
# encodings2 = encoder.encode(statements.tolist())
# pd.DataFrame(encodings2, index=statements.index).to_parquet('../data/sts_gb_enc.parquet')


