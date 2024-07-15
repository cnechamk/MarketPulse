#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
# from IPython.display import HTML

base_url = 'https://www.federalreserve.gov'


# In[2]:


df = pd.read_csv('./data/pressreleases.csv').reset_index(names='id')
df.index = pd.to_datetime(df.date)


# In[3]:


df = df[df['category']=='Monetary Policy']
df = df[df.index>='2008-01-01']


# In[4]:


include = (
    '/fomc/',
    '/monetarypolicy/fomc',
    '/newsevents/press',
    
    )
exclude =(
    'https://www.facebook.com/sharer', 
    'https://www.linkedin.com/shareArticle', 
    'https://twitter.com/share',
    '.pdf',
    'projtabl',
    'monetarypolicy/fomc.htm',
    'calendars',
    'faqs',
    'email',
    )


# In[5]:


class Collector:
    def __init__(self, df):
        self.df = df
        self.docs = {}
        self.failed = {}
        self.failed_link = {}
        self.link_docs = {}
        self.link_urls = {}
        self.prog = tqdm(total=len(self.df)-len(self.docs))

    def __call__(self, row):
        _, row = row
        try:
            self.collect(row)
            self.prog.update(1)
            return (row.id, None)
        except Exception as e:
            self.prog.update(1)
            return (row.id, e)
        
    def collect(self, row):
        if row.id in self.docs:
            return
        try:
            r = requests.get(row['link'])
            if not r.ok:
                self.failed[row.id] = r
                return
            soup = BeautifulSoup(r.content)
        except Exception as e:
            self.failed[row.id] = e
            return
        
        article = soup.find(attrs={'id':'article'})

        self.docs[row.id] = str(article)

        links = article.find_all(href=re.compile(''))

        link_urls = set()
        for link in links:
            link_url = link['href']
            if base_url[5:] not in link_url:
                link_url = base_url + link['href']
            if all((
                any((s in link_url for s in include)),
                not any((s in link_url for s in exclude)),
                link_url != base_url + '/fomc'
            )):
                link_urls.add(link_url)
        self.link_urls[row.id] = list(link_urls)
        link_articles = []
        for link_url in self.link_urls[row.id]:
            try:
                r = requests.get(link_url)
                if not r.ok:
                    self.failed_link[row.id] = (link_url, r)
                    continue
                link_soup = BeautifulSoup(r.content)
            except Exception as e:
                self.failed_link[row.id] = (link_url, e)
                continue
            link_article = link_soup.find(attrs={'id':'article'})
            if link_article is None:
                link_article = link_soup.find(attrs={'id':'content'})
                # if link_article:
                #     link_article = link_soup.find_all(attrs={'class':'row'})
                    # link_article = '\n'.join((a.text for a in link_article))
            if link_article is None:
                self.failed_link[row.id] = link_url
            else:
                link_articles.append(str(link_article))
            self.link_docs[row.id] = link_articles


# In[6]:


# c = Collector(df)
# for row in df.head().iterrows(): c(row)


# In[7]:


# HTML(c.docs[0])


# In[8]:


# HTML(c.link_docs[0][0])


# In[9]:


c = Collector(df)
with ThreadPoolExecutor(max_workers=8) as pool:
    success = list(pool.map(c, df.iterrows()))


# In[10]:


c.failed


# In[11]:


c.failed_link


# In[12]:


docs = pd.DataFrame(pd.Series(c.docs, name='doc'))


# In[13]:


links = pd.DataFrame.from_dict(c.link_docs, orient='index')
links.columns = [f'link_{i}' for i in links]
links['link_urls'] = c.link_urls


# In[14]:


df = df.merge(docs.join(links), left_on='id', right_index=True)


# In[15]:


df.to_parquet('./data/Monetary_Policy.parquet')


# In[ ]:




