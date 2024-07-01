#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
base_url = 'https://www.federalreserve.gov'


# In[2]:


def parse_events(events):
    for event in events:
        yield {
            'date':event.parent.time.text,
            'category':event.find('p',attrs={'class':re.compile('')}).em.text,
            'link':'https://www.federalreserve.gov' + event.find('a', attrs={'href':re.compile('')})['href'],
            'description':event.find('a', attrs={'href':re.compile('')}).em.text
        }


# In[3]:


response = requests.get(base_url + '/newsevents/pressreleases.htm')
soup = BeautifulSoup(response.content)


# In[4]:


all_events = []
for year_tag in soup.find(attrs={'id':'article'}).find_all('a'):
    year_url = year_tag['href']
    if 'archive' not in year_url:
        response = requests.get(base_url + year_url)
        year_soup = BeautifulSoup(response.content)
        events = year_soup.find_all(attrs={'class':re.compile('eventlist__event')})
        all_events += parse_events(events)


# In[7]:


pd.DataFrame(all_events).to_csv('pressreleases.csv', index=False)

