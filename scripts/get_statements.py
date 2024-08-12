import requests
from bs4 import BeautifulSoup
import re
import pandas as pd


BASE_URL = "https://www.federalreserve.gov"


def parse_events(events, min_date):

    min_date = pd.to_datetime(min_date)
    for event in events:
        date = pd.to_datetime(event.parent.time.text)
        if min_date and date < min_date:
            continue
        description = event.find("a", attrs={"href": re.compile("")}).em.text
        if "FOMC statement" not in description:
            continue
        link = event.find("a", attrs={"href": re.compile("")})["href"]
        link = BASE_URL + link

        yield {"date": date, "link": link, "description": description}


def get_page(s):

    r = requests.get(s.link)
    if r:
        soup = BeautifulSoup(r.content, features="lxml")
        article = soup.find(attrs={"id": "article"})

        return article.text


def get_statements(min_date=None, max_date=None):
    """Downloads all FOMC statements between start and end (inclusive)"""

    response = requests.get(BASE_URL + "/newsevents/pressreleases.htm")
    soup = BeautifulSoup(response.content, features="lxml")

    all_events = []
    for year_tag in soup.find(attrs={"id": "article"}).find_all("a"):
        year_url = year_tag["href"]
        if "archive" not in year_url:
            response = requests.get(BASE_URL + year_url)
            year_soup = BeautifulSoup(response.content, features="lxml")
            events = year_soup.find_all(attrs={"class": re.compile("eventlist__event")})
            all_events += parse_events(events, min_date)

    df = pd.DataFrame(all_events)
    df = df.reset_index(names="id")
    df.index = df["date"]
    df = df.sort_index()
    df = df[~df.index.duplicated("last")]

    if max_date:
        df = df[df.index < max_date]

    df["doc"] = df.apply(get_page, axis=1)

    return df[["doc"]]
