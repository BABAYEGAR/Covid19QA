from sys import platform
from urllib import request

import pandas as pd
from bs4 import BeautifulSoup

if platform == "darwin":
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

meta_data = pd.read_csv("Data/metadata.csv",
                        usecols=["title", "doi", "pmcid", "abstract", "publish_time", "authors", "journal", "license",
                                 "url"],
                        low_memory=False, encoding="utf8")


def scrape_body_text(number_of_articles):
    meta_data["body_text"] = None
    count = 0
    for index, data in meta_data.iterrows():
        if count <= number_of_articles:
            if "pdf" not in data.url:
                req = request.Request(data.url, headers={'User-Agent': 'Mozilla/5.0'})
                html = request.urlopen(req).read().decode('utf8', 'ignore')
                raw = BeautifulSoup(html, 'html.parser')
                for text in raw(['style', 'script']):
                    text.decompose()
                raw = ' '.join(raw.stripped_strings)
                meta_data.loc[index, ['body_text']] = raw
                print("Row " + str(index) + " Completed Successfully...")
                count += 1
            else:
                meta_data.drop(axis=0, index=index)
        else:
            break
    meta_data.to_csv("Data/data.csv")


scrape_body_text(1000)
