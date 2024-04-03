# -*- coding: utf-8 -*-
"""NLP_Russia_Ukraine_War_Tweet_Analysis

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/nlp-russia-ukraine-war-tweet-analysis-5dbedab2-29e8-4509-a3ba-d9dfe9365c98.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20240402/auto/storage/goog4_request%26X-Goog-Date%3D20240402T195345Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D22594e0ecefa420ce63c65c556bc0c1bfad42f97aefc384905a3cd6f1dbec7f87bf8c7a24804c497594ebf70d6a87be2cfe0ccefc875b4bb6eaec7b3be843be0fbda45b7ad31fe2c2a75374b9dd22559d555d39936a4cfc56610efd6280ba964342d39501e4b42008d7cd48d47c19baacc450dbe29fe2d43c84688a2df56251779ce1081c004e6fb7904f95f1f2f574ec6fdd89e946ce2c277ae17d4630dbe1aa91a4375a7b6b040fd33f59d56c2a25f4ed61cd72ea167c5f08f8ec17704d4bd9401a8734b07664cfc733e3650e936ce5cd3c0ae27a3c3daf5abe2ac434e7c1a04fc08caf2b0c869eecd8bba106e69d19500cfc6191e4ca357bf1001822dde48
"""

import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'ukraine-war-tweets-dataset-65-days:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F1981187%2F3271147%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240402%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240402T195345Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D5c343b606262335dd6ecb74b4d99d1976331e848975987561122c8a5ff3464787ad1c6aa8cc172a0fbdc6ac00df95e5efd84d53788961b7b8351affefa5d827373a4bced04c461caacfbdc20937aaf7de8966830a5f8b45d7d919cb819e74a4b9fd1cffe0ac00f44cd7d75781cc84f5a84918e68dcda19ff06ea1a4f52a7416ca76f944475895bf8be896ef28f039423a2fae3a08b8042743d5bf8352e6f574b4bb781f45f3f0476472acaa671c7908a19fc9ef83d19120e190bc8db2c733bdd7ff833381472dbf3064b41371dd32bef0adabf6d0645ff5f0b9b23b90d09a38ff5c61639c9f87c58cf0e55ed2c590e2055143ddbe280836b9192268493ddb146'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

!umount /kaggle/input/ 2> /dev/null
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np
data = pd.read_csv("/kaggle/input/ukraine-war-tweets-dataset-65-days/Ukraine_war.csv")
data

data.info()

data.describe()

data.isnull().sum()

data.columns

col = list(data.columns)
data

data['content']

data['content'][1]

data['lang'].value_counts()

data.lang.value_counts().sort_values().plot(kind = 'pie')

text = data['content'][100]
text

"""# Hashtag"""

import re
ht = re.findall(r"#(\w+)", text)
ht

def hashtag_extract(text_corpus):
    hashtag=[]
    for text in text_corpus:
        ht = re.findall(r"#(\w+)", text)
        hashtag.append(ht)

    return hashtag

def hashtag_freq(hashtag):
    a = nltk.FreqDist(hashtag)
    d = pd. DataFrame({'Hashtah':list(a.keys()),
                  'Freq': list(a.values())}
                     )
    d = d.nlargest(columns="Freq",n = 30 )
    return d

hashtags = hashtag_extract(data['content'])

hashtags

hashtags = sum(hashtags,[])
hashtags

import nltk

hash=hashtag_freq(hashtags)
hash

# @title Freq

from matplotlib import pyplot as plt
hash['Freq'].plot(kind='hist', bins=20, title='Freq')
plt.gca().spines[['top', 'right',]].set_visible(False)

# @title Freq

from matplotlib import pyplot as plt
hash['Freq'].plot(kind='line', figsize=(8, 4), title='Freq')
plt.gca().spines[['top', 'right']].set_visible(False)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize = (26,7))

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'hash' is your DataFrame
ax = sns.barplot(data=hash, x='Hashtah', y='Freq')
plt.xticks(rotation=90)
plt.show()

data['total_length']=data['content'].str.len()

data[['content','total_length']]

from matplotlib import pyplot as plt
_df_1['total_length'].plot(kind='line', figsize=(8, 4), title='total_length')
plt.gca().spines[['top', 'right']].set_visible(False)

from matplotlib import pyplot as plt
_df_0['total_length'].plot(kind='hist', bins=20, title='total_length')
plt.gca().spines[['top', 'right',]].set_visible(False)