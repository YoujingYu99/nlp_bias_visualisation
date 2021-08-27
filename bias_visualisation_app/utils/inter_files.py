import os
import sys
from os import listdir
from io import open
from conllu import parse_incr
import csv
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import requests
from werkzeug.utils import secure_filename
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pandas as pd
from PrecalculatedBiasCalculator import PrecalculatedBiasCalculator


sys.setrecursionlimit(10000)

calculator = PrecalculatedBiasCalculator()

neutral_words = [
    'is',
    'was',
    'who',
    'what',
    'where',
    'the',
    'it',
]

# POS tagging for different word types
adj_list = ['ADJ', 'ADV', 'ADP', 'JJ', 'JJR', 'JJS']
noun_list = ['NOUN', 'PRON' 'PROPN', 'NN', 'NNP', 'NNS', 'NNPS']
verb_list = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'VERB']


def load_obj_user_uploads(name):
    path_parent = os.path.dirname(os.getcwd())
    upload_df_path = os.path.join(path_parent, 'static', 'user_uploads', name)
    df_path = upload_df_path + '.csv'
    return pd.read_csv(df_path, error_bad_lines=False)

print(load_obj_user_uploads(name='gender_count_dataframe_user_uploads')['female_count'].tolist())