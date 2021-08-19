import os
import string
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

from bias_visualisation_app.utils.functions import load_obj_user_uploads, load_total_dataframe


def txt_list():
    """
    :param txt_dir: the path of the txt files to be extracted
    :return: a clean list containing the raw sentences
    """
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    txt_dir = os.path.join(fileDir, '..', '..','bias_visualisation_app', 'data', 'user_uploads')
    word_list = []

    with open(os.path.join(txt_dir, 'user_input_text'), 'r', encoding='utf-8') as file_in:
        for line in file_in:
            sent_text = nltk.sent_tokenize(line)
            for sent in sent_text:
                sent = sent.lower()
                sent = sent.translate(str.maketrans('', '', string.punctuation))
                tokens = nltk.word_tokenize(sent)
                word_list.append(tokens)

    return word_list


def calculate_sentence_bias_score(word_list):
    view_df = load_total_dataframe(name='total_dataframe_user_uploads')
    for sent in word_list:
        bias_list = []
        for word in sent:
            try:
                bias_value = view_df.loc[view_df['token'] == word, 'bias'].iloc[0]
                bias_list.append(bias_value)
            except:
                continue
        if len(bias_list) == 0:
            mean_bias_score = 0
        else:
            mean_bias_score = sum(bias_list)/len(bias_list)
        print(mean_bias_score)

word_list = txt_list()

print(calculate_sentence_bias_score(word_list))

