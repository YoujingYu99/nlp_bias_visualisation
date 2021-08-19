import os
import string
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

from bias_visualisation_app.utils.functions import load_obj_user_uploads


def txt_list():
    """
    :param txt_dir: the path of the txt files to be extracted
    :return: a clean list containing the raw sentences
    """
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    txt_dir = os.path.join(fileDir, '..', '..','bias_visualisation_app', 'data', 'user_uploads')
    word_list = []
    # Invoke all the english stopwords
    stop_word_list = set(stopwords.words('english'))

    with open(os.path.join(txt_dir, 'user_input_text'), 'r', encoding='utf-8') as file_in:
        for line in file_in:
            sent_text = nltk.sent_tokenize(line)
            for sent in sent_text:
                sent = sent.lower()
                sent = sent.translate(str.maketrans('', '', string.punctuation))
                tokens = nltk.word_tokenize(sent)

                word_list.append(tokens)

    return word_list


# def calculate_sentence_bias_score(word_list):
#     view_df = load_obj_user_uploads(name='total_dataframe_user_uploads')
#     for sent in word_list:
#         for word in sent:





print(txt_list())