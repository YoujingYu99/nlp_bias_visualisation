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
    sentence_score_list = []
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

        sentence_score = {'sentence': sent, 'score': mean_bias_score}
        sentence_score_list.append(sentence_score)
        sentence_score_df = pd.DataFrame(sentence_score_list)

    return sentence_score_df

word_list = txt_list()
sentence_score_df = calculate_sentence_bias_score(word_list)

def debiased_file(threshold_value, sentence_score_df):
    debiased_df = sentence_score_df.loc[sentence_score_df['score'] > threshold_value]
    debiased_sentence_list = []
    for index, row in debiased_df.iterrows():
        sentence = row['sentence']
        new_sentence = ' '.join(str(x) for x in sentence) + '.'
        debiased_sentence_list.append(new_sentence)

    path_parent = os.path.dirname(os.getcwd())
    save_path = os.path.join(path_parent, 'static')

    with open(os.path.join(save_path, 'debiased_file'), 'w+', encoding='utf-8') as f:
        f.write('\n'.join(debiased_sentence_list))


debiased_file(threshold_value=0.5, sentence_score_df=sentence_score_df)

