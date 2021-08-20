import os
import string
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from sklearn.preprocessing import MinMaxScaler

from bias_visualisation_app.utils.functions import load_obj_user_uploads, load_total_dataframe


def user_input_list():
    """
    :param txt_dir: the path of the txt files to be extracted
    :return: a clean list containing the raw sentences
    """
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    txt_dir = os.path.join(fileDir, '..', '..','bias_visualisation_app', 'data', 'user_uploads')
    word_list = []
    punct_list = []

    with open(os.path.join(txt_dir, 'user_input_text'), 'r', encoding='utf-8') as file_in:
        for line in file_in:
            sent_text = nltk.sent_tokenize(line)
            for sent in sent_text:
                sent = sent.lower()
                punct = sent[-1]
                sent = sent.translate(str.maketrans('', '', string.punctuation))
                tokens = nltk.word_tokenize(sent)
                word_list.append(tokens)
                punct_list.append(punct)
    return word_list, punct_list


def calculate_sentence_bias_score(word_list, punct_list):
    view_df = load_total_dataframe(name='total_dataframe_user_uploads')
    sentence_score_list = []
    count = 0
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

        sentence_score = {'sentence': sent, 'score': mean_bias_score, 'punct': punct_list[count]}
        sentence_score_list.append(sentence_score)
        count += 1

    # convert to dataframe and renormalise it.
    sentence_score_df = pd.DataFrame(sentence_score_list)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    sentence_score_df['score'] = scaler.fit_transform(sentence_score_df[['score']])

    return sentence_score_df

replacers = {"dont": "don't", "doesnt": "doesn't", "wont": "won't", "wouldnt": "wouldn't", "cant": "can't", "couldnt": "couldn't", "neednt": "needn't", "shouldnt": "shouldn't"}

special_phrases = ["dont", "doesnt", "wont", "wouldnt", "cant", "couldnt", "neednt", "shouldnt"]


def debiased_file(threshold_value):
    word_list, punct_list = user_input_list()
    sentence_score_df = calculate_sentence_bias_score(word_list, punct_list)
    debiased_df = sentence_score_df[sentence_score_df['score'].between(-abs(threshold_value), threshold_value)]
    debiased_sentence_list = []
    for index, row in debiased_df.iterrows():
        sentence = row['sentence']
        new_sentence = ' '.join(str(x) for x in sentence) + row['punct']

        try:
            new_sentence.replace(replacers)
        except:
            new_sentence = new_sentence

        debiased_sentence_list.append(new_sentence)

    path_parent = os.path.dirname(os.getcwd())
    save_path = os.path.join(path_parent, 'static')

    with open(os.path.join(save_path, 'debiased_file' + '.txt'), 'w+', encoding='utf-8') as f:
        f.write('\n'.join(debiased_sentence_list))


debiased_file(threshold_value=0.8)

