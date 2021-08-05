import os
import sys
from os import path
from io import open
import pickle
from conllu import parse, parse_incr
import csv
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import urllib
import requests
import werkzeug
from werkzeug.utils import secure_filename
import spacy
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from flask import url_for
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
from PIL import Image
import re
import cython
from gensim.models import phrases
from gensim import corpora, models, similarities
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from statistics import mean
from gensim.models import Word2Vec, KeyedVectors
from string import ascii_letters, digits
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from gensim.test.utils import datapath

from .parse_sentence import parse_sentence, textify_tokens
from .PcaBiasCalculator import PcaBiasCalculator
from .PrecalculatedBiasCalculator import PrecalculatedBiasCalculator

# NLP bias detection
# if environ.get("USE_PRECALCULATED_BIASES", "").upper() == "TRUE":
#     print("using precalculated biases")
#     calculator = PrecalculatedBiasCalculator()
# else:
#     calculator = PcaBiasCalculator()

# set recursion limit
sys.setrecursionlimit(10000)

calculator = PrecalculatedBiasCalculator()

neutral_words = [
    "is",
    "was",
    "who",
    "what",
    "where",
    "the",
    "it",
]

# POS tagging for different word types
adj_list = ['ADJ', 'ADV', 'ADP', 'JJ', 'JJR', 'JJS']
noun_list = ['NOUN', 'PRON' 'PROPN', 'NN', 'NNP', 'NNS', 'NNPS']
verb_list = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'VERB']


def tsv_reader(path, file):
    """
    :param path: the path into amalgum dataset
    :param file: the file name in the folder: amalgum_genre_docxxx
    :return: a list of rows (lists containing words in sentences)
    """
    if not file.endswith('.tsv'):
        file += '.tsv'
    if os.path.exists(os.path.join(path, file)):
        tsv_file = open(os.path.join(path, file), encoding='utf-8')
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        return read_tsv
    else:
        print(os.path.join(path, file))
        print('file not found')
        pass


def conllu_reader(path, file):
    """
    :param path: the path into amalgum dataset
    :param file: the file name in the folder: amalgum_genre_docxxx
    :return: a token list generator
    """
    file += '.conllu'
    if os.path.exists(os.path.join(path, file)):
        data_file = open(os.path.join(path, file), "r", encoding="utf-8")
        tokenlists = parse_incr(data_file)
        return tokenlists
    else:
        print(os.path.join(path, file))
        print('file not found')
        pass


def etree_reader(path, file):
    """
    :param path: the path into amalgum dataset
    :param file: the file name in the folder: amalgum_genre_docxxx
    :return: an element tree object
    """
    file += '.xml'
    if os.path.exists(os.path.join(path, file)):
        tree = ET.parse(os.path.join(path, file))
        return tree
    else:
        print(os.path.join(path, file))
        print('file not found')
        pass


def get_txt(file, path, save_path):
    """
    :param file: the file in the tsv folder
    :param path: the path of the file's parent directory
    :param save_path: the path to save the newly generated file
    :return: the plain text version of the file using the same name
    """
    f_read = tsv_reader(path, file)
    f_read = [x for x in f_read if x != []]
    f_out = []
    for row in f_read:
        line = row[0]
        if line.startswith('#Text='):
            f_out.append(line[6:])
    with open(os.path.join(save_path, file + '.txt'), 'w+', encoding='utf-8') as f:
        for line in f_out:
            f.write(line + '\n')
    f.close()
    if os.path.exists(os.path.join(save_path, file + '.txt')):
        print("writing completed: " + file)


def txt_list(path):
    """
    :param txt_dir: the path of the txt files to be extracted
    :return: a clean list containing the raw sentences
    """
    training_list = []
    txt_files = os.listdir(path)
    file_n = len(txt_files)
    print("{} files being processed".format(file_n))
    for file in txt_files:
        with open(os.path.join(path, file), "r", encoding='utf-8') as file_in:
            for line in file_in:
                # create word tokens as well as remove puntuation in one go
                rem_tok_punc = RegexpTokenizer(r'\w+')

                tokens = rem_tok_punc.tokenize(line)
                # convert the words to lower case
                words = [w.lower() for w in tokens]
                # Invoke all the english stopwords
                stop_word_list = set(stopwords.words('english'))
                # Remove stop words
                words = [w for w in words if not w in stop_word_list]

                training_list.append(words)

    return training_list


def tsv_txt(tsv_dir, txt_dir):
    """
    :param tsv_dir: the path of the tsv files
    :param txt_dir: the path of the txt files to be saved
    :return: extract all text from the tsv files and save to the txt directory
    """
    tsv_files = os.listdir(tsv_dir)
    file_n = len(tsv_files)
    print("{} files being processed".format(file_n))
    for file in tsv_files:
        file = file[:-4]
        get_txt(file, tsv_dir, txt_dir)


# Fetch Text From Url
def get_text_url(url):
    # page = urllib.request.urlopen(url)
    # soup = BeautifulSoup(page)
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'lxml')
    fetched_text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
    return fetched_text


# Fetch Text From Uploaded File
def get_text_file(corpora_file):
    # get filename
    filename = secure_filename(corpora_file.filename)
    fileDir = os.path.dirname(os.path.realpath('__file__'))

    # os.path.join is used so that paths work in every operating system
    save_user_path = os.path.join(fileDir, 'bias_visualisation_app\\data\\user_uploads')

    with open(os.path.join(save_user_path, filename), 'w+', encoding='utf-8') as f:
        for line in corpora_file:
            line = line.decode()

    return line


from nltk.stem.wordnet import WordNetLemmatizer
import random
import nltk.corpus as nc
import nltk.classify as cf
import nltk
import spacy

SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
OBJECTS = ["dobj", "dative", "attr", "oprd"]
ADJECTIVES = ["acomp", "advcl", "advmod", "amod", "appos", "nn", "nmod", "ccomp", "complm",
              "hmod", "infmod", "xcomp", "rcmod", "poss", "possessive"]
COMPOUNDS = ["compound"]
PREPOSITIONS = ["prep"]


def getSubsFromConjunctions(subs):
    moreSubs = []
    for sub in subs:
        # rights is a generator
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreSubs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"])
            if len(moreSubs) > 0:
                moreSubs.extend(getSubsFromConjunctions(moreSubs))
    return moreSubs


def getObjsFromConjunctions(objs):
    moreObjs = []
    for obj in objs:
        # rights is a generator
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreObjs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == "NOUN"])
            if len(moreObjs) > 0:
                moreObjs.extend(getObjsFromConjunctions(moreObjs))
    return moreObjs


def getVerbsFromConjunctions(verbs):
    moreVerbs = []
    for verb in verbs:
        rightDeps = {tok.lower_ for tok in verb.rights}
        if "and" in rightDeps:
            moreVerbs.extend([tok for tok in verb.rights if tok.pos_ == "VERB"])
            if len(moreVerbs) > 0:
                moreVerbs.extend(getVerbsFromConjunctions(moreVerbs))
    return moreVerbs


def findSubs(tok):
    head = tok.head
    while head.pos_ != "VERB" and head.pos_ != "NOUN" and head.head != head:
        head = head.head
    if head.pos_ == "VERB":
        subs = [tok for tok in head.lefts if tok.dep_ == "SUB"]
        if len(subs) > 0:
            verbNegated = isNegated(head)
            subs.extend(getSubsFromConjunctions(subs))
            return subs, verbNegated
        elif head.head != head:
            return findSubs(head)
    elif head.pos_ == "NOUN":
        return [head], isNegated(tok)
    return [], False


def isNegated(tok):
    negations = {"no", "not", "n't", "never", "none"}
    for dep in list(tok.lefts) + list(tok.rights):
        if dep.lower_ in negations:
            return True
    return False


def findSVs(tokens):
    svs = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB"]
    for v in verbs:
        subs, verbNegated = getAllSubs(v)
        if len(subs) > 0:
            for sub in subs:
                svs.append((sub.orth_, "!" + v.orth_ if verbNegated else v.orth_))
    return svs


def getObjsFromPrepositions(deps):
    objs = []
    for dep in deps:
        if dep.pos_ == "ADP" and dep.dep_ == "prep":
            objs.extend(
                [tok for tok in dep.rights if tok.dep_ in OBJECTS or (tok.pos_ == "PRON" and tok.lower_ == "me")])
    return objs


def getAdjectives(toks):
    toks_with_adjectives = []
    for tok in toks:
        adjs = [left for left in tok.lefts if left.dep_ in ADJECTIVES]
        adjs.append(tok)
        adjs.extend([right for right in tok.rights if tok.dep_ in ADJECTIVES])
        tok_with_adj = " ".join([adj.lower_ for adj in adjs])
        toks_with_adjectives.extend(adjs)

    return toks_with_adjectives


def getObjsFromAttrs(deps):
    for dep in deps:
        if dep.pos_ == "NOUN" and dep.dep_ == "attr":
            verbs = [tok for tok in dep.rights if tok.pos_ == "VERB"]
            if len(verbs) > 0:
                for v in verbs:
                    rights = list(v.rights)
                    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
                    objs.extend(getObjsFromPrepositions(rights))
                    if len(objs) > 0:
                        return v, objs
    return None, None


def getObjFromXComp(deps):
    for dep in deps:
        if dep.pos_ == "VERB" and dep.dep_ == "xcomp":
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(getObjsFromPrepositions(rights))
            if len(objs) > 0:
                return v, objs
    return None, None


def getAllSubs(v):
    verbNegated = isNegated(v)
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
    if len(subs) > 0:
        subs.extend(getSubsFromConjunctions(subs))
    else:
        foundSubs, verbNegated = findSubs(v)
        subs.extend(foundSubs)
    return subs, verbNegated


def getAllObjs(v):
    # rights is a generator
    rights = list(v.rights)
    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
    objs.extend(getObjsFromPrepositions(rights))

    potentialNewVerb, potentialNewObjs = getObjFromXComp(rights)
    if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
        objs.extend(potentialNewObjs)
        v = potentialNewVerb
    if len(objs) > 0:
        objs.extend(getObjsFromConjunctions(objs))
    return v, objs


def getAllObjsWithAdjectives(v):
    # rights is a generator
    rights = list(v.rights)
    objs = [tok for tok in rights if tok.dep_ in OBJECTS]

    if len(objs) == 0:
        objs = [tok for tok in rights if tok.dep_ in ADJECTIVES]

    objs.extend(getObjsFromPrepositions(rights))

    potentialNewVerb, potentialNewObjs = getObjFromXComp(rights)
    if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
        objs.extend(potentialNewObjs)
        v = potentialNewVerb
    if len(objs) > 0:
        objs.extend(getObjsFromConjunctions(objs))
    return v, objs


def findSVOs(tokens):
    svos = []
    verbs = [tok for tok in tokens if tok.pos_ == "AUX"]
    for v in verbs:
        subs, verbNegated = getAllSubs(v)
        # hopefully there are subs, if not, don't examine this verb any longer
        if len(subs) > 0:
            v, objs = getAllObjs(v)
            for sub in subs:
                for obj in objs:
                    objNegated = isNegated(obj)
                    svos.append((sub.lower_, "!" + v.lower_ if verbNegated or objNegated else v.lower_, obj.lower_))
    return svos


def findSVAOs(tokens):
    svos = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB" and tok.dep_ != "aux"]
    for v in verbs:
        subs, verbNegated = getAllSubs(v)
        # hopefully there are subs, if not, don't examine this verb any longer
        if len(subs) > 0:
            v, objs = getAllObjsWithAdjectives(v)
            for sub in subs:
                for obj in objs:
                    objNegated = isNegated(obj)
                    obj_desc_tokens = generate_left_right_adjectives(obj)
                    sub_compound = generate_sub_compound(sub)
                    svos.append((" ".join(tok.lower_ for tok in sub_compound),
                                 "!" + v.lower_ if verbNegated or objNegated else v.lower_,
                                 " ".join(tok.lower_ for tok in obj_desc_tokens)))
    return svos


def generate_sub_compound(sub):
    sub_compunds = []
    for tok in sub.lefts:
        if tok.dep_ in COMPOUNDS:
            sub_compunds.extend(generate_sub_compound(tok))
    sub_compunds.append(sub)
    for tok in sub.rights:
        if tok.dep_ in COMPOUNDS:
            sub_compunds.extend(generate_sub_compound(tok))
    return sub_compunds


def generate_left_right_adjectives(obj):
    obj_desc_tokens = []
    for tok in obj.lefts:
        if tok.dep_ in ADJECTIVES:
            obj_desc_tokens.extend(generate_left_right_adjectives(tok))
    obj_desc_tokens.append(obj)

    for tok in obj.rights:
        if tok.dep_ in ADJECTIVES:
            obj_desc_tokens.extend(generate_left_right_adjectives(tok))

    return obj_desc_tokens


male_names = nc.names.words('male.txt')
male_names.extend(['he', 'him'])
female_names = nc.names.words('female.txt')
female_names.extend(['she', 'her'])
models, acs = [], []

for n_letters in range(1, 6):
    data = []
    for male_name in male_names:
        feature = {'feature': male_name[-n_letters:].lower()}
        data.append((feature, 'male'))
    for female_name in female_names:
        feature = {'feature': female_name[-n_letters:].lower()}
        data.append((feature, 'female'))
    random.seed(7)
    random.shuffle(data)
    train_data = data[:int(len(data) / 2)]
    test_data = data[int(len(data) / 2):]
    model = cf.NaiveBayesClassifier.train(train_data)
    ac = cf.accuracy(model, test_data)
    models.append(model)
    acs.append(ac)

best_index = np.array(acs).argmax()
best_letters = best_index + 1

gender_model = models[best_index]
best_ac = acs[best_index]

neutral_sub_list = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours', 'it', 'its', 'they', 'them', 'their', 'theirs']

spec_chars = ['!',''','#','%','&',''','(',')',
              '*','+',',','-','.','/',':',';','<',
              '=','>','?','@','[','\\',']','^','_',
              '`','{','|','}','~','–']

def reset_gender(subject, subject_gender):
    if subject == 'he':
        subject_gender_new = 'male'
    elif subject == 'she':
        subject_gender_new = 'female'
    elif subject in neutral_sub_list:
        subject_gender_new = 'neutral'
    else:
        subject_gender_new = subject_gender
    return subject_gender_new

def clean_SVO_dataframe(SVO_df):
    # cleaning up the SVO dataframe
    SVO_df['subject_gender'] = SVO_df.apply(lambda x: reset_gender(x.subject, x.subject_gender), axis=1)
    SVO_df['object_gender'] = SVO_df.apply(lambda x: reset_gender(x.object, x.object_gender), axis=1)

    for char in spec_chars:
        SVO_df['subject'] = SVO_df['subject'].str.replace(char, ' ')
        SVO_df['object'] = SVO_df['object'].str.replace(char, ' ')
        SVO_df['verb'] = SVO_df['verb'].str.replace(char, ' ')

    # get base form of verb
    verb_list = SVO_df['verb'].to_list()
    verb_base_list = []
    for verb in verb_list:
        base_word = WordNetLemmatizer().lemmatize(verb, 'v')
        verb_base_list.append(base_word)

    SVO_df['verb'] = verb_base_list

    print(SVO_df)
    return SVO_df




def determine_gender_SVO(input_data):
    parser = spacy.load('en_core_web_md', disable=['ner', 'textcat'])

    sent_text = nltk.sent_tokenize(input_data)
    sub_list = []
    sub_gender_list = []
    verb_list = []
    obj_list = []
    obj_gender_list = []
    # now loop over each sentence and tokenize it separately
    for sentence in sent_text:
        parse = parser(sentence)
        try:
            SVO_list = findSVAOs(parse)
            for i in SVO_list:
                sub, verb, obj = i[0], i[1], i[2]
                sub_feature = {'feature': sub[-best_letters:]}
                sub_gender = gender_model.classify(sub_feature)
                obj_feature = {'feature': obj[-best_letters:]}
                obj_gender = gender_model.classify(obj_feature)

                sub_list.append(sub)
                sub_gender_list.append(sub_gender)
                verb_list.append(verb)
                obj_list.append(obj)
                obj_gender_list.append(obj_gender)

        except:
            continue

    SVO_df = pd.DataFrame(list(zip(sub_list, sub_gender_list, verb_list, obj_list, obj_gender_list)),
                          columns=['subject', 'subject_gender', 'verb', 'object', 'object_gender'])

    #cleaning up the SVO dataframe
    SVO_df = clean_SVO_dataframe(SVO_df)


    return SVO_df


def list_to_dataframe(view_results, scale_range=(-1, 1)):
    # put into a dataframe
    df = pd.DataFrame(view_results)
    # remove None
    df = df.dropna()
    # Normalise to -1 an 1
    scaler = MinMaxScaler(feature_range=scale_range)
    df['bias'] = scaler.fit_transform(df[['bias']])

    return df


def generate_list(dataframe):
    token_list = dataframe['token'].to_list()
    value_list = dataframe['bias'].to_list()
    pos_list = dataframe['pos'].to_list()

    return token_list, value_list, pos_list


def token_by_gender(token_list, value_list):
    # data
    # to convert lists to dictionary
    data = dict(zip(token_list, value_list))
    data = {k: v or 0 for (k, v) in data.items()}

    # separate into male and female dictionaries
    male_token = [k for (k, v) in data.items() if v > 0]
    female_token = [k for (k, v) in data.items() if v < 0]

    return female_token, male_token


def dataframe_by_gender(view_df):
    # selecting rows based on condition
    female_dataframe = view_df[view_df['bias'] < 0]
    male_dataframe = view_df[view_df['bias'] > 0]

    return female_dataframe, male_dataframe


# def dict_by_gender(token_list, value_list):
#     # convert lists to dictionary
#     data = dict(zip(token_list, value_list))
#     data = {k: v or 0 for (k, v) in data.items()}
#
#     # separate into male and female dictionaries
#     # sort from largest to smallest in each case
#     male_dict = {k: v for (k, v) in data.items() if v > 0}
#     male_dict = sorted(male_dict.items(), key=lambda x: x[1], reverse=True)
#     female_dict = {k: v for (k, v) in data.items() if v < 0}
#     female_dict = sorted(female_dict.items(), key=lambda x: x[1], reverse=True)
#
#     return male_dict, female_dict

# def save_obj(obj, name):
#     save_df_path = path.join(path.dirname(__file__), "..\\static\\", name)
#     df_path = save_df_path + '.pkl'
#     with open(df_path, 'wb') as f:
#         pickle.dump(obj, f)
#
# def save_obj_text(obj, name):
#     save_df_path = path.join(path.dirname(__file__), "..\\static\\", name)
#     df_path = save_df_path + '.pkl'
#     with open(df_path, 'wb') as f:
#         pickle.dump(obj, f)
#
# def save_obj_user_uploads(obj, name):
#     save_df_path = path.join(path.dirname(__file__), "..\\static\\user_uploads\\", name)
#     df_path = save_df_path + '.pkl'
#     with open(df_path, 'wb') as f:
#         pickle.dump(obj, f)
#
# def load_obj(name):
#     save_df_path = path.join(path.dirname(__file__), "..\\static\\")
#     with open(save_df_path + name + '.pkl', 'rb') as f:
#         return pickle.load(f)
#
# def load_obj_user_uploads(name):
#     upload_df_path = path.join(path.dirname(__file__), "..\\static\\user_uploads\\")
#     with open(upload_df_path + name + '.pkl', 'rb') as f:
#         return pickle.load(f)


def save_obj(obj, name):
    save_df_path = path.join(path.dirname(__file__), "..\\static\\", name)
    df_path = save_df_path + '.csv'
    obj.to_csv(df_path, index=False)


def save_obj_text(obj, name):
    save_df_path = path.join(path.dirname(__file__), "..\\static\\", name)
    df_path = save_df_path + '.csv'
    obj.to_csv(df_path, index=False)


def save_obj_user_uploads(obj, name):
    save_df_path = path.join(path.dirname(__file__), "..\\static\\user_uploads\\", name)
    df_path = save_df_path + '.csv'
    obj.to_csv(df_path, index=False)


def load_obj(name):
    save_df_path = path.join(path.dirname(__file__), "..\\static\\", name)
    df_path = save_df_path + '.csv'
    return pd.read_csv(df_path)


def load_obj_user_uploads(name):
    upload_df_path = path.join(path.dirname(__file__), "..\\static\\user_uploads", name)
    df_path = upload_df_path + '.csv'
    return pd.read_csv(df_path)


def generate_bias_values(input_data):
    objs = parse_sentence(input_data)
    results = []
    view_results = []
    for obj in objs:
        token_result = {
            "token": obj["text"],
            "bias": calculator.detect_bias(obj["text"]),
            "parts": [
                {
                    "whitespace": token.whitespace_,
                    "pos": token.pos_,
                    "dep": token.dep_,
                    "ent": token.ent_type_,
                    "skip": token.pos_
                            in ["AUX", "ADP", "PUNCT", "SPACE", "DET", "PART", "CCONJ"]
                            or len(token) < 2
                            or token.text.lower() in neutral_words,
                }
                for token in obj["tokens"]
            ],
        }
        results.append(token_result)
    # copy results and only keep the word and the bias value
    token_result2 = results.copy()
    for item in token_result2:
        if 'parts' in item.keys():
            if item['parts'][0]['pos'] in adj_list or item['parts'][0]['pos'] in noun_list or item['parts'][0][
                'pos'] in verb_list:
                item['pos'] = item['parts'][0]['pos']
            del item['parts']
        else:
            continue
        view_results.append(item)

    view_df = list_to_dataframe(view_results)
    save_obj_text(view_df, name='total_dataframe')

    SVO_df = determine_gender_SVO(input_data)
    save_obj_text(SVO_df, name='SVO_dataframe')


def frame_from_file(view_df):
    token_list, value_list, pos_list = generate_list(view_df)
    return view_df, (token_list, value_list)


def SVO_analysis(view_df):
    # columns = ['subject', 'subject_gender', 'verb', 'object', 'object_gender']
    female_sub_df = view_df.loc[view_df['subject_gender'] == 'female']
    female_obj_df = view_df.loc[view_df['object_gender'] == 'female']
    male_sub_df = view_df.loc[view_df['subject_gender'] == 'male']
    male_obj_df = view_df.loc[view_df['object_gender'] == 'male']

    # female_sub_df['Frequency'] = female_sub_df.groupby('subject').transform('count')
    # df['frequency'] = df['county'].map(df['county'].value_counts())
    female_sub_df_new = female_sub_df.copy()
    female_sub_df_new['Frequency'] = female_sub_df_new['verb'].map(female_sub_df_new['verb'].value_counts())
    female_sub_df_new.sort_values('Frequency', inplace=True, ascending=False)
    female_sub_df_new.drop(columns=['subject', 'subject_gender', 'object', 'object_gender'], inplace=True)
    female_sub_df_new.drop_duplicates(subset='verb',
                                  keep=False, inplace=True)

    female_obj_df_new = female_obj_df.copy()
    female_obj_df_new['Frequency'] = female_obj_df_new['verb'].map(female_obj_df_new['verb'].value_counts())
    female_obj_df_new.sort_values('Frequency', inplace=True, ascending=False)
    female_obj_df_new.drop(columns=['subject', 'subject_gender', 'object', 'object_gender'], inplace=True)
    female_obj_df_new.drop_duplicates(subset='verb',
                                  keep=False, inplace=True)

    male_sub_df_new = male_sub_df.copy()
    male_sub_df_new['Frequency'] = male_sub_df_new['verb'].map(male_sub_df_new['verb'].value_counts())
    male_sub_df_new.sort_values('Frequency', inplace=True, ascending=False)
    male_sub_df_new.drop(columns=['subject', 'subject_gender', 'object', 'object_gender'], inplace=True)
    male_sub_df_new.drop_duplicates(subset='verb',
                                keep=False, inplace=True)

    male_obj_df_new = male_obj_df.copy()
    male_obj_df_new['Frequency'] = male_obj_df_new['verb'].map(male_obj_df_new['verb'].value_counts())
    male_obj_df_new.sort_values('Frequency', inplace=True, ascending=False)
    male_obj_df_new.drop(columns=['subject', 'subject_gender', 'object', 'object_gender'], inplace=True)
    male_obj_df_new.drop_duplicates(subset='verb',
                                keep=False, inplace=True)

    return female_sub_df_new, female_obj_df_new, male_sub_df_new, male_obj_df_new


def gender_dataframe_from_tuple(view_df):
    female_dataframe, male_dataframe = dataframe_by_gender(view_df)

    save_obj(female_dataframe, name='fm_dic')
    save_obj(male_dataframe, name='m_dic')
    male_dataframe = load_obj(name='m_dic')
    male_dataframe = male_dataframe.sort_values(by='bias', ascending=False)
    male_dataframe = male_dataframe.drop_duplicates(subset=['token'])

    female_dataframe = load_obj(name='fm_dic')
    female_dataframe = female_dataframe.sort_values(by='bias', ascending=True)
    female_dataframe = female_dataframe.drop_duplicates(subset=['token'])

    return female_dataframe, male_dataframe


def parse_pos_dataframe(view_df):
    female_dataframe, male_dataframe = gender_dataframe_from_tuple(view_df)

    female_noun_df = female_dataframe[female_dataframe['pos'].isin(noun_list)]
    female_adj_df = female_dataframe[female_dataframe['pos'].isin(adj_list)]
    female_verb_df = female_dataframe[female_dataframe['pos'].isin(verb_list)]

    male_noun_df = male_dataframe[male_dataframe['pos'].isin(noun_list)]
    male_adj_df = male_dataframe[male_dataframe['pos'].isin(adj_list)]
    male_verb_df = male_dataframe.loc[male_dataframe['pos'].isin(verb_list)]

    return female_noun_df, female_adj_df, female_verb_df, male_noun_df, male_adj_df, male_verb_df


def bar_graph(dataframe, token_list, value_list):
    # set minus sign
    mpl.rcParams['axes.unicode_minus'] = False
    np.random.seed(12345)
    df = dataframe
    if len(token_list) > 15:
        set_x_tick = False
    else:
        set_x_tick = True

    plt.style.use('ggplot')
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig, ax = plt.subplots()

    # set up the colors
    cmap = mpl.colors.LinearSegmentedColormap.from_list('green_to_red', ['darkgreen', 'darkred'])
    df_mean = df.mean(axis=1)
    norm = plt.Normalize(df_mean.min(), df_mean.max())
    colors = cmap(norm(df_mean))

    ax.bar(
        token_list,
        value_list,
        yerr=df.std(axis=1) / np.sqrt(len(df.columns)),
        color=colors)
    fig.colorbar(ScalarMappable(cmap=cmap))

    ax.set_title('Word Bias Visualisation', fontsize=12)
    ax.set_xlabel('Word')
    ax.xaxis.set_visible(set_x_tick)
    ax.set_ylabel('Bias Value')
    plt.tight_layout()

    # save file to static
    bar_name = token_list[0] + token_list[-2]
    bar_name_ex = bar_name + '.png'
    save_img_path = path.join(path.dirname(__file__), "..\\static\\", bar_name)
    bar_path = save_img_path + '.png'
    plt.savefig(bar_path)
    plot_bar = url_for('static', filename=bar_name_ex)

    return plot_bar


def specific_bar_graph(df_name='specific_df'):
    # set minus sign
    try:
        mpl.rcParams['axes.unicode_minus'] = False
        np.random.seed(12345)
        df = load_obj(name=df_name)
        set_x_tick = True

        plt.style.use('ggplot')
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['SimHei']
        fig, ax = plt.subplots()

        # set up the colors
        cmap = mpl.colors.LinearSegmentedColormap.from_list('green_to_red', ['darkgreen', 'darkred'])
        df_mean = df.mean(axis=1)
        norm = plt.Normalize(df_mean.min(), df_mean.max())
        colors = cmap(norm(df_mean))

        ax.barh(
            df['token'],
            df['bias'],
            yerr=df.std(axis=1) / np.sqrt(len(df.columns)),
            color=colors)
        fig.colorbar(ScalarMappable(cmap=cmap))

        ax.set_title('Specific Word Bias', fontsize=12)
        ax.set_xlabel('Bias Value')
        ax.xaxis.set_visible(set_x_tick)

        ax.set_ylabel('Word')
        plt.tight_layout()

        # save file to static
        bar_name = df['token'].iloc[0] + df['token'].iloc[-2]
        bar_name_ex = bar_name + '.png'
        save_img_path = path.join(path.dirname(__file__), "..\\static\\", bar_name)
        bar_path = save_img_path + '.png'
        plt.savefig(bar_path)
        plot_bar = url_for('static', filename=bar_name_ex)

        return plot_bar

    except:
        mpl.rcParams['axes.unicode_minus'] = False
        np.random.seed(12345)
        df = load_obj(name=df_name)
        set_x_tick = True

        plt.style.use('ggplot')
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['SimHei']
        fig, ax = plt.subplots()

        # set up the colors
        cmap = mpl.colors.LinearSegmentedColormap.from_list('green_to_red', ['darkgreen', 'darkred'])
        df_mean = df.mean(axis=1)
        norm = plt.Normalize(df_mean.min(), df_mean.max())
        colors = cmap(norm(df_mean))

        ax.barh(
            df['verb'],
            df['Frequency'],
            yerr=df.std(axis=1) / np.sqrt(len(df.columns)),
            color=colors)
        fig.colorbar(ScalarMappable(cmap=cmap))

        ax.set_title('Specific Word Frequency', fontsize=12)
        ax.set_xlabel('Frequency')
        ax.xaxis.set_visible(set_x_tick)

        ax.set_ylabel('Word')
        plt.tight_layout()

        # save file to static
        bar_name = df['verb'].iloc[0] + df['verb'].iloc[1]
        bar_name_ex = bar_name + '.png'
        save_img_path = path.join(path.dirname(__file__), "..\\static\\", bar_name)
        bar_path = save_img_path + '.png'
        plt.savefig(bar_path)
        plot_bar = url_for('static', filename=bar_name_ex)

        return plot_bar


# def bar_graph(token_list, value_list):
#     plt.rcParams['axes.unicode_minus'] = False
#
#
#     def autolable(rects):
#         for rect in rects:
#             height = rect.get_height()
#             if height >= 0:
#                 plt.text(rect.get_x() + rect.get_width() / 2.0 - 0.3, height + 0.02, '%.3f' % height)
#             else:
#                 plt.text(rect.get_x() + rect.get_width() / 2.0 - 0.3, height - 0.06, '%.3f' % height)
#                 plt.axhline(y=0, color='black')
#
#     # normalise
#     norm = plt.Normalize(-1, 1)
#     norm_values = norm(value_list)
#     map_vir = cm.get_cmap(name='viridis')
#     colors = map_vir(norm_values)
#     fig = plt.figure()
#     plt.subplot(111)
#     ax = plt.bar(token_list, value_list, width=0.5, color=colors, edgecolor='black')
#
#     sm = cm.ScalarMappable(cmap=map_vir, norm=norm)
#     sm.set_array([])
#     plt.colorbar(sm)
#     autolable(ax)
#
#     # save file to static
#     bar_name = token_list[0]
#     bar_name_ex = bar_name + '.png'
#     save_img_path = path.join(path.dirname(__file__), "..\\static\\", bar_name)
#     bar_path = save_img_path + '.png'
#     plt.savefig(bar_path)
#     plot_bar = url_for('static', filename=bar_name_ex)
#
#     return plot_bar


def transform_format(val):
    if val.any() == 0:
        return 255
    else:
        return val


def cloud_image(token_list, value_list):
    # data
    # to convert lists to dictionary
    data = dict(zip(token_list, value_list))
    data = {k: v or 0 for (k, v) in data.items()}

    # separate into male and female dictionaries
    male_data = {k: v for (k, v) in data.items() if v > 0}
    female_data = {k: v for (k, v) in data.items() if v < 0}

    # cloud
    cloud_color = "magma"
    cloud_bg_color = "white"
    # cloud_custom_font = False

    # transform mask
    # female_mask_path = path.join(path.dirname(__file__), "..\\static\\images", "female_symbol.png")
    # male_mask_path = path.join(path.dirname(__file__), "..\\static\\images", "male_symbol.png")
    #
    # female_cloud_mask = np.array(Image.open(female_mask_path))
    # male_cloud_mask = np.array(Image.open(male_mask_path))

    cloud_scale = 0.1
    cloud_horizontal = 1
    bigrams = True

    # Setting up wordcloud from previously set variables.
    female_wordcloud = WordCloud(collocations=bigrams, regexp=None,
                                 relative_scaling=cloud_scale, width=1000,
                                 height=500, background_color=cloud_bg_color, max_words=10000,
                                 contour_width=0,
                                 colormap=cloud_color)

    male_wordcloud = WordCloud(collocations=bigrams, regexp=None, relative_scaling=cloud_scale,
                               width=1000,
                               height=500, background_color=cloud_bg_color, max_words=10000,
                               contour_width=0,
                               colormap=cloud_color)

    try:
        female_wordcloud.generate_from_frequencies(female_data)

        # save file to static
        female_cloud_name = str(next(iter(female_data))) + 'femalecloud'
        female_cloud_name_ex = female_cloud_name + '.png'
        save_img_path = path.join(path.dirname(__file__), "..\\static\\", female_cloud_name)
        img_path = save_img_path + '.png'
        female_wordcloud.to_file(img_path)

        plot_female_cloud = url_for('static', filename=female_cloud_name_ex)

    except:
        # https: // www.wattpad.com / 729617965 - there % 27s - nothing - here - 3
        # https://images-na.ssl-images-amazon.com/images/I/41wjfr0wSsL.png
        print("Not enough words for female cloud!")
        plot_female_cloud = url_for('static', filename="nothing_here.jpg")

    try:
        male_wordcloud.generate_from_frequencies(male_data)

        # save file to static
        male_cloud_name = str(next(iter(male_data))) + 'malecloud'
        male_cloud_name_ex = male_cloud_name + '.png'
        save_img_path = path.join(path.dirname(__file__), "..\\static\\", male_cloud_name)
        img_path = save_img_path + '.png'
        male_wordcloud.to_file(img_path)

        plot_male_cloud = url_for('static', filename=male_cloud_name_ex)

    except:
        print("Not enough words for male cloud!")
        plot_male_cloud = url_for('static', filename="nothing_here.jpg")

    return plot_female_cloud, plot_male_cloud


def tsne_graph(token_list, iterations=3000, seed=20, title="TSNE Visualisation of Word-Vectors for Amalgum(Overall)"):
    """Creates a TSNE model and plots it"""

    # define word2vec model
    model_path = path.join(path.dirname(__file__), "../data/gum_word2vec.model")
    w2vmodel = Word2Vec.load(model_path)

    # manually define which words we want to explore
    my_word_list = []
    my_word_vectors = []

    words_to_explore = token_list

    for i in words_to_explore:
        try:
            if my_word_list not in my_word_list:
                my_word_vectors.append(w2vmodel.wv[i])
                my_word_list.append(i)
        except KeyError:
            continue

    tsne_model = TSNE(perplexity=5, n_components=2, init='pca', n_iter=iterations,
                      random_state=seed)
    new_values = tsne_model.fit_transform(my_word_vectors)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    # save file to static
    tsne_name = token_list[0] + token_list[-2] + 'tsne'
    tsne_name_ex = tsne_name + '.jpg'
    save_img_path = path.join(path.dirname(__file__), "..\\static\\", tsne_name)
    tsne_path = save_img_path + '.jpg'

    plt.figure(figsize=(10, 10))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(my_word_list[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.ylabel("TSNE Latent Dimension 1")
    plt.xlabel("TSNE Latent Dimension 2")
    plt.title(title)
    plt.savefig(tsne_path)
    plot_tsne = url_for('static', filename=tsne_name_ex)

    return plot_tsne


def tsne_graph_male(token_list, value_list, iterations=3000, seed=20, title="TSNE Visualisation(Male)"):
    """Creates a TSNE model and plots it"""

    # define word2vec model
    model_path = path.join(path.dirname(__file__), "../data/gum_word2vec.model")
    w2vmodel = Word2Vec.load(model_path)

    # manually define which words we want to explore
    my_word_list = []
    my_word_vectors = []

    words_to_explore = token_by_gender(token_list, value_list)[1]

    for i in words_to_explore:
        try:
            if my_word_list not in my_word_list:
                my_word_vectors.append(w2vmodel.wv[i])
                my_word_list.append(i)
        except KeyError:
            continue

    tsne_model = TSNE(perplexity=5, n_components=2, init='pca', n_iter=iterations,
                      random_state=seed)
    new_values = tsne_model.fit_transform(my_word_vectors)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    # save file to static
    tsne_name = token_list[0] + token_list[-2] + 'tsne_male'
    tsne_name_ex = tsne_name + '.jpg'
    save_img_path = path.join(path.dirname(__file__), "..\\static\\", tsne_name)
    tsne_path = save_img_path + '.jpg'

    plt.figure(figsize=(10, 10))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(my_word_list[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.ylabel("TSNE Latent Dimension 1")
    plt.xlabel("TSNE Latent Dimension 2")
    plt.title(title)
    plt.savefig(tsne_path)
    plot_tsne_male = url_for('static', filename=tsne_name_ex)

    return plot_tsne_male


def tsne_graph_female(token_list, value_list, iterations=3000, seed=20, title="TSNE Visualisation (Female)"):
    """Creates a TSNE model and plots it"""

    # define word2vec model
    model_path = path.join(path.dirname(__file__), "../data/gum_word2vec.model")
    w2vmodel = Word2Vec.load(model_path)

    # manually define which words we want to explore
    my_word_list = []
    my_word_vectors = []

    words_to_explore = token_by_gender(token_list, value_list)[1]

    for i in words_to_explore:
        try:
            if my_word_list not in my_word_list:
                my_word_vectors.append(w2vmodel.wv[i])
                my_word_list.append(i)
        except KeyError:
            continue

    tsne_model = TSNE(perplexity=5, n_components=2, init='pca', n_iter=iterations,
                      random_state=seed)
    new_values = tsne_model.fit_transform(my_word_vectors)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    # save file to static
    tsne_name = token_list[0] + token_list[-2] + 'tsne_female'
    tsne_name_ex = tsne_name + '.jpg'
    save_img_path = path.join(path.dirname(__file__), "..\\static\\", tsne_name)
    tsne_path = save_img_path + '.jpg'

    plt.figure(figsize=(10, 10))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(my_word_list[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.ylabel("TSNE Latent Dimension 1")
    plt.xlabel("TSNE Latent Dimension 2")
    plt.title(title)
    plt.savefig(tsne_path)
    plot_tsne_female = url_for('static', filename=tsne_name_ex)

    return plot_tsne_female


def pca_graph(token_list, title="PCA Visualisation of Word-Vectors for Amalgum"):
    """Creates a PCA model and plots it"""

    # define word2vec model
    model_path = path.join(path.dirname(__file__), "../data/gum_word2vec.model")
    w2vmodel = Word2Vec.load(model_path)

    # manually define which words we want to explore
    my_word_list = []
    my_word_vectors = []

    words_to_explore = token_list

    for i in words_to_explore:
        try:
            if my_word_list not in my_word_list:
                my_word_vectors.append(w2vmodel.wv[i])
                my_word_list.append(i)
        except KeyError:
            continue

    pca_model = PCA(n_components=2, svd_solver='full')
    new_values = pca_model.fit_transform(my_word_vectors)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    # save file to static
    pca_name = token_list[0] + token_list[-2] + 'pca'
    pca_name_ex = pca_name + '.jpg'
    save_img_path = path.join(path.dirname(__file__), "..\\static\\", pca_name)
    pca_path = save_img_path + '.jpg'

    plt.figure(figsize=(10, 10))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(my_word_list[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.ylabel("PCA Latent Dimension 1")
    plt.xlabel("PCA Latent Dimension 2")
    plt.title(title)
    plt.savefig(pca_path)
    plot_pca = url_for('static', filename=pca_name_ex)

    return plot_pca


def pca_graph_male(token_list, value_list, title="PCA Visualisation(Male)"):
    """Creates a PCA model and plots it"""

    # define word2vec model
    model_path = path.join(path.dirname(__file__), "../data/gum_word2vec.model")
    w2vmodel = Word2Vec.load(model_path)

    # manually define which words we want to explore
    my_word_list = []
    my_word_vectors = []

    words_to_explore = token_by_gender(token_list, value_list)[1]

    for i in words_to_explore:
        try:
            if my_word_list not in my_word_list:
                my_word_vectors.append(w2vmodel.wv[i])
                my_word_list.append(i)
        except KeyError:
            continue

    pca_model = PCA(n_components=2, svd_solver='full')
    new_values = pca_model.fit_transform(my_word_vectors)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    # save file to static
    pca_name = token_list[0] + token_list[-2] + 'pca_male'
    pca_name_ex = pca_name + '.jpg'
    save_img_path = path.join(path.dirname(__file__), "..\\static\\", pca_name)
    pca_path = save_img_path + '.jpg'

    plt.figure(figsize=(10, 10))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(my_word_list[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.ylabel("PCA Latent Dimension 1")
    plt.xlabel("PCA Latent Dimension 2")
    plt.title(title)
    plt.savefig(pca_path)
    plot_pca_male = url_for('static', filename=pca_name_ex)

    return plot_pca_male


def pca_graph_female(token_list, value_list, title="PCA Visualisation(Female)"):
    """Creates a PCA model and plots it"""

    # define word2vec model
    model_path = path.join(path.dirname(__file__), "../data/gum_word2vec.model")
    w2vmodel = Word2Vec.load(model_path)

    # manually define which words we want to explore
    my_word_list = []
    my_word_vectors = []

    words_to_explore = token_by_gender(token_list, value_list)[1]

    for i in words_to_explore:
        try:
            if my_word_list not in my_word_list:
                my_word_vectors.append(w2vmodel.wv[i])
                my_word_list.append(i)
        except KeyError:
            continue

    pca_model = PCA(n_components=2, svd_solver='full')
    new_values = pca_model.fit_transform(my_word_vectors)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    # save file to static
    pca_name = token_list[0] + token_list[-2] + 'pca_female'
    pca_name_ex = pca_name + '.jpg'
    save_img_path = path.join(path.dirname(__file__), "..\\static\\", pca_name)
    pca_path = save_img_path + '.jpg'

    plt.figure(figsize=(10, 10))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(my_word_list[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.ylabel("PCA Latent Dimension 1")
    plt.xlabel("PCA Latent Dimension 2")
    plt.title(title)
    plt.savefig(pca_path)
    plot_pca_female = url_for('static', filename=pca_name_ex)

    return plot_pca_female


def df_based_on_question(select_wordtype, select_gender, view_df, input_SVO_dataframe):
    female_tot_df, male_tot_df = gender_dataframe_from_tuple(view_df)
    female_noun_df, female_adj_df, female_verb_df = parse_pos_dataframe(view_df)[:3]
    male_noun_df, male_adj_df, male_verb_df = parse_pos_dataframe(view_df)[-3:]
    female_sub_df, female_obj_df, male_sub_df, male_obj_df = SVO_analysis(input_SVO_dataframe)
    if select_gender == 'female':
        if select_wordtype == 'nouns':
            return female_noun_df
        if select_wordtype == 'adjectives':
            return female_adj_df
        if select_wordtype == 'subject_verbs':
            return female_sub_df
        if select_wordtype == 'object_verbs':
            return female_obj_df
        else:
            raise werkzeug.exceptions.BadRequest(
                'Please recheck your question'
            )
    if select_gender == 'male':
        if select_wordtype == 'nouns':
            return male_noun_df
        if select_wordtype == 'adjectives':
            return male_adj_df
        if select_wordtype == 'subject_verbs':
            return female_sub_df
        if select_wordtype == 'object_verbs':
            return female_obj_df
        else:
            raise werkzeug.exceptions.BadRequest(
                'Please recheck your question'
            )

# p = 'bias_visualisation_app/data/amalgum/amalgum_balanced/tsv'
# p1 = 'bias_visualisation_app/data/amalgum/amalgum_balanced/txt'
#
# tsv_txt(p, p1)
