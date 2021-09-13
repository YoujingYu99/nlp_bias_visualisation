import os
import glob
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
from .PrecalculatedBiasCalculator import PrecalculatedBiasCalculator

# remove the . before .PrecalculatedBiasCalculator when running tests.


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
        read_tsv = csv.reader(tsv_file, delimiter='\t')
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
        data_file = open(os.path.join(path, file), 'r', encoding='utf-8')
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
        print('writing completed: ' + file)


def txt_list(txt_dir):
    """
    :param txt_dir: the path of the txt files to be extracted
    :return: a clean list containing the raw sentences
    """
    training_list = []
    txt_files = os.listdir(txt_dir)
    file_n = len(txt_files)
    print('{} files being processed'.format(file_n))
    for file in txt_files:
        if file.endswith(".txt"):
            with open(os.path.join(txt_dir, file), 'r', encoding='utf-8') as file_in:
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
    print('{} files being processed'.format(file_n))
    for file in tsv_files:
        file = file[:-4]
        get_txt(file, tsv_dir, txt_dir)


# Fetch text from Url
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
    save_user_path = os.path.join(fileDir, 'bias_visualisation_app', 'static', 'user_uploads_text')
    # need to write out the lines
    lines = ""
    with open(os.path.join(save_user_path, filename), 'w+', encoding='utf-8') as f:
        for line in corpora_file:
            line = line.decode()
            lines = lines + line
        # save file to the filename
        f.write(lines)

    return lines

def save_user_file_text(user_text):
    # user inputs a string
    fileDir = os.path.dirname(os.path.realpath('__file__'))

    # os.path.join is used so that paths work in every operating system
    save_user_path = os.path.join(fileDir, 'bias_visualisation_app', 'static', 'user_uploads_text')

    with open(os.path.join(save_user_path, 'user_input_text.txt'), 'w+', encoding='utf-8') as f:
       f.write(user_text)




def save_obj(obj, name):
    path_parent = os.path.dirname(os.getcwd())
    save_df_path = os.path.join(path_parent, 'visualising_data_bias', 'bias_visualisation_app', 'static', name)
    df_path = save_df_path + '.csv'
    obj.to_csv(df_path, index=False)


def save_obj_text(obj, name):
    path_parent = os.path.dirname(os.getcwd())
    save_df_path = os.path.join(path_parent, 'visualising_data_bias', 'bias_visualisation_app', 'static', 'user_downloads', name)
    df_path = save_df_path + '.csv'
    obj.to_csv(df_path, index=False)


def save_obj_user_uploads(obj, name):
    path_parent = os.path.dirname(os.getcwd())
    save_df_path = os.path.join(path_parent, 'visualising_data_bias', 'bias_visualisation_app', 'static', 'user_uploads', name)
    df_path = save_df_path + '.csv'
    obj.to_csv(df_path, index=False)



def concat_csv_excel(csv_path):
    csv_files = [f for f in listdir(csv_path) if f.endswith('.csv')]
    writer = pd.ExcelWriter(os.path.join(csv_path, 'complete_file.xlsx'), engine='xlsxwriter')
    for file in csv_files:
        df = pd.read_csv(os.path.join(csv_path, file))
        df.to_excel(writer, sheet_name=os.path.splitext(file)[0], index=False)
    writer.save()

def load_obj(df_path,name):
    save_df_path = os.path.join(df_path, name)
    df_path = save_df_path + '.csv'
    return pd.read_csv(df_path, error_bad_lines=False)


def load_obj_user_uploads(df_path, name):
    upload_df_path = os.path.join(df_path, name)
    actual_df_path = upload_df_path + '.csv'
    return pd.read_csv(actual_df_path, error_bad_lines=False)








# p = 'bias_visualisation_app/data/amalgum/amalgum_balanced/tsv'
# p1 = 'bias_visualisation_app/data/amalgum/amalgum_balanced/txt'
#
# tsv_txt(p, p1)
