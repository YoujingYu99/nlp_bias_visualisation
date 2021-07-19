import os
from io import open
from conllu import parse, parse_incr
import csv
import xml.etree.ElementTree as ET


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
