"""
This file is to prepare the amalgum dataset to train our model
extract txt from the tsv file
"""

from readers import *
import os


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


# from dataset_prep import *
#
# p = 'data/amalgum/amalgum_balanced/tsv'
# p1 = 'data/amalgum/amalgum_balanced/txt'
#
# tsv_txt(p, p1)