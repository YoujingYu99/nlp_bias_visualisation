import os
import nltk
from nltk.tokenize import RegexpTokenizer


# def txt_list(txt_dir):
#     """
#     :param txt_dir: the path of the txt files to be extracted
#     :return: a clean list containing the raw sentences
#     """
#     txt_files = os.listdir(txt_dir)
#     print(txt_files)
#     file_n = len(txt_files)
#     print('{} files being processed'.format(file_n))
#     with open(os.path.join(txt_dir, 'total.txt'), 'w+', encoding="utf-8", errors='ignore') as outfile:
#         for fname in txt_files:
#             with open(os.path.join(test_path,fname)) as infile:
#                 for line in infile:
#                     outfile.write(line)


def txt_list(txt_dir):
    """
    :param txt_dir: the path of the txt files to be extracted
    :return: a clean list containing the raw sentences
    """
    count = 0
    training_list = []
    txt_files = os.listdir(txt_dir)
    file_n = len(txt_files)
    print('{} files being processed'.format(file_n))
    for file in txt_files:
        while count < 45:
            if file.endswith(".txt"):
                with open(os.path.join(txt_dir, file), 'r', encoding='utf-8') as file_in:
                    for line in file_in:
                        # create word tokens as well as remove puntuation in one go
                        rem_tok_punc = RegexpTokenizer(r'\w+')
                        tokens = nltk.word_tokenize(line)
                        # convert the words to lower case
                        words = [w.lower() for w in tokens]
                        # Remove stop words
                        words = [w for w in words ]
                        words = ' '.join(words)
                        training_list.append(words)
                count += 1
    with open(os.path.join(txt_dir, "written2_test.txt"), 'w', encoding='utf-8') as output:
        for row in training_list:
            output.write(str(row) + '\n')


    return training_list

path_parent = os.path.dirname(os.getcwd())
test_path = os.path.join(path_parent, 'data', 'ANC_written2')

txt_list(test_path)