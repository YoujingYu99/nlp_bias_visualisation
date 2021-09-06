import os
import shutil
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

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

path_parent = os.path.dirname(os.getcwd())
test_path = os.path.join(path_parent, 'data', 'ANC')

print(txt_list(test_path)[10])