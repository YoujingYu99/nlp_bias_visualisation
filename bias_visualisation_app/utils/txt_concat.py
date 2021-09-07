import os
import nltk
from nltk.tokenize import RegexpTokenizer





def txt_concat(txt_dir):
    """
    :param txt_dir: the path of the txt files to be extracted
    :return: concatenated txt file
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
    with open(os.path.join(txt_dir, "concat_test.txt"), 'w', encoding='utf-8') as output:
        for row in training_list:
            output.write(str(row) + '\n')


    return training_list

path_parent = os.path.dirname(os.getcwd())
test_path = os.path.join(path_parent, 'data', 'enwiki_txt')

txt_concat(test_path)