
import gensim
import os
import glob
import pandas as pd
import numpy as np
import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

nltk.download('stopwords')

txt_dir = "data/amalgum/amalgum_balanced/txt"

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
        with open(os.path.join(path, file),"r", encoding='utf-8') as file_in:
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

training_data = txt_list(txt_dir)
#Train a Word2Vec model using Gensim
Embedding_Dim = 100
#train word2vec model
model = gensim.models.Word2Vec(sentences=training_data, size=Embedding_Dim, workers=4, min_count=1)
#Vocabulary size
words = list(model.wv.vocab)
print('Here is the Vocabulary Size.. %d' % len(words))

model.save("gum_word2vec.model", path='data/')
