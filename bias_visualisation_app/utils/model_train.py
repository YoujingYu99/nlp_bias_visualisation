from functions import *
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

txt_dir = 'bias_visualisation_app/data/amalgum/amalgum_balanced/txt'


training_data = txt_list(txt_dir)

#Train a Word2Vec model using Gensim
Embedding_Dim = 100
#train word2vec model
model = gensim.models.Word2Vec(sentences=training_data, size=Embedding_Dim, workers=4, min_count=1)
#Vocabulary size
words = list(model.wv.vocab)
print('Here is the Vocabulary Size.. %d' % len(words))

model.save("gum_word2vec.model", path='bias_visualisation_app/data/')
