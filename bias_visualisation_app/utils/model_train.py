# only run this file if you want to train your own model instead of using the model trained on amalgum dataset.
import os
import gensim
import nltk
from .functions_files import txt_list

nltk.download('stopwords')
fileDir = os.path.dirname(os.path.realpath('__file__'))
txt_dir = os.path.join(fileDir, 'bias_visualisation_app', 'data', 'amalgum', 'amalgum_balanced')

training_data = txt_list(txt_dir)

# train a Word2Vec model using Gensim
Embedding_Dim = 100
# train word2vec model
model = gensim.models.Word2Vec(sentences=training_data, size=Embedding_Dim, workers=4, min_count=1)
# vocabulary size
words = list(model.wv.vocab)
print('Here is the Vocabulary Size.. %d' % len(words))

model.save('gum_word2vec.model', path=os.path.join(fileDir, 'bias_visualisation_app', 'resources'))
