from os import path
from gensim.models import KeyedVectors, Word2Vec


model_path=path.join(path.dirname(__file__), "./data/gum_word2vec.model")
model = Word2Vec.load(model_path)

#Finding similar words
print(model.wv.most_similar('woman'))
print(model.wv['woman'])
print(model.wv['man'])