from nltk.stem.wordnet import WordNetLemmatizer
import nltk.corpus as nc
import nltk
import spacy
import pandas as pd

SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl", "compounds", "pobj"]
OBJECTS = ["dobj", "dative", "attr", "oprd"]
ADJECTIVES = ["acomp", "advcl", "advmod", "amod", "appos", "nn", "nmod", "ccomp", "complm",
              "hmod", "infmod", "xcomp", "rcmod", "poss", "possessive"]
COMPOUNDS = ["compound"]
PREPOSITIONS = ["prep"]

male_names = nc.names.words('male.txt')
male_names.extend(['he', 'He', 'him', 'Him', 'himself', 'Himself'])
female_names = nc.names.words('female.txt')
female_names.extend(['she', 'She', 'her', 'Her', 'herself', 'Herself'])

def getAdjectives(toks):
    toks_with_adjectives = []
    for tok in toks:
        adjs = [left for left in tok.lefts if left.dep_ in ADJECTIVES]
        adjs.append(tok)
        adjs.extend([right for right in tok.rights if tok.dep_ in ADJECTIVES])
        tok_with_adj = " ".join([adj.lower_ for adj in adjs])
        toks_with_adjectives.extend(adjs)

    return toks_with_adjectives

def generate_left_right_adjectives(obj):
    obj_desc_tokens = []
    for tok in obj.lefts:
        if tok.dep_ in ADJECTIVES:
            obj_desc_tokens.extend(generate_left_right_adjectives(tok))
    obj_desc_tokens.append(obj)

    for tok in obj.rights:
        if tok.dep_ in ADJECTIVES:
            obj_desc_tokens.extend(generate_left_right_adjectives(tok))

    return obj_desc_tokens

def determine_gender(token):
    if token == 'nothing':
        gender = 'neutral_intransitive'
    elif token in female_names or 'girl' in token or 'woman' in token or 'mrs' in token or 'Mrs' in token or 'Miss' in token or 'miss' in token:
        gender = 'female'
    elif token in male_names or 'boy' in token or (
            'man' in token and 'woman' not in token) or 'Mr' in token or 'Mister' in token:
        gender = 'male'
    elif token in neutral_sub_list:
        gender = 'neutral'
    else:
        gender = 'neutral'
    return gender

sentence = 'She takes off her coat. Hilary is supported. Mary smiles. Lucy has been smiling. Linda eats bread. '


def determine_gender_modifier_test(sentence):
    parser = spacy.load('en_core_web_md', disable=['ner', 'textcat'])

    sent_text = nltk.sent_tokenize(sentence)
    for sentence in sent_text:
        parse = parser(sentence)
        print(parse)
        try:
            modifier_list = findmodifiers(parse)
            print(modifier_list)
        except:
            continue

noun_list = ['NOUN', 'PRON' 'PROPN', 'NN', 'NNP', 'NNS', 'NNPS']

def findmodifiers(tokens):
    for tok in tokens:
        print(tok.pos_)
    nouns = [tok for tok in tokens if tok.pos_ in noun_list]
    print(nouns)
    adj_noun_pair = getAdjectives(nouns)
    return adj_noun_pair


determine_gender_modifier_test(sentence)