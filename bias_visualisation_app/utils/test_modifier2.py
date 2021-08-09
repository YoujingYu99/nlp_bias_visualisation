
import nltk.corpus as nc
import nltk
import spacy

SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl", "compounds", "pobj"]
OBJECTS = ["dobj", "dative", "attr", "oprd"]
ADJECTIVES = ["acomp", "advcl", "advmod", "amod", "appos", "nn", "nmod", "ccomp", "complm",
              "hmod", "infmod", "xcomp", "rcmod", "poss", "possessive"]
COMPOUNDS = ["compound"]
PREPOSITIONS = ["prep"]

male_names = nc.names.words('male.txt')
male_names.extend(['he', 'He', 'him', 'Him', 'himself', 'Himself'])
female_names = nc.names.words('female.txt')
female_names.extend(['she', 'She', 'her', 'Her', 'herself', 'Herself', 'woman', 'Woman', 'women', 'Women'])

def getAdjectives(toks):
    toks_with_adjectives = []
    for tok in toks:
        adjs = [left for left in tok.lefts if left.dep_ in ADJECTIVES]
        adjs.append(tok)
        adjs = [str(token) for token in adjs]
        toks_with_adjectives.append(adjs)

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


sentence = 'Lucy eats a tasty black bread. The elegant powerful woman wears shiny black glasses. The dark tall man drinks water. He admires vulnerable strong women. The kind beautiful girl picks a cup.'


def determine_gender_modifier_test(sentence):
    parser = spacy.load('en_core_web_md', disable=['ner', 'textcat'])
    sent_text = nltk.sent_tokenize(sentence)
    tot_female_adj_list = []
    tot_male_adj_list = []
    for sentence in sent_text:
        parse = parser(sentence)
        try:
            female_adj_list, male_adj_list = findmodifiers(parse)
            tot_female_adj_list.extend(female_adj_list)
            tot_male_adj_list.extend(male_adj_list)
        except:
            continue
    return tot_female_adj_list, male_adj_list
noun_list = ['NOUN', 'PRON' 'PROPN', 'NN', 'NNP', 'NNS', 'NNPS']

def findmodifiers(tokens):
    nouns = [tok for tok in tokens if tok.pos_ in noun_list]
    adj_noun_pair = getAdjectives(nouns)
    female_adj_list, male_adj_list = gender_adjs(adj_noun_pair)
    return female_adj_list, male_adj_list

def gender_adjs(adj_noun_pair):
    female_adj_list = []
    male_adj_list = []
    for pair in adj_noun_pair:
        noun = pair[-1]
        if noun in female_names or 'girl' in noun or 'woman' in noun or 'mrs' in noun or 'Mrs' in noun or 'Miss' in noun or 'miss' in noun:
            adjs = pair[:-1]
            female_adj_list.extend(adjs)
        elif noun in male_names or 'boy' in noun or ('man' in noun and 'woman' not in noun) or 'Mr' in noun or 'Mister' in noun:
            adjs = pair[:-1]
            male_adj_list.extend(adjs)
        else:
            continue

    return female_adj_list, male_adj_list







determine_gender_modifier_test(sentence)