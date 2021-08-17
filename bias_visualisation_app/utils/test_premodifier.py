
import nltk.corpus as nc
import nltk
import spacy

SUBJECTS = ['nsubj', 'nsubjpass', 'csubj', 'csubjpass', 'agent', 'expl', 'compounds', 'pobj']
OBJECTS = ['dobj', 'dative', 'attr', 'oprd']
ADJECTIVES = ['acomp', 'advcl', 'advmod', 'amod', 'appos', 'nn', 'nmod', 'ccomp', 'complm',
              'hmod', 'infmod', 'xcomp', 'rcmod', 'poss', 'possessive']
COMPOUNDS = ['compound']
PREPOSITIONS = ['prep']

male_names = nc.names.words('male.txt')
male_names.extend(['he', 'He', 'him', 'Him', 'himself', 'man', 'men', 'gentleman', 'gentlemen'])
female_names = nc.names.words('female.txt')
female_names.extend(['she', 'She', 'her', 'Her', 'herself', 'Herself', 'woman', 'Woman', 'women', 'Women'])

def getPremodifiers(toks):
    toks_with_premodifiers = []
    for tok in toks:
        premodifiers = [left for left in tok.lefts if left.dep_ in ADJECTIVES]
        premodifiers.append(tok)
        premodifiers = [str(token) for token in premodifiers]
        toks_with_premodifiers.append(premodifiers)

    return toks_with_premodifiers



sentence = 'Lucy eats a tasty black bread. The elegant powerful woman wears shiny black glasses. The dark tall man drinks water. He admires vulnerable strong women. The kind boy picks a cup. Our women are strong. I hate their men.'


def determine_gender_premodifier_test(input_data):
    parser = spacy.load('en_core_web_md', disable=['ner', 'textcat'])
    input_data = input_data.lower()
    sent_text = nltk.sent_tokenize(input_data)
    tot_female_premodifier_list = []
    tot_male_premodifier_list = []
    for sentence in sent_text:
        parse = parser(sentence)
        try:
            female_premodifier_list, male_premodifier_list = findpremodifiers(parse)
            tot_female_premodifier_list.extend(female_premodifier_list)
            tot_male_premodifier_list.extend(male_premodifier_list)
        except:
            continue
    return tot_female_premodifier_list, tot_male_premodifier_list

noun_list = ['NOUN', 'PRON' 'PROPN', 'NN', 'NNP', 'NNS', 'NNPS']

def findpremodifiers(tokens):
    nouns = [tok for tok in tokens if tok.pos_ in noun_list]
    premodifier_noun_pair = getPremodifiers(nouns)
    female_premodifier_list, male_premodifier_list = gender_premodifiers(premodifier_noun_pair)
    return female_premodifier_list, male_premodifier_list

def gender_premodifiers(premodifier_noun_pair):
    female_premodifier_list = []
    male_premodifier_list = []
    for pair in premodifier_noun_pair:
        noun = pair[-1]
        if noun in female_names or 'girl' in noun or 'woman' in noun or 'mrs' in noun or 'miss' in noun:
            premodifiers = pair[:-1]
            female_premodifier_list.extend(premodifiers)
        elif noun in male_names or 'boy' in noun or (
                'man' in noun and 'woman' not in noun) or 'mr' in noun or 'mister' in noun:
            premodifiers = pair[:-1]
            male_premodifier_list.extend(premodifiers)
        else:
            continue

    return female_premodifier_list, male_premodifier_list






print(determine_gender_premodifier_test(sentence))