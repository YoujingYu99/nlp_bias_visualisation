
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
female_names.extend(['she', 'She', 'her', 'Her', 'herself', 'Herself'])

def getAdjectives(toks):
    first_modifier_list = []
    for tok in toks:
        print(tok)
        print(tok.lefts)
        adjs = [left for left in tok.lefts if left.dep_ in ADJECTIVES]
        first_modifier_list.append(adjs[-1])

    return first_modifier_list

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


sentence = 'Lucy eats a tasty black bread. The elegant powerful woman wears shiny black glasses.'


def determine_gender_modifier_test(sentence):
    parser = spacy.load('en_core_web_md', disable=['ner', 'textcat'])
    sent_text = nltk.sent_tokenize(sentence)
    tot_first_modifier_female = []
    tot_second_modifier_female = []
    tot_first_modifier_male = []
    tot_second_modifier_male = []
    for sentence in sent_text:
        parse = parser(sentence)
        try:
            first_modifier_list_female, second_modifier_list_female, first_modifier_list_male, second_modifier_list_male = findmodifiers(parse)
            tot_first_modifier_female.extend(first_modifier_list_female)
            tot_second_modifier_female.extend(second_modifier_list_male)
            tot_first_modifier_male.extend(first_modifier_list_male)
            tot_second_modifier_male.extend(second_modifier_list_male)
        except:
            continue
    print(tot_first_modifier_female)
    return tot_first_modifier_female, tot_second_modifier_female, tot_first_modifier_male, tot_second_modifier_male

noun_list = ['NOUN', 'PRON', 'PROPN', 'NN', 'NNP', 'NNS', 'NNPS']

def findmodifiers(tokens):
    nouns = [tok for tok in tokens if tok.pos_ in noun_list]
    str_nouns = [str(noun) for noun in nouns ]

    female_noun_list = [noun for noun in str_nouns if noun in female_names or 'girl' in noun or 'woman' in noun or 'mrs' in noun or 'Mrs' in noun or 'Miss' in noun or 'miss' in noun]

    male_noun_list = [noun for noun in str_nouns if noun in male_names or 'boy' in noun or ('man' in noun and 'woman' not in noun) or 'Mr' in noun or 'Mister' in noun]
    
    first_modifier_list_female, second_modifier_list_female = getAdjectives(female_noun_list)
    first_modifier_list_male, second_modifier_list_male = getAdjectives(male_noun_list)

    return first_modifier_list_female, second_modifier_list_female, first_modifier_list_male, second_modifier_list_male


determine_gender_modifier_test(sentence)
