import nltk.corpus as nc
import nltk
import spacy


SUBJECTS = ['nsubj', 'nsubjpass', 'csubj', 'csubjpass', 'agent', 'expl', 'compounds', 'pobj']
OBJECTS = ['dobj', 'dative', 'attr', 'oprd']
ADJECTIVES = ['acomp', 'advcl', 'advmod', 'amod', 'appos', 'nn', 'nmod', 'ccomp', 'complm',
              'hmod', 'infmod', 'xcomp', 'rcmod', 'poss', 'possessive']
COMPOUNDS = ['compound']
PREPOSITIONS = ['prep']



aux_word_list = ['are', 'is', 'were', 'was', 'be']
neg_word_list = ['not', "n't"]
det_word_list = ['a', 'an']

male_nouns = nc.names.words('male.txt')
male_nouns.extend(['he', 'him', 'himself', 'gentleman', 'gentlemen', 'man', 'men', 'male'])
male_nouns = [x.lower() for x in male_nouns]
female_nouns = nc.names.words('female.txt')
female_nouns.extend(['she', 'her', 'herself', 'lady', 'ladys', 'woman', 'women', 'female'])
female_nouns = [x.lower() for x in female_nouns]


sentence = "Most writers are male. Most writers are not a woman. Most writers are not male. The teacher is not a man. The majority are women. The suspect is a woman. Her father was a man who lived an extraordinary life. Women are not victims. Men are minority. The woman isn't a teacher. Sarah is an engineer. The culprit is Linda."

def isNegated(tok):
    negations = {'no', 'not', "n't", 'never', 'none'}
    for dep in list(tok.lefts) + list(tok.rights):
        if dep.lower_ in negations:
            return True
    return False

def findfemalefollow_auxs(tokens):
    female_follow_aux_list = []
    for tok in tokens:
        if tok.pos_ == 'AUX' and tok.dep_ == 'ROOT':
            is_negated = isNegated(tok)
            # e.g. Most writers are female
            noun_list = [right_tok.text for right_tok in tok.rights if right_tok.text in female_nouns]
            if noun_list:
                try:
                    follow_aux_list = [left_tok.text for left_tok in tok.lefts if left_tok.dep_ == 'nsubj' and left_tok.pos_ == 'NOUN']
                    follow_aux = follow_aux_list[-1]
                    if is_negated == True:
                        female_follow_aux_list.append('!' + str(follow_aux))
                    else:
                        female_follow_aux_list.append(str(follow_aux))
                except:
                    pass
            else:
                continue

    return female_follow_aux_list

def findmalefollow_auxs(tokens):
    male_follow_aux_list = []
    for tok in tokens:
        if tok.pos_ == 'AUX' and tok.dep_ == 'ROOT':
            is_negated = isNegated(tok)
            # e.g. Most writers are male
            noun_list = [right_tok.text for right_tok in tok.rights if right_tok.text in male_nouns]
            if noun_list:
                try:
                    follow_aux_list = [left_tok.text for left_tok in tok.lefts if left_tok.dep_ == 'nsubj' and left_tok.pos_ == 'NOUN']
                    follow_aux = follow_aux_list[-1]
                    if is_negated == True:
                        male_follow_aux_list.append('!' + str(follow_aux))
                    else:
                        male_follow_aux_list.append(str(follow_aux))
                except:
                    pass
            else:
                continue

    return male_follow_aux_list

def findfemalebefore_auxs(tokens):
    female_before_aux_list = []
    for tok in tokens:
        if tok.pos_ == 'AUX' and tok.dep_ == 'ROOT':
            is_negated = isNegated(tok)
            # e.g. Women are the main source of income.
            noun_list = [left_tok.text for left_tok in tok.lefts if left_tok.text in female_nouns]
            if noun_list:
                try:
                    before_aux_list = [right_tok.text for right_tok in tok.rights if right_tok.dep_ == 'attr']
                    before_aux = before_aux_list[-1]
                    if is_negated == True:
                        female_before_aux_list.append('!' + str(before_aux))
                    else:
                        female_before_aux_list.append(str(before_aux))
                except:
                    pass
            else:
                continue

    return female_before_aux_list

def findmalebefore_auxs(tokens):
    male_before_aux_list = []
    for tok in tokens:
        if tok.pos_ == 'AUX' and tok.dep_ == 'ROOT':
            is_negated = isNegated(tok)
            # e.g. Men are the most irrational species.
            noun_list = [left_tok.text for left_tok in tok.lefts if left_tok.text in male_nouns]
            if noun_list:
                try:
                    before_aux_list = [right_tok.text for right_tok in tok.rights if right_tok.dep_ == 'attr']
                    before_aux = before_aux_list[-1]
                    if is_negated == True:
                        male_before_aux_list.append('!' + str(before_aux))
                    else:
                        male_before_aux_list.append(str(before_aux))
                except:
                    pass
            else:
                continue

    return male_before_aux_list

def determine_gender_follow_aux_test(input_data):
    input_data = input_data.lower()
    parser = spacy.load('en_core_web_md', disable=['ner', 'textcat'])
    sent_text = nltk.sent_tokenize(input_data)
    tot_female_follow_aux_list = []
    tot_male_follow_aux_list = []
    tot_female_before_aux_list = []
    tot_male_before_aux_list = []
    for sent in sent_text:
        parse = parser(sent)
        try:
            female_follow_aux_list = findfemalefollow_auxs(parse)
            male_follow_aux_list = findmalefollow_auxs(parse)
            female_before_aux_list = findfemalebefore_auxs(parse)
            male_before_aux_list = findmalebefore_auxs(parse)
            tot_female_follow_aux_list.extend(female_follow_aux_list)
            tot_male_follow_aux_list.extend(male_follow_aux_list)
            tot_female_before_aux_list.extend(female_before_aux_list)
            tot_male_before_aux_list.extend(male_before_aux_list)
        except:
            continue
    return tot_female_follow_aux_list, tot_female_before_aux_list, tot_male_follow_aux_list, tot_male_before_aux_list


print(determine_gender_follow_aux_test(sentence))