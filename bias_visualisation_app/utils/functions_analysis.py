from nltk.stem.wordnet import WordNetLemmatizer
import nltk.corpus as nc
import nltk
import spacy
import string
from io import open
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import sys
import os
from flask import url_for
from .parse_sentence import parse_sentence
from .functions_files import save_obj_text, concat_csv_excel, save_obj, load_obj, load_obj_user_uploads
from .PrecalculatedBiasCalculator import PrecalculatedBiasCalculator

# remove the . before .parse_sentence when running tests.

sys.setrecursionlimit(10000)

calculator = PrecalculatedBiasCalculator()
# NLP bias detection
# if environ.get('USE_PRECALCULATED_BIASES', '').upper() == 'TRUE':
#     print('using precalculated biases')
#     calculator = PrecalculatedBiasCalculator()
# else:
#     calculator = PcaBiasCalculator()

# set recursion limit

SUBJECTS = ['nsubj', 'nsubjpass', 'csubj', 'csubjpass', 'agent', 'expl', 'compounds', 'pobj']
OBJECTS = ['dobj', 'dative', 'attr', 'oprd']
ADJECTIVES = ['acomp', 'advcl', 'advmod', 'amod', 'appos', 'nn', 'nmod', 'ccomp', 'complm',
              'hmod', 'infmod', 'xcomp', 'rcmod', 'poss', 'possessive']
COMPOUNDS = ['compound']
PREPOSITIONS = ['prep']

# POS tagging for different word types
adj_list = ['ADJ', 'ADV', 'ADP', 'JJ', 'JJR', 'JJS']
noun_list = ['NOUN', 'PRON' 'PROPN', 'NN', 'NNP', 'NNS', 'NNPS']
verb_list = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'VERB']
non_sub_pos = ['DET', 'AUX']

neutral_words = [
    'is',
    'was',
    'who',
    'what',
    'where',
    'the',
    'it',
]

def getSubsFromConjunctions(subs):
    moreSubs = []
    for sub in subs:
        # rights is a generator
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        # look for multiple subjects
        if 'and' in rightDeps:
            moreSubs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == 'NOUN'])
            if len(moreSubs) > 0:
                moreSubs.extend(getSubsFromConjunctions(moreSubs))
    return moreSubs


def getObjsFromConjunctions(objs):
    moreObjs = []
    for obj in objs:
        # rights is a generator
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights}
        # look for multiple objects
        if 'and' in rightDeps:
            moreObjs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == 'NOUN'])
            if len(moreObjs) > 0:
                moreObjs.extend(getObjsFromConjunctions(moreObjs))
    return moreObjs


def getVerbsFromConjunctions(verbs):
    moreVerbs = []
    for verb in verbs:
        rightDeps = {tok.lower_ for tok in verb.rights}
        # look for multiple verbs
        if 'and' in rightDeps:
            moreVerbs.extend([tok for tok in verb.rights if tok.pos_ == 'VERB'])
            if len(moreVerbs) > 0:
                moreVerbs.extend(getVerbsFromConjunctions(moreVerbs))
    return moreVerbs


def findSubs(tok):
    head = tok.head
    while head.pos_ != 'VERB' and head.pos_ != 'NOUN' and head.head != head:
        head = head.head
    if head.pos_ == 'VERB':
        subs = [tok for tok in head.lefts if tok.dep_ == 'SUB']
        if len(subs) > 0:
            verbNegated = isNegated(head)
            subs.extend(getSubsFromConjunctions(subs))
            return subs, verbNegated
        elif head.head != head:
            return findSubs(head)
    elif head.pos_ == 'NOUN':
        return [head], isNegated(tok)
    return [], False


def isNegated(tok):
    negations = {'no', 'not', "n't", 'never', 'none'}
    for dep in list(tok.lefts) + list(tok.rights):
        if dep.lower_ in negations:
            return True
    return False


def findSVs(tokens):
    svs = []
    verbs = [tok for tok in tokens if tok.pos_ == 'VERB']
    for v in verbs:
        subs, verbNegated = getAllSubs(v)
        if len(subs) > 0:
            for sub in subs:
                svs.append((sub.orth_, '!' + v.orth_ if verbNegated else v.orth_))
    return svs


def getObjsFromPrepositions(deps):
    objs = []
    for dep in deps:
        if dep.pos_ == 'ADP' and dep.dep_ == 'prep':
            objs.extend(
                [tok for tok in dep.rights if tok.dep_ in OBJECTS or (tok.pos_ == 'PRON' and tok.lower_ == 'me')])
    return objs


def getPremodifiers(toks):
    toks_with_premodifiers = []
    for tok in toks:
        premodifiers = [left for left in tok.lefts if left.dep_ in ADJECTIVES]
        premodifiers.append(tok)
        premodifiers = [str(token) for token in premodifiers]
        toks_with_premodifiers.append(premodifiers)

    return toks_with_premodifiers


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


def getObjsFromAttrs(deps):
    for dep in deps:
        if dep.pos_ == 'NOUN' and dep.dep_ == 'attr':
            verbs = [tok for tok in dep.rights if tok.pos_ == 'VERB']
            if len(verbs) > 0:
                for v in verbs:
                    rights = list(v.rights)
                    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
                    objs.extend(getObjsFromPrepositions(rights))
                    if len(objs) > 0:
                        return v, objs
    return None, None


def getObjFromXComp(deps):
    for dep in deps:
        if dep.pos_ == 'VERB' and dep.dep_ == 'xcomp':
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(getObjsFromPrepositions(rights))
            if len(objs) > 0:
                return v, objs
    return None, None



def getAllSubs(v):
    verbNegated = isNegated(v)
    # subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS elif  type(tok.dep_) == int or float  and tok.pos_ != 'DET']
    subs = []
    for tok in v.lefts:
        if tok.dep_ in SUBJECTS and tok.pos_ not in non_sub_pos:
            subs.append(tok)
        elif type(tok.dep_) == int or float and tok.pos_ not in non_sub_pos:
            subs.append(tok)
        else:
            continue
    if len(subs) > 0:
        subs.extend(getSubsFromConjunctions(subs))

    else:
        foundSubs, verbNegated = findSubs(v)
        subs.extend(foundSubs)
    return subs, verbNegated


def getAllObjs(v):
    # rights is a generator
    rights = list(v.rights)
    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
    objs.extend(getObjsFromPrepositions(rights))

    potentialNewVerb, potentialNewObjs = getObjFromXComp(rights)
    if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
        objs.extend(potentialNewObjs)
        v = potentialNewVerb
    if len(objs) > 0:
        objs.extend(getObjsFromConjunctions(objs))
    else:
        (v, objs) = v, []
    return v, objs


def getAllObjsWithAdjectives(v):
    # rights is a generator
    rights = list(v.rights)
    objs = [tok for tok in rights if tok.dep_ in OBJECTS]

    if len(objs) == 0:
        objs = [tok for tok in rights if tok.dep_ in ADJECTIVES]

    objs.extend(getObjsFromPrepositions(rights))

    potentialNewVerb, potentialNewObjs = getObjFromXComp(rights)
    if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
        objs.extend(potentialNewObjs)
        v = potentialNewVerb
    if len(objs) > 0:
        objs.extend(getObjsFromConjunctions(objs))
    return v, objs


def findSVOs(tokens):
    svos = []
    verbs = [tok for tok in tokens if tok.pos_ == 'AUX']
    for v in verbs:
        subs, verbNegated = getAllSubs(v)
        # hopefully there are subs, if not, don't examine this verb any longer
        if len(subs) > 0:
            v, objs = getAllObjs(v)
            for sub in subs:
                for obj in objs:
                    objNegated = isNegated(obj)
                    svos.append((sub.lower_, '!' + v.lower_ if verbNegated or objNegated else v.lower_, obj.lower_))
    return svos


def findverbs(tokens):
    verbs = []
    # verbs is a list of lists, each element list contains either a single verb or a phrasal verb pair
    for tok in tokens:
        if tok.pos_ == 'VERB' and tok.dep_ != 'aux':
            # look for phrasal verbs
            try:
                particle_list = [right_tok for right_tok in tok.rights if right_tok.dep_ == 'prt']
                verbs.append([tok, particle_list[0]])
            except:
                verbs.append([tok])
    not_verbs = []
    for tok in tokens:
        if tok.pos_ == 'VERB' and tok.tag_ == 'VBN':
            try:
                # look for phrasal verbs
                particle_list = [right_tok for right_tok in tok.rights if right_tok.dep_ == 'prt']
                not_verbs.append([tok, particle_list[0]])
            except:
                not_verbs.append([tok])

    return verbs, not_verbs


def findSVAOs(tokens):
    svos = []
    verbs, not_verbs = findverbs(tokens)
    if len(not_verbs) == 0:
        for v in verbs:
            main_v = v[0]

            subs, verbNegated = getAllSubs(main_v)
            # hopefully there are subs, if not, don't examine this verb any longer
            if len(subs) > 0:
                main_v, objs = getAllObjs(main_v)
                if len(objs) > 0:
                    for sub in subs:
                        for obj in objs:
                            objNegated = isNegated(obj)
                            obj_desc_tokens = generate_left_right_adjectives(obj)
                            sub_compound = generate_sub_compound(sub)
                            svos.append((' '.join(tok.lower_ for tok in sub_compound),
                                         '!' + str(v).lower() if verbNegated or objNegated else str(v).lower(),
                                         ' '.join(tok.lower_ for tok in obj_desc_tokens)))

                if len(objs) == 0:
                    svos = [str(subs[0]), str(v)]
                    svos.append('nothing')
                    svos = tuple(svos)
                    svos = [svos]

    elif not_verbs[0] not in verbs:
        for v in verbs:
            main_v = v[0]
            subs, verbNegated = getAllSubs(main_v)
            # hopefully there are subs, if not, don't examine this verb any longer
            if len(subs) > 0:
                main_v, objs = getAllObjs(main_v)
                if len(objs) > 0:
                    for sub in subs:
                        for obj in objs:
                            objNegated = isNegated(obj)
                            obj_desc_tokens = generate_left_right_adjectives(obj)
                            sub_compound = generate_sub_compound(sub)
                            svos.append((' '.join(tok.lower_ for tok in sub_compound),
                                         '!' + str(v).lower() if verbNegated or objNegated else str(v).lower(),
                                         ' '.join(tok.lower_ for tok in obj_desc_tokens)))

                if len(objs) == 0:
                    svos = [str(subs[0]), str(v)]
                    svos.append('nothing')
                    svos = tuple(svos)
                    svos = [svos]

    else:
        # new_verbs = [tok for tok in tokens if tok.pos_ == 'VERB' and tok.tag_ == 'VBN']
        new_verbs = []
        for tok in tokens:
            if tok.pos_ == 'VERB' and tok.tag_ == 'VBN':
                try:
                    # look for phrasal verbs
                    particle_list = [right_tok for right_tok in tok.rights if right_tok.dep_ == 'prt']
                    new_verbs.append([tok, particle_list[0]])
                except:
                    new_verbs.append([tok])
        tokens_new = [t for t in tokens]
        tokens_new_str = [str(t) for t in tokens]
        for v in new_verbs:
            main_v = v[0]
            new_objs, new_verbNegated = getAllSubs(main_v)
            get_index = tokens_new_str.index(str(main_v))
            after_tok_list = tokens_new[get_index + 1:]
            after_tok_list_str = tokens_new_str[get_index + 1:]
            if 'by' in after_tok_list_str:
                new_subs = []
                for after_tok in after_tok_list:
                    if after_tok.dep_ in SUBJECTS and after_tok.pos_ not in non_sub_pos:
                        new_subs.append(after_tok)
                    elif type(after_tok.dep_) == int or float and after_tok.pos_ not in non_sub_pos:
                        new_subs.append(after_tok)
                # 'by' is at position 0
                new_sub = new_subs[1]
                svos = [str(new_sub), str(v.lower()), str(new_objs[0])]
                svos = tuple(svos)
                svos = [svos]

            else:
                svos = ['neutral', str(v.lower()), str(new_objs[0])]
                svos = tuple(svos)
                svos = [svos]

    return svos


def generate_sub_compound(sub):
    sub_compunds = []
    for tok in sub.lefts:
        if tok.dep_ in COMPOUNDS:
            sub_compunds.extend(generate_sub_compound(tok))
    sub_compunds.append(sub)
    for tok in sub.rights:
        if tok.dep_ in COMPOUNDS:
            sub_compunds.extend(generate_sub_compound(tok))
    return sub_compunds


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


male_names = nc.names.words('male.txt')
male_names.extend(['he', 'him', 'himself', 'man', 'men', 'gentleman', 'gentlemen'])
female_names = nc.names.words('female.txt')
female_names.extend(['she', 'her', 'herself', 'woman', 'women', 'lady'])
neutral_sub_list = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours', 'it', 'its', 'they', 'them', 'their', 'theirs', 'neutral']

spec_chars = ['!', ''','#','%','&',''', '(', ')',
              '*', '+', ',', '-', '.', '/', ':', ';', '<',
              '=', '>', '?', '@', '[', '\\', ']', '^', '_',
              '`', '{', '|', '}', '~', '–']



def clean_SVO_dataframe(SVO_df):
    for char in spec_chars:
        SVO_df['subject'] = SVO_df['subject'].str.replace(char, ' ')
        SVO_df['object'] = SVO_df['object'].str.replace(char, ' ')

    # get base form of verb
    verb_list = SVO_df['verb'].to_list()
    verb_base_list = []
    for verb in verb_list:
        verb = verb.replace("[", '').replace("]", "")
        verb.strip()
        if '!' in verb:
            verb = verb.replace('!', '')
            verb.strip()
            try:
                main_verb, particle = verb.split()[0], verb.split()[1]
                base_word = WordNetLemmatizer().lemmatize(main_verb, 'v')
                base_word.strip()
                base_phrasal_verb = '!' + base_word + ' ' + particle
                verb_base_list.append(base_phrasal_verb)
            except:
                verb = verb.split()[0]
                base_word = WordNetLemmatizer().lemmatize(verb, 'v')
                base_word.strip()
                verb_base_list.append('!' + base_word)
        else:
            verb.strip()
            try:
                main_verb, particle = verb.split()[0], verb.split()[1]
                base_word = WordNetLemmatizer().lemmatize(main_verb, 'v')
                base_word.strip()
                base_phrasal_verb = base_word + ' ' + particle
                verb_base_list.append(base_phrasal_verb)
            except:
                verb = verb.split()[0]
                base_word = WordNetLemmatizer().lemmatize(verb, 'v')
                base_word.strip()
                verb_base_list.append(base_word)

    SVO_df['verb'] = verb_base_list
    SVO_df = SVO_df.apply(lambda x: x.astype(str).str.lower())

    return SVO_df



def clean_premodifier_dataframe(premodifier_df):
    for char in spec_chars:
        premodifier_df['female_premodifier'] = premodifier_df['female_premodifier'].astype(str).str.replace(char, ' ')
        premodifier_df['male_premodifier'] = premodifier_df['male_premodifier'].astype(str).str.replace(char, ' ')

    # get base form of words
    female_premodifier_list = premodifier_df['female_premodifier'].to_list()
    female_premodifier_base_list = []
    for premodifier in female_premodifier_list:
        base_word = WordNetLemmatizer().lemmatize(premodifier)
        base_word.strip()
        female_premodifier_base_list.append(base_word)
    premodifier_df['female_premodifier'] = female_premodifier_base_list

    male_premodifier_list = premodifier_df['male_premodifier'].to_list()
    male_premodifier_base_list = []
    for premodifier in male_premodifier_list:
        base_word = WordNetLemmatizer().lemmatize(premodifier)
        base_word.strip()
        male_premodifier_base_list.append(base_word)
    premodifier_df['male_premodifier'] = male_premodifier_base_list

    premodifier_df = premodifier_df.apply(lambda x: x.astype(str).str.lower())

    return premodifier_df


def clean_postmodifier_dataframe(postmodifier_df):
    for char in spec_chars:
        postmodifier_df['female_postmodifier'] = postmodifier_df['female_postmodifier'].astype(str).str.replace(char, ' ')
        postmodifier_df['male_postmodifier'] = postmodifier_df['male_postmodifier'].astype(str).str.replace(char, ' ')

    # get base form of words
    female_postmodifier_list = postmodifier_df['female_postmodifier'].to_list()
    female_postmodifier_base_list = []
    for postmodifier in female_postmodifier_list:
        base_word = WordNetLemmatizer().lemmatize(postmodifier)
        base_word.strip()
        female_postmodifier_base_list.append(base_word)
    postmodifier_df['female_postmodifier'] = female_postmodifier_base_list

    male_postmodifier_list = postmodifier_df['male_postmodifier'].to_list()
    male_postmodifier_base_list = []
    for postmodifier in male_postmodifier_list:
        base_word = WordNetLemmatizer().lemmatize(postmodifier)
        base_word.strip()
        male_postmodifier_base_list.append(base_word)
    postmodifier_df['male_postmodifier'] = male_postmodifier_base_list

    postmodifier_df = postmodifier_df.apply(lambda x: x.astype(str).str.lower())

    return postmodifier_df


def clean_aux_dataframe(aux_df):

    # get base form of words
    female_before_aux_list = aux_df['female_before_aux'].to_list()
    female_before_aux_base_list = []
    for aux in female_before_aux_list:
        aux = str(aux)
        try:
            if '!' in aux:
                aux = aux.replace('!', '')
                aux.strip()
                base_word = WordNetLemmatizer().lemmatize(aux)
                base_word.strip()
                female_before_aux_base_list.append('!' + base_word)
            else:
                base_word = WordNetLemmatizer().lemmatize(aux)
                base_word.strip()
                female_before_aux_base_list.append(base_word)
        except:
            continue
    aux_df['female_before_aux'] = female_before_aux_base_list

    male_before_aux_list = aux_df['male_before_aux'].to_list()
    male_before_aux_base_list = []
    for aux in male_before_aux_list:
        aux = str(aux)
        try:
            if '!' in aux:
                aux = aux.replace('!', '')
                aux.strip()
                base_word = WordNetLemmatizer().lemmatize(aux)
                base_word.strip()
                male_before_aux_base_list.append('!' + base_word)
            else:
                base_word = WordNetLemmatizer().lemmatize(aux)
                base_word.strip()
                male_before_aux_base_list.append(base_word)
        except:
            continue
    aux_df['male_before_aux'] = male_before_aux_base_list

    # get base form of words
    female_follow_aux_list = aux_df['female_follow_aux'].to_list()
    female_follow_aux_base_list = []
    for aux in female_follow_aux_list:
        aux = str(aux)
        try:
            if '!' in aux:
                aux = aux.replace('!', '')
                aux.strip()
                base_word = WordNetLemmatizer().lemmatize(aux)
                base_word.strip()
                female_follow_aux_base_list.append('!' + base_word)
            else:
                base_word = WordNetLemmatizer().lemmatize(aux)
                base_word.strip()
                female_follow_aux_base_list.append(base_word)
        except:
            continue
    aux_df['female_follow_aux'] = female_follow_aux_base_list

    male_follow_aux_list = aux_df['male_follow_aux'].to_list()
    male_follow_aux_base_list = []
    for aux in male_follow_aux_list:
        aux = str(aux)
        try:
            if '!' in aux:
                aux = aux.replace('!', '')
                aux.strip()
                base_word = WordNetLemmatizer().lemmatize(aux)
                base_word.strip()
                male_follow_aux_base_list.append('!' + base_word)
            else:
                base_word = WordNetLemmatizer().lemmatize(aux)
                base_word.strip()
                male_follow_aux_base_list.append(base_word)
        except:
            continue
    aux_df['male_follow_aux'] = male_follow_aux_base_list

    aux_df = aux_df.apply(lambda x: x.astype(str).str.lower())

    return aux_df


def clean_possess_dataframe(possess_df):
    for char in spec_chars:
        possess_df['female_possessive'] = possess_df['female_possessive'].astype(str).str.replace(char, ' ')
        possess_df['male_possessive'] = possess_df['male_possessive'].astype(str).str.replace(char, ' ')
        possess_df['female_possessor'] = possess_df['female_possessor'].astype(str).str.replace(char, ' ')
        possess_df['male_possessor'] = possess_df['male_possessor'].astype(str).str.replace(char, ' ')

    # get base form of words
    female_possessive_list = possess_df['female_possessive'].to_list()
    female_possessive_base_list = []
    for possessive in female_possessive_list:
        base_word = WordNetLemmatizer().lemmatize(possessive)
        base_word.strip()
        female_possessive_base_list.append(base_word)
    possess_df['female_possessive'] = female_possessive_base_list

    male_possessive_list = possess_df['male_possessive'].to_list()
    male_possessive_base_list = []
    for possessive in male_possessive_list:
        base_word = WordNetLemmatizer().lemmatize(possessive)
        base_word.strip()
        male_possessive_base_list.append(base_word)
    possess_df['male_possessive'] = male_possessive_base_list

    female_possessor_list = possess_df['female_possessor'].to_list()
    female_possessor_base_list = []
    for possessor in female_possessor_list:
        base_word = WordNetLemmatizer().lemmatize(possessor)
        base_word.strip()
        female_possessor_base_list.append(base_word)
    possess_df['female_possessor'] = female_possessor_base_list

    male_possessor_list = possess_df['male_possessor'].to_list()
    male_possessor_base_list = []
    for possessor in male_possessor_list:
        base_word = WordNetLemmatizer().lemmatize(possessor)
        base_word.strip()
        male_possessor_base_list.append(base_word)
    possess_df['male_possessor'] = male_possessor_base_list

    possess_df = possess_df.apply(lambda x: x.astype(str).str.lower())

    return possess_df


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


def determine_gender_SVO(input_data):
    parser = spacy.load('en_core_web_md', disable=['ner', 'textcat'])
    input_data = input_data.lower()
    sent_text = nltk.sent_tokenize(input_data)
    sub_list = []
    sub_gender_list = []
    verb_list = []
    obj_list = []
    obj_gender_list = []
    # now loop over each sentence and tokenize it separately
    for sentence in sent_text:
        parse = parser(sentence)
        try:
            SVO_list = findSVAOs(parse)
            for i in SVO_list:
                sub, verb, obj = i[0], i[1], i[2]
                sub_gender = determine_gender(sub)
                obj_gender = determine_gender(obj)

                sub_list.append(sub)
                sub_gender_list.append(sub_gender)
                verb_list.append(verb)
                obj_list.append(obj)
                obj_gender_list.append(obj_gender)

        except:
            continue

    SVO_df = pd.DataFrame(list(zip(sub_list, sub_gender_list, verb_list, obj_list, obj_gender_list)),
                          columns=['subject', 'subject_gender', 'verb', 'object', 'object_gender'])

    SVO_df = clean_SVO_dataframe(SVO_df)
    return SVO_df


def determine_gender_premodifier(input_data):
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
    list_of_series = [pd.Series(tot_female_premodifier_list), pd.Series(tot_male_premodifier_list)]

    premodifier_df = pd.concat(list_of_series, axis=1)
    premodifier_df.columns = ['female_premodifier', 'male_premodifier']

    premodifier_df = clean_premodifier_dataframe(premodifier_df)

    return premodifier_df


def determine_gender_postmodifier(input_data):
    input_data = input_data.lower()
    sent_text = nltk.sent_tokenize(input_data)
    tot_female_postmodifier_list = []
    tot_male_postmodifier_list = []
    for sent in sent_text:
        try:
            female_postmodifier_list = findfemalePostmodifiers(sent)
            male_postmodifier_list = findmalePostmodifiers(sent)
            tot_female_postmodifier_list.extend(female_postmodifier_list)
            tot_male_postmodifier_list.extend(male_postmodifier_list)
        except:
            continue

    list_of_series = [pd.Series(tot_female_postmodifier_list), pd.Series(tot_male_postmodifier_list)]

    postmodifier_df = pd.concat(list_of_series, axis=1)
    postmodifier_df.columns = ['female_postmodifier', 'male_postmodifier']

    postmodifier_df = clean_postmodifier_dataframe(postmodifier_df)

    return postmodifier_df


post_modifiers = ['compounds', 'pobj']
post_modifiers_noun_list = ['women', 'female', 'men', 'male']
female_postmodifier_noun_list = ['women', 'female']
male_postmodifier_noun_list = ['men', 'male']


def findfemalePostmodifiers(sent):
    tokens = nltk.word_tokenize(sent)
    tags = nltk.pos_tag(tokens)
    female_postmodifier_list = []
    for female_noun in female_postmodifier_noun_list:
        try:
            # You are interested in splitting the sentence here
            sentence_split = tokens.index(female_noun)
            # Find the words where tag meets your criteria (must be a noun / adjective)
            nouns_before_split = [word for (word, tag) in tags[sentence_split + 1: sentence_split + 2] if tag.startswith(('NN', 'JJ'))]
            post_modifier = nouns_before_split[0]
            female_postmodifier_list.append(post_modifier)
        except:
            continue

    return female_postmodifier_list

def findmalePostmodifiers(sent):
    tokens = nltk.word_tokenize(sent)
    tags = nltk.pos_tag(tokens)
    male_postmodifier_list = []
    for male_noun in male_postmodifier_noun_list:
        try:
            # You are interested in splitting the sentence here
            sentence_split = tokens.index(male_noun)
            # Find the words where tag meets your criteria (must be a noun / proper noun)
            nouns_before_split = [word for (word, tag) in tags[sentence_split + 1:] if tag.startswith(('NN', 'JJ'))]
            post_modifier = nouns_before_split[0]
            male_postmodifier_list.append(post_modifier)
        except:
            continue

    return male_postmodifier_list


aux_word_list = ['are', 'is', 'were', 'was', 'be']
det_word_list = ['a', 'an', 'the']
neg_word_list = ['not', "n't"]

male_nouns = nc.names.words('male.txt')
male_nouns.extend(['he', 'him', 'himself', 'gentleman', 'gentlemen', 'man', 'men', 'male'])
male_nouns = [x.lower() for x in male_nouns]
female_nouns = nc.names.words('female.txt')
female_nouns.extend(['she', 'her', 'herself', 'lady', 'ladys', 'woman', 'women', 'female'])
female_nouns = [x.lower() for x in female_nouns]


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

def determine_gender_aux(input_data):
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

    list_of_series = [pd.Series(tot_female_before_aux_list), pd.Series(tot_male_before_aux_list),
                      pd.Series(tot_female_follow_aux_list), pd.Series(tot_male_follow_aux_list)]

    aux_df = pd.concat(list_of_series, axis=1)
    aux_df.columns = ['female_before_aux', 'male_before_aux', 'female_follow_aux', 'male_follow_aux']
    aux_df = clean_aux_dataframe(aux_df)

    return aux_df


def findfemale_possessives(sent):
    # e.g. women's rights
    tokens = nltk.word_tokenize(sent)
    tags = nltk.pos_tag(tokens)
    female_possessive_list = []
    for female_noun in female_nouns:
        try:
            # You are interested in splitting the sentence here
            sentence_split = tokens.index(female_noun)
            # Find the words where tag meets your criteria (must be a noun / proper noun)
            if tokens[sentence_split + 1] == "'s":
                possessive = tokens[sentence_split + 2]
                if tags[sentence_split + 2][1].startswith('NN'):
                    female_possessive_list.append(possessive)
            else:
                pass
        except:
            continue

    return female_possessive_list


def findfemale_possessors(sent):
    # e.g. Norway's women
    tokens = nltk.word_tokenize(sent)
    tags = nltk.pos_tag(tokens)
    female_possessor_list = []
    for female_noun in female_nouns:
        try:
            # You are interested in splitting the sentence here
            sentence_split = tokens.index(female_noun)
            # Find the words where tag meets your criteria (must be a noun / proper noun)
            if tokens[sentence_split - 1] == "'s":
                possessor = tokens[sentence_split - 2]
                if tags[sentence_split - 2][1].startswith('NN'):
                    female_possessor_list.append(possessor)
            else:
                pass
        except:
            continue

    return female_possessor_list


def findmale_possessives(sent):
    # e.g. men's shoes
    tokens = nltk.word_tokenize(sent)
    tags = nltk.pos_tag(tokens)
    male_possessive_list = []
    for male_noun in male_nouns:
        try:
            # You are interested in splitting the sentence here
            sentence_split = tokens.index(male_noun)
            # Find the words where tag meets your criteria (must be a noun / proper noun)
            if tokens[sentence_split + 1] == "'s":
                possessive = tokens[sentence_split + 2]
                if tags[sentence_split + 2][1].startswith('NN'):
                    male_possessive_list.append(possessive)
            else:
                pass
        except:
            continue

    return male_possessive_list


def findmale_possessors(sent):
    # e.g. Norway's women
    tokens = nltk.word_tokenize(sent)
    tags = nltk.pos_tag(tokens)
    male_possessor_list = []
    for male_noun in male_nouns:
        try:
            # You are interested in splitting the sentence here
            sentence_split = tokens.index(male_noun)
            # Find the words where tag meets your criteria (must be a noun / proper noun)
            if tokens[sentence_split - 1] == "'s":
                possessor = tokens[sentence_split - 2]
                if tags[sentence_split - 2][1].startswith('NN'):
                    male_possessor_list.append(possessor)
            else:
                pass
        except:
            continue

    return male_possessor_list


def determine_gender_possess(input_data):
    input_data = input_data.lower()
    sent_text = nltk.sent_tokenize(input_data)
    tot_female_possessive_list = []
    tot_male_possessive_list = []
    tot_female_possessor_list = []
    tot_male_possessor_list = []
    for sent in sent_text:
        try:
            female_possessive_list = findfemale_possessives(sent)
            male_possessive_list = findmale_possessives(sent)
            female_possessor_list = findfemale_possessors(sent)
            male_possessor_list = findmale_possessors(sent)
            tot_female_possessive_list.extend(female_possessive_list)
            tot_male_possessive_list.extend(male_possessive_list)
            tot_female_possessor_list.extend(female_possessor_list)
            tot_male_possessor_list.extend(male_possessor_list)
        except:
            continue

    list_of_series = [pd.Series(tot_female_possessive_list), pd.Series(tot_male_possessive_list),
                      pd.Series(tot_female_possessor_list), pd.Series(tot_male_possessor_list)]
    possess_df = pd.concat(list_of_series, axis=1)
    possess_df.columns = ['female_possessive', 'male_possessive', 'female_possessor', 'male_possessor']

    possess_df = clean_possess_dataframe(possess_df)

    return possess_df


def gender_count(input_data):
    input_data = input_data.lower()
    sent_text = nltk.sent_tokenize(input_data)
    tot_count_female = 0
    tot_count_male = 0
    for sent in sent_text:
        tokens = nltk.word_tokenize(sent)
        for tok in tokens:
            if tok in female_nouns:
                tot_count_female += 1
            if tok in male_nouns:
                tot_count_male += 1

    data = [[tot_count_female, tot_count_male]]
    gender_count_df = pd.DataFrame(data, columns=['female_count', 'male_count'])

    return gender_count_df


def list_to_dataframe(view_results, scale_range=(-1, 1)):
    # put into a dataframe
    df = pd.DataFrame(view_results)
    # remove None
    df = df.dropna()
    # Normalise to -1 an 1
    scaler = MinMaxScaler(feature_range=scale_range)
    # lemmatize the tokens
    # get base form of token
    tok_list = df['token'].to_list()
    tok_base_list = []
    for tok in tok_list:
        base_word = WordNetLemmatizer().lemmatize(tok)
        base_word.strip()
        tok_base_list.append(base_word)

    df['token'] = tok_base_list
    df['bias'] = scaler.fit_transform(df[['bias']])
    df.drop_duplicates(subset='token',
                       keep=False, inplace=True)

    return df


def generate_list(dataframe):
    token_list = dataframe['token'].to_list()
    value_list = dataframe['bias'].to_list()
    pos_list = dataframe['pos'].to_list()

    return token_list, value_list, pos_list


def token_by_gender(token_list, value_list):
    # data
    # to convert lists to dictionary
    data = dict(zip(token_list, value_list))
    data = {k: v or 0 for (k, v) in data.items()}

    # separate into male and female dictionaries
    male_token = [k for (k, v) in data.items() if v > 0]
    female_token = [k for (k, v) in data.items() if v < 0]

    return female_token, male_token


def dataframe_by_gender(view_df):
    # selecting rows based on condition
    female_dataframe = view_df[view_df['bias'] < 0]
    male_dataframe = view_df[view_df['bias'] > 0]

    return female_dataframe, male_dataframe


def generate_bias_values(input_data):
    objs = parse_sentence(input_data)
    results = []
    view_results = []
    for obj in objs:
        token_result = {
            'token': obj['text'],
            'bias': calculator.detect_bias(obj['text']),
            'parts': [
                {
                    'whitespace': token.whitespace_,
                    'pos': token.pos_,
                    'dep': token.dep_,
                    'ent': token.ent_type_,
                    'skip': token.pos_
                            in ['AUX', 'ADP', 'PUNCT', 'SPACE', 'DET', 'PART', 'CCONJ']
                            or len(token) < 2
                            or token.text.lower() in neutral_words,
                }
                for token in obj['tokens']
            ],
        }
        results.append(token_result)
    # copy results and only keep the word and the bias value
    token_result2 = results.copy()
    for item in token_result2:
        if 'parts' in item.keys():
            if item['parts'][0]['pos'] in adj_list or item['parts'][0]['pos'] in noun_list or item['parts'][0][
                'pos'] in verb_list:
                item['pos'] = item['parts'][0]['pos']
            del item['parts']
        else:
            continue
        view_results.append(item)


    view_df = list_to_dataframe(view_results)
    save_obj_text(view_df, name='total_dataframe')

    SVO_df = determine_gender_SVO(input_data)
    save_obj_text(SVO_df, name='SVO_dataframe')

    premodifier_df = determine_gender_premodifier(input_data)
    save_obj_text(premodifier_df, name='premodifier_dataframe')

    postmodifier_df = determine_gender_postmodifier(input_data)
    save_obj_text(postmodifier_df, name='postmodifier_dataframe')

    aux_df = determine_gender_aux(input_data)
    save_obj_text(aux_df, name='aux_dataframe')

    possess_df = determine_gender_possess(input_data)
    save_obj_text(possess_df, name='possess_dataframe')

    profession_df = determine_gender_professions(view_df)
    save_obj_text(profession_df, name='profession_dataframe')

    gender_count_df = gender_count(input_data)
    save_obj_text(gender_count_df, name='gender_count_dataframe')

    parent_path = os.path.dirname(os.getcwd())
    csv_path = os.path.join(parent_path, 'visualising_data_bias', 'bias_visualisation_app', 'static', 'user_downloads')
    concat_csv_excel(csv_path)


def frame_from_file(view_df):
    token_list, value_list, pos_list = generate_list(view_df)
    return view_df, (token_list, value_list)


def SVO_analysis(view_df):
    # columns = ['subject', 'subject_gender', 'verb', 'object', 'object_gender']
    female_sub_df = view_df.loc[view_df['subject_gender'] == 'female']
    female_obj_df = view_df.loc[view_df['object_gender'] == 'female']
    intran_df = view_df.loc[view_df['object_gender'] == 'neutral_intransitive']

    male_sub_df = view_df.loc[view_df['subject_gender'] == 'male']
    male_obj_df = view_df.loc[view_df['object_gender'] == 'male']
    female_intran_df = intran_df.loc[intran_df['subject_gender'] == 'female']
    male_intran_df = intran_df.loc[intran_df['subject_gender'] == 'male']

    # female_sub_df['Frequency'] = female_sub_df.groupby('subject').transform('count')
    # df['frequency'] = df['county'].map(df['county'].value_counts())
    female_sub_df_new = female_sub_df.copy()
    female_sub_df_new['Frequency'] = female_sub_df_new['verb'].map(female_sub_df_new['verb'].value_counts())
    female_sub_df_new.sort_values('Frequency', inplace=True, ascending=False)
    female_sub_df_new.drop(columns=['subject', 'subject_gender', 'object', 'object_gender'], inplace=True)
    female_sub_df_new.drop_duplicates(subset='verb',
                                      keep=False, inplace=True)

    female_obj_df_new = female_obj_df.copy()
    female_obj_df_new['Frequency'] = female_obj_df_new['verb'].map(female_obj_df_new['verb'].value_counts())
    female_obj_df_new.sort_values('Frequency', inplace=True, ascending=False)
    female_obj_df_new.drop(columns=['subject', 'subject_gender', 'object', 'object_gender'], inplace=True)
    female_obj_df_new.drop_duplicates(subset='verb',
                                      keep=False, inplace=True)

    male_sub_df_new = male_sub_df.copy()
    male_sub_df_new['Frequency'] = male_sub_df_new['verb'].map(male_sub_df_new['verb'].value_counts())
    male_sub_df_new.sort_values('Frequency', inplace=True, ascending=False)
    male_sub_df_new.drop(columns=['subject', 'subject_gender', 'object', 'object_gender'], inplace=True)
    male_sub_df_new.drop_duplicates(subset='verb',
                                    keep=False, inplace=True)

    male_obj_df_new = male_obj_df.copy()
    male_obj_df_new['Frequency'] = male_obj_df_new['verb'].map(male_obj_df_new['verb'].value_counts())
    male_obj_df_new.sort_values('Frequency', inplace=True, ascending=False)
    male_obj_df_new.drop(columns=['subject', 'subject_gender', 'object', 'object_gender'], inplace=True)
    male_obj_df_new.drop_duplicates(subset='verb',
                                    keep=False, inplace=True)

    female_intran_df_new = female_intran_df.copy()
    female_intran_df_new['Frequency'] = female_intran_df_new['verb'].map(female_intran_df_new['verb'].value_counts())
    female_intran_df_new.sort_values('Frequency', inplace=True, ascending=False)
    female_intran_df_new.drop(columns=['subject', 'subject_gender', 'object', 'object_gender'], inplace=True)
    female_intran_df_new.drop_duplicates(subset='verb',
                                         keep=False, inplace=True)

    male_intran_df_new = male_intran_df.copy()
    male_intran_df_new['Frequency'] = male_intran_df_new['verb'].map(male_intran_df_new['verb'].value_counts())
    male_intran_df_new.sort_values('Frequency', inplace=True, ascending=False)
    male_intran_df_new.drop(columns=['subject', 'subject_gender', 'object', 'object_gender'], inplace=True)
    male_intran_df_new.drop_duplicates(subset='verb',
                                       keep=False, inplace=True)

    return female_sub_df_new, female_obj_df_new, female_intran_df_new, male_sub_df_new, male_obj_df_new, male_intran_df_new


def premodifier_analysis(view_df):
    # columns = ['female_adj', 'male_adj']
    female_premodifier_df = view_df.drop('male_premodifier', axis=1)
    male_premodifier_df = view_df.drop('female_premodifier', axis=1)
    female_premodifier_new = female_premodifier_df.copy()
    female_premodifier_new['Frequency'] = female_premodifier_new['female_premodifier'].map(
        female_premodifier_new['female_premodifier'].value_counts())
    female_premodifier_new.sort_values('Frequency', inplace=True, ascending=False)
    female_premodifier_new.drop_duplicates(subset='female_premodifier',
                                           keep=False, inplace=True)

    male_premodifier_new = male_premodifier_df.copy()
    male_premodifier_new['Frequency'] = male_premodifier_new['male_premodifier'].map(
        male_premodifier_new['male_premodifier'].value_counts())
    male_premodifier_new.sort_values('Frequency', inplace=True, ascending=False)
    male_premodifier_new.drop_duplicates(subset='male_premodifier',
                                         keep=False, inplace=True)

    female_premodifier_new.rename(columns={'female_premodifier': 'word'}, inplace=True)
    male_premodifier_new.rename(columns={'male_premodifier': 'word'}, inplace=True)

    return female_premodifier_new, male_premodifier_new


def postmodifier_analysis(view_df):
    # columns = ['female_adj', 'male_adj']
    female_postmodifier_df = view_df.drop('male_postmodifier', axis=1)
    male_postmodifier_df = view_df.drop('female_postmodifier', axis=1)
    female_postmodifier_new = female_postmodifier_df.copy()
    female_postmodifier_new['Frequency'] = female_postmodifier_new['female_postmodifier'].map(
        female_postmodifier_new['female_postmodifier'].value_counts())
    female_postmodifier_new.sort_values('Frequency', inplace=True, ascending=False)
    female_postmodifier_new.drop_duplicates(subset='female_postmodifier',
                                            keep=False, inplace=True)

    male_postmodifier_new = male_postmodifier_df.copy()
    male_postmodifier_new['Frequency'] = male_postmodifier_new['male_postmodifier'].map(
        male_postmodifier_new['male_postmodifier'].value_counts())
    male_postmodifier_new.sort_values('Frequency', inplace=True, ascending=False)
    male_postmodifier_new.drop_duplicates(subset='male_postmodifier',
                                          keep=False, inplace=True)

    female_postmodifier_new.rename(columns={'female_postmodifier': 'word'}, inplace=True)
    male_postmodifier_new.rename(columns={'male_postmodifier': 'word'}, inplace=True)

    return female_postmodifier_new, male_postmodifier_new


def aux_analysis(view_df):
    # columns = ['female_before_aux', male_before_aux', female_follow_aux', male_follow_aux' ]
    female_before_aux_df = view_df[['female_before_aux']]
    male_before_aux_df = view_df[['male_before_aux']]
    female_follow_aux_df = view_df[['female_follow_aux']]
    male_follow_aux_df = view_df[['male_follow_aux']]

    female_before_aux_new = female_before_aux_df.copy()
    female_before_aux_new['Frequency'] = female_before_aux_new['female_before_aux'].map(
        female_before_aux_new['female_before_aux'].value_counts())
    female_before_aux_new.sort_values('Frequency', inplace=True, ascending=False)
    female_before_aux_new.drop_duplicates(subset='female_before_aux',
                                          keep=False, inplace=True)

    male_before_aux_new = male_before_aux_df.copy()
    male_before_aux_new['Frequency'] = male_before_aux_new['male_before_aux'].map(
        male_before_aux_new['male_before_aux'].value_counts())
    male_before_aux_new.sort_values('Frequency', inplace=True, ascending=False)
    male_before_aux_new.drop_duplicates(subset='male_before_aux',
                                        keep=False, inplace=True)

    female_follow_aux_new = female_follow_aux_df.copy()
    female_follow_aux_new['Frequency'] = female_follow_aux_new['female_follow_aux'].map(
        female_follow_aux_new['female_follow_aux'].value_counts())
    female_follow_aux_new.sort_values('Frequency', inplace=True, ascending=False)
    female_follow_aux_new.drop_duplicates(subset='female_follow_aux',
                                          keep=False, inplace=True)

    male_follow_aux_new = male_follow_aux_df.copy()
    male_follow_aux_new['Frequency'] = male_follow_aux_new['male_follow_aux'].map(
        male_follow_aux_new['male_follow_aux'].value_counts())
    male_follow_aux_new.sort_values('Frequency', inplace=True, ascending=False)
    male_follow_aux_new.drop_duplicates(subset='male_follow_aux',
                                        keep=False, inplace=True)

    female_before_aux_new.rename(columns={'female_before_aux': 'word'}, inplace=True)
    male_before_aux_new.rename(columns={'male_before_aux': 'word'}, inplace=True)

    female_follow_aux_new.rename(columns={'female_follow_aux': 'word'}, inplace=True)
    male_follow_aux_new.rename(columns={'male_follow_aux': 'word'}, inplace=True)

    return female_before_aux_new, male_before_aux_new, female_follow_aux_new, male_follow_aux_new


def possess_analysis(view_df):
    # columns = ['female_possessive', male_possessive', female_possessor', male_possessor' ]
    female_possessive_df = view_df[['female_possessive']]
    male_possessive_df = view_df[['male_possessive']]
    female_possessor_df = view_df[['female_possessor']]
    male_possessor_df = view_df[['male_possessor']]

    female_possessive_new = female_possessive_df.copy()
    female_possessive_new['Frequency'] = female_possessive_new['female_possessive'].map(
        female_possessive_new['female_possessive'].value_counts())
    female_possessive_new.sort_values('Frequency', inplace=True, ascending=False)
    female_possessive_new.drop_duplicates(subset='female_possessive',
                                          keep=False, inplace=True)

    male_possessive_new = male_possessive_df.copy()
    male_possessive_new['Frequency'] = male_possessive_new['male_possessive'].map(
        male_possessive_new['male_possessive'].value_counts())
    male_possessive_new.sort_values('Frequency', inplace=True, ascending=False)
    male_possessive_new.drop_duplicates(subset='male_possessive',
                                        keep=False, inplace=True)

    female_possessor_new = female_possessor_df.copy()
    female_possessor_new['Frequency'] = female_possessor_new['female_possessor'].map(
        female_possessor_new['female_possessor'].value_counts())
    female_possessor_new.sort_values('Frequency', inplace=True, ascending=False)
    female_possessor_new.drop_duplicates(subset='female_possessor',
                                         keep=False, inplace=True)

    male_possessor_new = male_possessor_df.copy()
    male_possessor_new['Frequency'] = male_possessor_new['male_possessor'].map(
        male_possessor_new['male_possessor'].value_counts())
    male_possessor_new.sort_values('Frequency', inplace=True, ascending=False)
    male_possessor_new.drop_duplicates(subset='male_possessor',
                                       keep=False, inplace=True)

    female_possessive_new.rename(columns={'female_possessive': 'word'}, inplace=True)
    male_possessive_new.rename(columns={'male_possessive': 'word'}, inplace=True)

    female_possessor_new.rename(columns={'female_possessor': 'word'}, inplace=True)
    male_possessor_new.rename(columns={'male_possessor': 'word'}, inplace=True)

    return female_possessive_new, male_possessive_new, female_possessor_new, male_possessor_new

def profession_analysis(view_df):
    # columns = ['female_profession', 'female_bias', male_profession', 'male_bias']
    female_profession_df = view_df[['female_profession', 'female_bias']].copy()
    female_profession_df.sort_values(by=['female_bias'])
    female_profession_df.rename(columns={'female_profession': 'token'}, inplace=True)
    female_profession_df.rename(columns={'female_bias': 'bias'}, inplace=True)

    male_profession_df = view_df[['male_profession', 'male_bias']].copy()
    male_profession_df.sort_values(by=['male_bias'])
    male_profession_df.rename(columns={'male_profession': 'token'}, inplace=True)
    male_profession_df.rename(columns={'male_bias': 'bias'}, inplace=True)


    return female_profession_df, male_profession_df



def txt_profession_list():
    """
    :param txt_dir: the path of the txt files to be extracted
    :return: a clean list containing the raw sentences
    """
    profession_list = []
    profession_txt = os.path.join(os.path.dirname(__file__), '..', 'data', 'professions.txt')
    with open(profession_txt, 'r', encoding='utf-8') as file_in:
        for line in file_in:
            profession_list.append(line.strip())
        profession_list = [x.lower() for x in profession_list]
    return profession_list


def determine_gender_professions(view_df):
    profession_list = txt_profession_list()
    view_dict = view_df.to_dict('records')
    female_professions = []
    female_professions_bias = []
    male_professions = []
    male_professions_bias = []
    for item in view_dict:
        if item['pos'] == 'NOUN':
            if item['token'] in profession_list:
                if item['bias'] < 0:
                    female_professions.append(item['token'])
                    female_professions_bias.append(item['bias'])
                elif item['bias'] > 0:
                    male_professions.append(item['token'])
                    male_professions_bias.append(item['bias'])

    #convert list to dataframe
    list_of_series = [pd.Series(female_professions), pd.Series(female_professions_bias), pd.Series(male_professions), pd.Series(male_professions_bias)]
    profession_df = pd.concat(list_of_series, axis=1)
    profession_df.columns = ['female_profession', 'female_bias', 'male_profession', 'male_bias']

    return profession_df

def gender_count_analysis(view_df):
    female_count = int(view_df['female_count'].iloc[0])
    male_count = int(view_df['male_count'].iloc[0])

    return female_count, male_count


def gender_dataframe_from_tuple(view_df):
    path_parent = os.path.dirname(os.getcwd())
    df_path = os.path.join(path_parent, 'visualising_data_bias', 'bias_visualisation_app', 'static')
    female_dataframe, male_dataframe = dataframe_by_gender(view_df)

    save_obj(female_dataframe, name='fm_dic')
    save_obj(male_dataframe, name='m_dic')
    male_dataframe = load_obj(df_path, name='m_dic')
    male_dataframe = male_dataframe.sort_values(by='bias', ascending=False)
    male_dataframe = male_dataframe.drop_duplicates(subset=['token'])

    female_dataframe = load_obj(df_path, name='fm_dic')
    female_dataframe = female_dataframe.sort_values(by='bias', ascending=True)
    female_dataframe = female_dataframe.drop_duplicates(subset=['token'])

    return female_dataframe, male_dataframe


def parse_pos_dataframe(view_df):
    female_dataframe, male_dataframe = gender_dataframe_from_tuple(view_df)

    female_noun_df = female_dataframe[female_dataframe['pos'].isin(noun_list)]
    female_adj_df = female_dataframe[female_dataframe['pos'].isin(adj_list)]
    female_verb_df = female_dataframe[female_dataframe['pos'].isin(verb_list)]

    male_noun_df = male_dataframe[male_dataframe['pos'].isin(noun_list)]
    male_adj_df = male_dataframe[male_dataframe['pos'].isin(adj_list)]
    male_verb_df = male_dataframe.loc[male_dataframe['pos'].isin(verb_list)]

    return female_noun_df, female_adj_df, female_verb_df, male_noun_df, male_adj_df, male_verb_df


female_synonyms = ['woman', 'women', 'female', 'females']
male_synonyms = ['man', 'men', 'male', 'males']

def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return True
    else:
        return False

def analyse_question(input_question, view_df, input_SVO_dataframe, input_premodifier_dataframe,
                         input_postmodifier_dataframe, input_aux_dataframe, input_possess_dataframe, input_profession_dataframe):
    female_tot_df, male_tot_df = gender_dataframe_from_tuple(view_df)
    female_noun_df, female_adj_df, female_verb_df = parse_pos_dataframe(view_df)[:3]
    male_noun_df, male_adj_df, male_verb_df = parse_pos_dataframe(view_df)[-3:]
    female_sub_df, female_obj_df, female_intran_df, male_sub_df, male_obj_df, male_intran_df = SVO_analysis(
        input_SVO_dataframe)
    female_premodifier_df, male_premodifier_df = premodifier_analysis(input_premodifier_dataframe)
    female_postmodifier_df, male_postmodifier_df = postmodifier_analysis(input_postmodifier_dataframe)
    female_before_aux_df, male_before_aux_df, female_follow_aux_df, male_follow_aux_df = aux_analysis(
        input_aux_dataframe)
    female_possessive_df, male_possessive_df, female_possessor_df, male_possessor_df = possess_analysis(
        input_possess_dataframe)
    female_profession_df, male_profession_df = profession_analysis(input_profession_dataframe)

    user_question = input_question.lower()
    user_question = user_question.translate(str.maketrans('', '', string.punctuation))
    user_question_list = user_question.split()
    # first step is to determine gender
    if common_member(user_question_list, female_synonyms) is True and common_member(user_question_list, male_synonyms) is False:
        # second step is to parse the dataform
        select_gender = 'Female'
        if 'nouns' in user_question_list:
            select_wordtype = 'Nouns'
            return select_gender, select_wordtype, female_noun_df
        if 'adjectives' in user_question_list:
            select_wordtype = 'Adjectives'
            return  select_gender, select_wordtype, female_adj_df
        if 'verbs' in user_question_list and 'intransitive' not in user_question_list:
            select_wordtype = 'Verbs'
            return  select_gender, select_wordtype, female_verb_df
        if 'auxiliary' in user_question_list and 'before' in user_question_list:
            # e.g. Men are the most irrational species.
            select_wordtype = 'Subject Auxiliary Words'
            return  select_gender, select_wordtype, female_before_aux_df
        if 'auxiliary' in user_question_list and 'after' in user_question_list:
            select_wordtype = 'Object Auxiliary Words'
            return  select_gender, select_wordtype, female_follow_aux_df
        if 'possessors' in user_question_list:
            select_wordtype = 'Possessors'
            return  select_gender, select_wordtype, female_possessor_df
        if 'possessives' in user_question_list:
            select_wordtype = 'Possessives'
            return  select_gender, select_wordtype, female_possessive_df
        if 'premodifiers' in user_question_list:
            select_wordtype = 'Premodifiers'
            return  select_gender, select_wordtype, female_premodifier_df
        if 'postmodifiers' in user_question_list:
            select_wordtype = 'Postmodifiers'
            return  select_gender, select_wordtype, female_postmodifier_df
        if 'professions' in user_question_list or 'jobs' in user_question_list:
            select_wordtype = 'Professions'
            return  select_gender, select_wordtype, female_profession_df
        if 'intransitive' in user_question_list:
            select_wordtype = 'Intransitive Verbs'
            return  select_gender, select_wordtype, female_intran_df
        if 'actions' in user_question_list and 'by' in user_question_list:
            select_wordtype = 'Actions as Subjects'
            return  select_gender, select_wordtype, female_sub_df
        if 'actions' in user_question_list and 'do' in user_question_list:
            select_wordtype = 'Actions as Subjects'
            return select_gender, select_wordtype, female_sub_df
        if 'actions' in user_question_list and 'against' in user_question_list:
            select_wordtype = 'Actions as Objects'
            return  select_gender, select_wordtype, female_obj_df

    elif common_member(user_question_list, female_synonyms) is False and common_member(user_question_list, male_synonyms) is True:
        select_gender = 'Male'
        if 'nouns' in user_question_list:
            select_wordtype = 'Nouns'
            return select_gender, select_wordtype, male_noun_df
        if 'adjectives' in user_question_list:
            select_wordtype = 'Adjectives'
            return select_gender, select_wordtype, male_adj_df
        if 'verbs' in user_question_list and 'intransitive' not in user_question_list:
            select_wordtype = 'Verbs'
            return select_gender, select_wordtype, male_verb_df
        if 'auxiliary' in user_question_list and 'before' in user_question_list:
            # e.g. Men are the most irrational species.
            select_wordtype = 'Subject Auxiliary Words'
            return select_gender, select_wordtype, male_before_aux_df
        if 'auxiliary' in user_question_list and 'after' in user_question_list:
            select_wordtype = 'Object Auxiliary Words'
            return select_gender, select_wordtype, male_follow_aux_df
        if 'possessors' in user_question_list:
            select_wordtype = 'Possessors'
            return select_gender, select_wordtype, male_possessor_df
        if 'possessives' in user_question_list:
            select_wordtype = 'Possessives'
            return select_gender, select_wordtype, male_possessive_df
        if 'premodifiers' in user_question_list:
            select_wordtype = 'Premodifiers'
            return select_gender, select_wordtype, male_premodifier_df
        if 'postmodifiers' in user_question_list:
            select_wordtype = 'Postmodifiers'
            return select_gender, select_wordtype, male_postmodifier_df
        if 'professions' in user_question_list or 'jobs' in user_question_list:
            select_wordtype = 'Professions'
            return select_gender, select_wordtype, male_profession_df
        if 'intransitive' in user_question_list:
            select_wordtype = 'Intransitive Verbs'
            return select_gender, select_wordtype, male_intran_df
        if 'actions' in user_question_list and 'by' in user_question_list:
            select_wordtype = 'Actions as Subjects'
            return select_gender, select_wordtype, male_sub_df
        if 'actions' in user_question_list and 'do' in user_question_list:
            select_wordtype = 'Actions as Subjects'
            return select_gender, select_wordtype, male_sub_df
        if 'actions' in user_question_list and 'against' in user_question_list:
            select_wordtype = 'Actions as Objects'
            return select_gender, select_wordtype, male_obj_df


def user_input_list():
    """
    :param txt_dir: the path of the txt files to be extracted
    :return: a clean list containing the raw sentences
    """
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    txt_dir = os.path.join(fileDir, 'bias_visualisation_app', 'data', 'user_uploads')
    original_word_list = []
    word_list = []

    with open(os.path.join(txt_dir, 'user_input_text.txt'), 'r', encoding='utf-8') as file_in:
        for line in file_in:
            sent_text = nltk.sent_tokenize(line)
            for sent in sent_text:
                new_sent = sent.lower()
                new_sent = new_sent.translate(str.maketrans('', '', string.punctuation))
                tokens = nltk.word_tokenize(new_sent)
                original_word_list.append(sent)
                word_list.append(tokens)

    return original_word_list, word_list


def calculate_sentence_bias_score(original_word_list ,word_list):
    path_parent = os.path.dirname(os.getcwd())
    df_path = os.path.join(path_parent, 'visualising_data_bias', 'bias_visualisation_app', 'static', 'user_uploads')
    view_df = load_obj_user_uploads(df_path, name='total_dataframe_user_uploads')
    sentence_score_list = []
    count = 0
    for sent in word_list:
        bias_list = []
        for word in sent:
            try:
                bias_value = view_df.loc[view_df['token'] == word, 'bias'].iloc[0]
                bias_list.append(bias_value)
            except:
                continue

        if len(bias_list) == 0:
            mean_bias_score = 0
        else:
            mean_bias_score = sum(bias_list)/len(bias_list)

        sentence_score = {'sentence': original_word_list[count], 'score': mean_bias_score}
        sentence_score_list.append(sentence_score)
        count += 1

    # convert to dataframe and renormalise it.
    sentence_score_df = pd.DataFrame(sentence_score_list)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    sentence_score_df['score'] = scaler.fit_transform(sentence_score_df[['score']])

    return sentence_score_df

replacers = {"dont": "don't", "doesnt": "doesn't", "wont": "won't", "wouldnt": "wouldn't", "cant": "can't", "couldnt": "couldn't", "neednt": "needn't", "shouldnt": "shouldn't"}

special_phrases = ["dont", "doesnt", "wont", "wouldnt", "cant", "couldnt", "neednt", "shouldnt"]

def debiased_file(threshold_value):
    original_word_list, word_list = user_input_list()
    sentence_score_df = calculate_sentence_bias_score(original_word_list, word_list)
    debiased_df = sentence_score_df[sentence_score_df['score'].between(-abs(threshold_value), threshold_value)]
    debiased_sentence_list = []
    for index, row in debiased_df.iterrows():
        new_sentence = row['sentence']

        try:
            new_sentence.replace(replacers)
        except:
            new_sentence = new_sentence

        debiased_sentence_list.append(new_sentence)

    path_parent = os.path.dirname(os.getcwd())
    save_path = os.path.join(path_parent,'visualising_data_bias', 'bias_visualisation_app', 'static')

    with open(os.path.join(save_path, 'debiased_file' + '.txt'), 'w+', encoding='utf-8') as f:
        f.write('\n'.join(debiased_sentence_list))

import dtale
from pivottablejs import pivot_ui
def style_dataframe(df, df_name):
    # # should list all possible situations here!!!
    # if 'bias' in df.columns:
    #     df_styler = df.style.set_precision(2).background_gradient(axis=0, gmap=df['bias']).hide_index()
    # elif 'Frequency' in df.columns:
    #     df_styler = df.style.set_precision(2).background_gradient(axis=0, gmap=df['Frequency']).hide_index()
    #
    # # can add more styling options here:
    #
    df_name = df_name + '.html'
    df_path = os.path.join(os.path.dirname(__file__), "..", "static", df_name)
    # with open(df_path, 'w') as f:
    #     # try:
    #     #     f.write(df_styler.render())
    #     # except ValueError:
    #     #     pass
    #     f.write(df_styler.render())
    #
    # return url_for('static', filename=df_name)
    pivot_ui(df, outfile_path=df_path)
    return url_for('static', filename=df_name)
