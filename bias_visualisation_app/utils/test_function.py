from nltk.stem.wordnet import WordNetLemmatizer
import random
import nltk.corpus as nc
import nltk.classify as cf
import nltk
import spacy
import numpy as np

SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl", "compounds", "pobj"]
OBJECTS = ["dobj", "dative", "attr", "oprd"]
ADJECTIVES = ["acomp", "advcl", "advmod", "amod", "appos", "nn", "nmod", "ccomp", "complm",
              "hmod", "infmod", "xcomp", "rcmod", "poss", "possessive"]
COMPOUNDS = ["compound"]
PREPOSITIONS = ["prep"]


def getSubsFromConjunctions(subs):
    moreSubs = []
    for sub in subs:
        # rights is a generator
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        # look for multiple subjects
        if "and" in rightDeps:
            moreSubs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"])
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
        if "and" in rightDeps:
            moreObjs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == "NOUN"])
            if len(moreObjs) > 0:
                moreObjs.extend(getObjsFromConjunctions(moreObjs))
    return moreObjs


def getVerbsFromConjunctions(verbs):
    moreVerbs = []
    for verb in verbs:
        rightDeps = {tok.lower_ for tok in verb.rights}
        # look for multiple verbs
        if "and" in rightDeps:
            moreVerbs.extend([tok for tok in verb.rights if tok.pos_ == "VERB"])
            if len(moreVerbs) > 0:
                moreVerbs.extend(getVerbsFromConjunctions(moreVerbs))
    return moreVerbs


def findSubs(tok):
    head = tok.head
    while head.pos_ != "VERB" and head.pos_ != "NOUN" and head.head != head:
        head = head.head
    if head.pos_ == "VERB":
        subs = [tok for tok in head.lefts if tok.dep_ == "SUB"]
        if len(subs) > 0:
            verbNegated = isNegated(head)
            subs.extend(getSubsFromConjunctions(subs))
            return subs, verbNegated
        elif head.head != head:
            return findSubs(head)
    elif head.pos_ == "NOUN":
        return [head], isNegated(tok)
    return [], False


def isNegated(tok):
    negations = {"no", "not", "n't", "never", "none"}
    for dep in list(tok.lefts) + list(tok.rights):
        if dep.lower_ in negations:
            return True
    return False


def findSVs(tokens):
    svs = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB"]
    for v in verbs:
        subs, verbNegated = getAllSubs(v)
        if len(subs) > 0:
            for sub in subs:
                svs.append((sub.orth_, "!" + v.orth_ if verbNegated else v.orth_))
    return svs


def getObjsFromPrepositions(deps):
    objs = []
    for dep in deps:
        if dep.pos_ == "ADP" and dep.dep_ == "prep":
            objs.extend(
                [tok for tok in dep.rights if tok.dep_ in OBJECTS or (tok.pos_ == "PRON" and tok.lower_ == "me")])
    return objs


def getAdjectives(toks):
    toks_with_adjectives = []
    for tok in toks:
        adjs = [left for left in tok.lefts if left.dep_ in ADJECTIVES]
        adjs.append(tok)
        adjs.extend([right for right in tok.rights if tok.dep_ in ADJECTIVES])
        tok_with_adj = " ".join([adj.lower_ for adj in adjs])
        toks_with_adjectives.extend(adjs)

    return toks_with_adjectives


def getObjsFromAttrs(deps):
    for dep in deps:
        if dep.pos_ == "NOUN" and dep.dep_ == "attr":
            verbs = [tok for tok in dep.rights if tok.pos_ == "VERB"]
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
        if dep.pos_ == "VERB" and dep.dep_ == "xcomp":
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(getObjsFromPrepositions(rights))
            if len(objs) > 0:
                return v, objs
    return None, None

non_sub_pos = ["DET", "AUX"]

def getAllSubs(v):
    verbNegated = isNegated(v)
    #subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS elif  type(tok.dep_) == int or float  and tok.pos_ != "DET"]
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
    verbs = [tok for tok in tokens if tok.pos_ == "AUX"]
    for v in verbs:
        subs, verbNegated = getAllSubs(v)
        # hopefully there are subs, if not, don't examine this verb any longer
        if len(subs) > 0:
            v, objs = getAllObjs(v)
            for sub in subs:
                for obj in objs:
                    objNegated = isNegated(obj)
                    svos.append((sub.lower_, "!" + v.lower_ if verbNegated or objNegated else v.lower_, obj.lower_))
    return svos

# def findSVAOs(tokens):
#     svos = []
#     # exclude the auxiliary verbs such as 'She is smart.' Ignore adjective analysis since adjectives have already been identified in the previous algorithms.
#     verbs = [tok for tok in tokens if tok.pos_ == "VERB" and tok.dep_ != "aux"]
#     print(verbs)
#     for v in verbs:
#         print('start getting subs')
#         subs, verbNegated = getAllSubs(v)
#         # hopefully there are subs, if not, don't examine this verb any longer
#         if len(subs) > 0:
#             v, objs = getAllObjs(v)
#             print('verb, objects')
#             print(v, objs)
#             if len(objs) > 0:
#                 print(objs)
#                 print('obj not empty')
#                 for sub in subs:
#                     for obj in objs:
#                         objNegated = isNegated(obj)
#                         obj_desc_tokens = generate_left_right_adjectives(obj)
#                         sub_compound = generate_sub_compound(sub)
#                         svos.append((" ".join(tok.lower_ for tok in sub_compound),
#                                      "!" + v.lower_ if verbNegated or objNegated else v.lower_,
#                                      " ".join(tok.lower_ for tok in obj_desc_tokens)))
#
#             if len(objs) == 0:
#                 print(objs)
#                 print('obj empty')
#                 svos = [str(subs[0]), str(v)]
#                 svos.append(" ")
#             print('SVO list')
#             print(svos)
#     return svos

def findSVAOs(tokens):
    svos = []
    # exclude the auxiliary verbs such as 'She is smart.' Ignore adjective analysis since adjectives have already been identified in the previous algorithms.
    verbs = [tok for tok in tokens if tok.pos_ == "VERB" and tok.dep_ != "aux"]
    print('first identified verbs', verbs)
    #not_verbs = [tok for tok in tokens if tok.pos_ == "VERB" and tok.tag_ == "VBN"][0]#
    not_verbs = []
    for tok in tokens:
        if tok.pos_ == "VERB" and tok.tag_ == "VBN":
            not_verbs.append(tok)

    #if (not_verbs not in verbs or len(not_verbs) == 0):
    if len(not_verbs) == 0:
        print('safe to proceed with first identified verb!')
        for v in verbs:
            subs, verbNegated = getAllSubs(v)
            # hopefully there are subs, if not, don't examine this verb any longer
            if len(subs) > 0:
                v, objs = getAllObjs(v)
                if len(objs) > 0:
                    for sub in subs:
                        for obj in objs:
                            objNegated = isNegated(obj)
                            obj_desc_tokens = generate_left_right_adjectives(obj)
                            sub_compound = generate_sub_compound(sub)
                            svos.append((" ".join(tok.lower_ for tok in sub_compound),
                                         "!" + v.lower_ if verbNegated or objNegated else v.lower_,
                                         " ".join(tok.lower_ for tok in obj_desc_tokens)))

                if len(objs) == 0:
                    svos = [str(subs[0]), str(v)]
                    svos.append('nothing')

    elif not_verbs[0] not in verbs:
        print('safe to proceed with first identified verb!')
        for v in verbs:
            subs, verbNegated = getAllSubs(v)
            # hopefully there are subs, if not, don't examine this verb any longer
            if len(subs) > 0:
                v, objs = getAllObjs(v)
                if len(objs) > 0:
                    for sub in subs:
                        for obj in objs:
                            objNegated = isNegated(obj)
                            obj_desc_tokens = generate_left_right_adjectives(obj)
                            sub_compound = generate_sub_compound(sub)
                            svos.append((" ".join(tok.lower_ for tok in sub_compound),
                                         "!" + v.lower_ if verbNegated or objNegated else v.lower_,
                                         " ".join(tok.lower_ for tok in obj_desc_tokens)))

                if len(objs) == 0:
                    svos = [str(subs[0]), str(v)]
                    svos.append('nothing')

    else:
        new_verbs = [tok for tok in tokens if tok.pos_ == "VERB" and tok.tag_ == "VBN"]
        tokens_new = [t for t in tokens]
        tokens_new_str = [str(t) for t in tokens]
        for new_verb in new_verbs:
                new_objs, new_verbNegated = getAllSubs(new_verb)
                get_index = tokens_new_str.index(str(new_verb))
                after_tok_list = tokens_new[get_index + 1:]
                after_tok_list_str = tokens_new_str[get_index + 1:]
                if 'by' in after_tok_list_str:
                    for after_tok in after_tok_list:
                        new_subs = []
                        if after_tok.dep_ in SUBJECTS and after_tok.pos_ not in non_sub_pos:
                            new_subs.append(after_tok)
                        elif type(after_tok.dep_) == int or float and after_tok.pos_ not in non_sub_pos:
                            new_subs.append(after_tok)
                        new_sub = new_subs[0]

                    svos = [str(new_sub), str(new_verb), str(new_objs[0])]

                else:
                    svos = ['neutral', str(new_verb), str(new_objs[0])]

    print('Final SVO list', svos)
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



sentence = 'Hilary is supported'



def determine_gender_SVO(sentence):
    parser = spacy.load('en_core_web_md', disable=['ner', 'textcat'])

    sent_text = nltk.sent_tokenize(sentence)
    for sentence in sent_text:
        parse = parser(sentence)
        try:
            SVO_list = findSVAOs(parse)
            print(SVO_list)
        except:
            continue


determine_gender_SVO(sentence)























# male_names = nc.names.words('male.txt')
# male_names.extend(['he', 'him'])
# female_names = nc.names.words('female.txt')
# female_names.extend(['she', 'her'])
# models, acs = [], []
#
# for n_letters in range(1, 6):
#     data = []
#     for male_name in male_names:
#         feature = {'feature': male_name[-n_letters:].lower()}
#         data.append((feature, 'male'))
#     for female_name in female_names:
#         feature = {'feature': female_name[-n_letters:].lower()}
#         data.append((feature, 'female'))
#     random.seed(7)
#     random.shuffle(data)
#     train_data = data[:int(len(data) / 2)]
#     test_data = data[int(len(data) / 2):]
#     model = cf.NaiveBayesClassifier.train(train_data)
#     ac = cf.accuracy(model, test_data)
#     models.append(model)
#     acs.append(ac)
#
# best_index = np.array(acs).argmax()
# best_letters = best_index + 1
#
# gender_model = models[best_index]
# best_ac = acs[best_index]
#
# neutral_sub_list = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours', 'it', 'its', 'they', 'them', 'their', 'theirs']
#
# spec_chars = ['!',''','#','%','&',''','(',')',
#               '*','+',',','-','.','/',':',';','<',
#               '=','>','?','@','[','\\',']','^','_',
#               '`','{','|','}','~','–']
#
# def reset_gender(subject, subject_gender):
#     if subject == 'he':
#         subject_gender_new = 'male'
#     elif subject == 'she':
#         subject_gender_new = 'female'
#     elif subject in neutral_sub_list:
#         subject_gender_new = 'neutral'
#     else:
#         subject_gender_new = subject_gender
#     return subject_gender_new
#
# def clean_SVO_dataframe(SVO_df):
#     # cleaning up the SVO dataframe
#     SVO_df['subject_gender'] = SVO_df.apply(lambda x: reset_gender(x.subject, x.subject_gender), axis=1)
#     SVO_df['object_gender'] = SVO_df.apply(lambda x: reset_gender(x.object, x.object_gender), axis=1)
#
#     for char in spec_chars:
#         SVO_df['subject'] = SVO_df['subject'].str.replace(char, ' ')
#         SVO_df['object'] = SVO_df['object'].str.replace(char, ' ')
#         SVO_df['verb'] = SVO_df['verb'].str.replace(char, ' ')
#
#     # get base form of verb
#     verb_list = SVO_df['verb'].to_list()
#     verb_base_list = []
#     for verb in verb_list:
#         base_word = WordNetLemmatizer().lemmatize(verb, 'v')
#         base_word.strip()
#         verb_base_list.append(base_word)
#
#     SVO_df['verb'] = verb_base_list
#
#     print(SVO_df)
#     return SVO_df
#
#
#
#
# def determine_gender_SVO(input_data):
#     parser = spacy.load('en_core_web_md', disable=['ner', 'textcat'])
#
#     sent_text = nltk.sent_tokenize(input_data)
#     sub_list = []
#     sub_gender_list = []
#     verb_list = []
#     obj_list = []
#     obj_gender_list = []
#     # now loop over each sentence and tokenize it separately
#     for sentence in sent_text:
#         parse = parser(sentence)
#         try:
#             SVO_list = findSVAOs(parse)
#             for i in SVO_list:
#                 sub, verb, obj = i[0], i[1], i[2]
#                 sub_feature = {'feature': sub[-best_letters:]}
#                 sub_gender = gender_model.classify(sub_feature)
#                 obj_feature = {'feature': obj[-best_letters:]}
#                 obj_gender = gender_model.classify(obj_feature)
#
#                 sub_list.append(sub)
#                 sub_gender_list.append(sub_gender)
#                 verb_list.append(verb)
#                 obj_list.append(obj)
#                 obj_gender_list.append(obj_gender)
#
#         except:
#             continue
#
#     SVO_df = pd.DataFrame(list(zip(sub_list, sub_gender_list, verb_list, obj_list, obj_gender_list)),
#                           columns=['subject', 'subject_gender', 'verb', 'object', 'object_gender'])
#
#     #cleaning up the SVO dataframe
#     SVO_df = clean_SVO_dataframe(SVO_df)
#
#
#     return SVO_df