import nltk.corpus as nc
import nltk


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
female_nouns = nc.names.words('male.txt')
female_nouns.extend(['she', 'her', 'herself', 'lady', 'ladys', 'woman', 'women', 'male'])
female_nouns = [x.lower() for x in male_nouns]


sentence = "Most writers are male. Most writers are a woman. Most writers are not male. The teacher is not a man. The majority are women. The suspect is a woman. Her father was a man who lived an extraordinary life. Women are not victims. Men are minority. The woman isn't a teacher. Sarah is an engineer. The culprit is Linda."



def findfemalefollow_auxs(sent):
    # e.g. Most writers are female. Most writers are a woman. Most writers are not female. The teacher is not a man.
    tokens = nltk.word_tokenize(sent)
    tags = nltk.pos_tag(tokens)
    female_follow_aux_list = []
    for female_noun in female_nouns:
        try:
            # You are interested in splitting the sentence here
            sentence_split = tokens.index(female_noun)
            # Find the words where tag meets your criteria (must be a noun / proper noun)
            if tokens[sentence_split - 1] in aux_word_list:
                #e.g.Most writers are female
                follow_aux = tokens[sentence_split - 2]
                if tags[sentence_split - 2][1].startswith('NN'):
                    female_follow_aux_list.append(follow_aux)
            elif tokens[sentence_split - 1] in det_word_list and tokens[sentence_split - 2] in aux_word_list:
                # Most writers are a woman
                follow_aux = tokens[sentence_split - 3]
                if tags[sentence_split - 3][1].startswith('NN'):
                    female_follow_aux_list.append(follow_aux)
            elif tokens[sentence_split - 1] in neg_word_list and tokens[sentence_split - 2] in aux_word_list:
                # Most writers are not female.
                follow_aux = tokens[sentence_split - 3]
                if tags[sentence_split - 3][1].startswith('NN'):
                    female_follow_aux_list.append('!' + str(follow_aux).lower())
            elif tokens[sentence_split - 1] in det_word_list and tokens[sentence_split - 2] in neg_word_list and tokens[sentence_split - 3] in aux_word_list:
                # Most writers are not a man.
                follow_aux = tokens[sentence_split - 4]
                if tags[sentence_split - 4][1].startswith('NN'):
                    female_follow_aux_list.append('!' + str(follow_aux).lower())
            else:
                pass
        except:
            continue

    return female_follow_aux_list

def findmalefollow_auxs(sent):
    # e.g. Most writers are male. Most writers are a woman. Most writers are not male. The teacher is not a man.
    tokens = nltk.word_tokenize(sent)
    tags = nltk.pos_tag(tokens)
    male_follow_aux_list = []
    for male_noun in male_nouns:
        try:
            # You are interested in splitting the sentence here
            sentence_split = tokens.index(male_noun)
            # Find the words where tag meets your criteria (must be a noun / proper noun)
            if tokens[sentence_split - 1] in aux_word_list:
                #e.g.Most writers are male
                follow_aux = tokens[sentence_split - 2]
                if tags[sentence_split - 2][1].startswith('NN'):
                    male_follow_aux_list.append(follow_aux)
            elif tokens[sentence_split - 1] in det_word_list and tokens[sentence_split - 2] in aux_word_list:
                # Most writers are a woman
                follow_aux = tokens[sentence_split - 3]
                if tags[sentence_split - 3][1].startswith('NN'):
                    male_follow_aux_list.append(follow_aux)
            elif tokens[sentence_split - 1] in neg_word_list and tokens[sentence_split - 2] in aux_word_list:
                # Most writers are not male.
                follow_aux = tokens[sentence_split - 3]
                if tags[sentence_split - 3][1].startswith('NN'):
                    male_follow_aux_list.append('!' + str(follow_aux).lower())
            elif tokens[sentence_split - 1] in det_word_list and tokens[sentence_split - 2] in neg_word_list and tokens[sentence_split - 3] in aux_word_list:
                # Most writers are not a man.
                follow_aux = tokens[sentence_split - 4]
                if tags[sentence_split - 4][1].startswith('NN'):
                    male_follow_aux_list.append('!' + str(follow_aux).lower())
            else:
                pass
        except:
            continue

    return male_follow_aux_list

def findfemalebefore_auxs(sent):
    # Women are minority. Mary is a teacher. Women are not minority. Mary is not a teacher
    tokens = nltk.word_tokenize(sent)
    tags = nltk.pos_tag(tokens)
    female_before_aux_list = []
    for female_noun in female_nouns:
        try:
            # You are interested in splitting the sentence here
            sentence_split = tokens.index(female_noun)
            # Find the words where tag meets your criteria (must be a noun / proper noun)
            if tokens[sentence_split + 1] in aux_word_list:
                # Women are minority.
                before_aux = tokens[sentence_split + 2]
                if tags[sentence_split + 2][1].startswith('NN'):
                    female_before_aux_list.append(before_aux)
            elif tokens[sentence_split + 1] in det_word_list and tokens[sentence_split + 2] in aux_word_list:
                # Women are a minority
                before_aux = tokens[sentence_split + 3]
                if tags[sentence_split + 3][1].startswith('NN'):
                    female_before_aux_list.append(before_aux)
            elif tokens[sentence_split + 1] in neg_word_list and tokens[sentence_split + 2] in aux_word_list:
                # Women are not minority.
                before_aux = tokens[sentence_split + 3]
                if tags[sentence_split + 3][1].startswith('NN'):
                    female_before_aux_list.append('!' + str(before_aux).lower())
            elif tokens[sentence_split + 1] in det_word_list and tokens[sentence_split + 2] in neg_word_list and tokens[sentence_split + 3] in aux_word_list:
                # Women are not a minority.
                before_aux = tokens[sentence_split + 4]
                if tags[sentence_split + 4][1].startswith('NN'):
                    female_before_aux_list.append('!' + str(before_aux).lower())
            else:
                pass
        except:
            continue

    return female_before_aux_list

def findmalebefore_auxs(sent):
    # Women are minority. Mary is a teacher. Women are not minority. Mary is not a teacher
    tokens = nltk.word_tokenize(sent)
    tags = nltk.pos_tag(tokens)
    male_before_aux_list = []
    for male_noun in male_nouns:
        try:
            # You are interested in splitting the sentence here
            sentence_split = tokens.index(male_noun)
            # Find the words where tag meets your criteria (must be a noun / proper noun)
            if tokens[sentence_split + 1] in aux_word_list:
                # Women are minority.
                before_aux = tokens[sentence_split + 2]
                if tags[sentence_split + 2][1].startswith('NN'):
                    male_before_aux_list.append(before_aux)
            elif tokens[sentence_split + 1] in det_word_list and tokens[sentence_split + 2] in aux_word_list:
                # Women are a minority
                before_aux = tokens[sentence_split + 3]
                if tags[sentence_split + 3][1].startswith('NN'):
                    male_before_aux_list.append(before_aux)
            elif tokens[sentence_split + 1] in neg_word_list and tokens[sentence_split + 2] in aux_word_list:
                # Women are not minority.
                before_aux = tokens[sentence_split + 3]
                if tags[sentence_split + 3][1].startswith('NN'):
                    male_before_aux_list.append('!' + str(before_aux).lower())
            elif tokens[sentence_split + 1] in det_word_list and tokens[sentence_split + 2] in neg_word_list and tokens[sentence_split + 3] in aux_word_list:
                # Women are not a minority.
                before_aux = tokens[sentence_split + 4]
                if tags[sentence_split + 4][1].startswith('NN'):
                    male_before_aux_list.append('!' + str(before_aux).lower())
            else:
                pass
        except:
            continue

    return male_before_aux_list

def determine_gender_follow_aux_test(sentence):
    sentence = sentence.lower()
    sent_text = nltk.sent_tokenize(sentence)
    tot_female_follow_aux_list = []
    tot_male_follow_aux_list = []
    tot_female_before_aux_list = []
    tot_male_before_aux_list = []
    for sent in sent_text:
        try:
            female_follow_aux_list = findmalefollow_auxs(sent)
            male_follow_aux_list = findmalefollow_auxs(sent)
            female_before_aux_list = findmalebefore_auxs(sent)
            male_before_aux_list = findmalebefore_auxs(sent)
            tot_female_follow_aux_list.extend(female_follow_aux_list)
            tot_male_follow_aux_list.extend(male_follow_aux_list)
            tot_female_before_aux_list.extend(female_before_aux_list)
            tot_male_before_aux_list.extend(male_before_aux_list)
        except:
            continue
    return tot_female_follow_aux_list, tot_female_before_aux_list, tot_male_follow_aux_list, tot_male_before_aux_list


print(determine_gender_follow_aux_test(sentence))