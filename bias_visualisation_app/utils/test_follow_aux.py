import nltk.corpus as nc
import nltk
from nltk.corpus import brown
import spacy

SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl", "compounds", "pobj"]
OBJECTS = ["dobj", "dative", "attr", "oprd"]
ADJECTIVES = ["acomp", "advcl", "advmod", "amod", "appos", "nn", "nmod", "ccomp", "complm",
              "hmod", "infmod", "xcomp", "rcmod", "poss", "possessive"]
COMPOUNDS = ["compound"]
PREPOSITIONS = ["prep"]



aux_word_list = ["are", "is", "were", "was", "be"]
det_word_list = ["a", "an"]

male_nouns = nc.names.words('male.txt')
male_nouns.extend(['he', 'him', 'himself', 'gentleman', 'gentlemen', 'man', 'men', 'male'])
male_nouns = [x.lower() for x in male_nouns]
female_nouns = nc.names.words('female.txt')
female_nouns.extend(['she', 'her', 'herself', 'lady', 'ladys', 'woman', 'women', 'female'])
female_nouns = [x.lower() for x in female_nouns]


sentence = 'Most writers are female. The majority are women. The suspect is a woman. Her father was a man who lived an extraordinary life. Women are victims. Men are minority. The woman is a teacher. Sarah is an engineer. The culprit is Linda.'



def findfemalefollow_auxs(sent):
    tokens = nltk.word_tokenize(sent)
    tags = nltk.pos_tag(tokens)
    female_follow_aux_list = []
    for female_noun in female_nouns:
        try:
            # You are interested in splitting the sentence here
            sentence_split = tokens.index(female_noun)
            # Find the words where tag meets your criteria (must be a noun / proper noun)
            if tokens[sentence_split - 1] in aux_word_list:
                follow_aux = tokens[sentence_split - 2]
                if tags[sentence_split - 2][1].startswith('NN'):
                    female_follow_aux_list.append(follow_aux)
            elif tokens[sentence_split - 1] in det_word_list:
                follow_aux = tokens[sentence_split - 3]
                if tags[sentence_split - 3][1].startswith('NN'):
                    female_follow_aux_list.append(follow_aux)
            else:
                pass
        except:
            continue

    return female_follow_aux_list

def findmalefollow_auxs(sent):
    tokens = nltk.word_tokenize(sent)
    tags = nltk.pos_tag(tokens)
    male_follow_aux_list = []
    for male_noun in male_nouns:
        try:
            # You are interested in splitting the sentence here
            sentence_split = tokens.index(male_noun)
            # Find the words where tag meets your criteria (must be a noun / proper noun)
            if tokens[sentence_split - 1] in aux_word_list:
                follow_aux = tokens[sentence_split - 2]
                if tags[sentence_split - 2][1].startswith('NN'):
                    male_follow_aux_list.append(follow_aux)
            elif tokens[sentence_split - 1] in det_word_list:
                follow_aux = tokens[sentence_split - 3]
                if tags[sentence_split - 3][1].startswith('NN'):
                    male_follow_aux_list.append(follow_aux)
            else:
                pass
        except:
            continue

    return male_follow_aux_list

def findfemalebefore_auxs(sent):
    tokens = nltk.word_tokenize(sent)
    tags = nltk.pos_tag(tokens)
    female_before_aux_list = []
    for female_noun in female_nouns:
        try:
            # You are interested in splitting the sentence here
            sentence_split = tokens.index(female_noun)
            # Find the words where tag meets your criteria (must be a noun / proper noun)
            if tokens[sentence_split + 1] in aux_word_list:
                before_aux = tokens[sentence_split + 2]
                if tags[sentence_split + 2][1].startswith('NN'):
                    female_before_aux_list.append(before_aux)
            elif tokens[sentence_split + 1] in det_word_list:
                before_aux = tokens[sentence_split + 3]
                if tags[sentence_split + 3][1].startswith('NN'):
                    female_before_aux_list.append(before_aux)
            else:
                pass
        except:
            continue

    return female_before_aux_list

def findmalebefore_auxs(sent):
    tokens = nltk.word_tokenize(sent)
    tags = nltk.pos_tag(tokens)
    male_before_aux_list = []
    for male_noun in male_nouns:
        try:
            # You are interested in splitting the sentence here
            sentence_split = tokens.index(male_noun)
            # Find the words where tag meets your criteria (must be a noun / proper noun)
            if tokens[sentence_split + 1] in aux_word_list:
                before_aux = tokens[sentence_split + 2]
                if tags[sentence_split + 2][1].startswith('NN'):
                    male_before_aux_list.append(before_aux)
            elif tokens[sentence_split + 1] in det_word_list:
                before_aux = tokens[sentence_split + 3]
                if tags[sentence_split + 3][1].startswith('NN'):
                    male_before_aux_list.append(before_aux)
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
            female_follow_aux_list = findfemalefollow_auxs(sent)
            male_follow_aux_list = findmalefollow_auxs(sent)
            female_before_aux_list = findfemalebefore_auxs(sent)
            male_before_aux_list = findmalebefore_auxs(sent)
            tot_female_follow_aux_list.extend(female_follow_aux_list)
            tot_male_follow_aux_list.extend(male_follow_aux_list)
            tot_female_before_aux_list.extend(female_before_aux_list)
            tot_male_before_aux_list.extend(male_before_aux_list)
        except:
            continue
    return tot_female_follow_aux_list, tot_female_before_aux_list, tot_male_follow_aux_list, tot_male_before_aux_list


print(determine_gender_follow_aux_test(sentence))