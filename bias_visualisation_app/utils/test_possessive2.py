import nltk.corpus as nc
import nltk

SUBJECTS = ['nsubj', 'nsubjpass', 'csubj', 'csubjpass', 'agent', 'expl', 'compounds', 'pobj']
OBJECTS = ['dobj', 'dative', 'attr', 'oprd']
ADJECTIVES = ['acomp', 'advcl', 'advmod', 'amod', 'appos', 'nn', 'nmod', 'ccomp', 'complm',
              'hmod', 'infmod', 'xcomp', 'rcmod', 'poss', 'possessive']
COMPOUNDS = ['compound']
PREPOSITIONS = ['prep']

male_nouns = nc.names.words('male.txt')
male_nouns.extend(['he', 'him', 'himself', 'gentleman', 'gentlemen', 'man', 'men', 'male'])
male_nouns = [x.lower() for x in male_nouns]
female_nouns = nc.names.words('female.txt')
female_nouns.extend(['she', 'her', 'herself', 'lady', 'ladys', 'woman', 'women', 'female'])
female_nouns = [x.lower() for x in female_nouns]

sentence = "We need to protect women's rights. Men's health is as important. I can look after the Simpsons' cat."


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


def determine_gender_possessive_test(sentence):
    sentence = sentence.lower()
    sent_text = nltk.sent_tokenize(sentence)
    tot_female_possessive_list = []
    tot_male_possessive_list = []
    for sent in sent_text:
        try:
            female_possessive_list = findfemale_possessives(sent)
            male_possessive_list = findmale_possessives(sent)
            tot_female_possessive_list.extend(female_possessive_list)
            tot_male_possessive_list.extend(male_possessive_list)
        except:
            continue
    return tot_female_possessive_list, tot_male_possessive_list


print(determine_gender_possessive_test(sentence))
