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

post_modifiers = ["compounds", "pobj"]
post_modifiers_noun_list = ["women", "Women", "female", "Female", "men", "Men", "male", "Male"]
female_postmodifier_noun_list = ["women", "female"]
male_postmodifier_noun_list = ["men", "male"]

sentence = 'Women writers support male fighters. Male cleaners are more careful. Lucy likes female dramas. Women like sunglasses.'



def findfemalePostmodifiers(sent):
    tokens = nltk.word_tokenize(sent)
    tags = nltk.pos_tag(tokens)
    female_postmodifier_list = []
    for female_noun in female_postmodifier_noun_list:
        try:
            # You are interested in splitting the sentence here
            sentence_split = tokens.index(female_noun)
            # Find the words where tag meets your criteria (must be a noun / proper noun)
            nouns_before_split = [word for (word, tag) in tags[sentence_split + 1: sentence_split + 2] if tag.startswith('NN')]
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
            nouns_before_split = [word for (word, tag) in tags[sentence_split + 1:] if tag.startswith('NN')]
            post_modifier = nouns_before_split[0]
            male_postmodifier_list.append(post_modifier)
        except:
            continue

    return male_postmodifier_list

def determine_gender_postmodifier_test(sentence):
    sentence = sentence.lower()
    sent_text = nltk.sent_tokenize(sentence)
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
    return tot_female_postmodifier_list, tot_male_postmodifier_list


print(determine_gender_postmodifier_test(sentence))

# women_noundist = nltk.FreqDist(w2 for ((w1, t1), (w2, t2)) in
#             nltk.bigrams(brown.tagged_words(tagset="universal"))
#             if w1.lower() == "women" and t2 == "NOUN")
# print(women_noundist)
#
# men_noundist = nltk.FreqDist(w2 for ((w1, t1), (w2, t2)) in
#             nltk.bigrams(brown.tagged_words(tagset="universal"))
#             if w1.lower() == "men" and t2 == "NOUN")
#
# female_noundist = nltk.FreqDist(w2 for ((w1, t1), (w2, t2)) in
#             nltk.bigrams(brown.tagged_words(tagset="universal"))
#             if w1.lower() == "female" and t2 == "NOUN")

# male_noundist = nltk.FreqDist(w2 for ((w1, t1), (w2, t2)) in
#             nltk.bigrams(brown.tagged_words(tagset="universal"))
#             if w1.lower() == "male" and t2 == "NOUN")
