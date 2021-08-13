import nltk.corpus as nc
import nltk

male_nouns = nc.names.words('male.txt')
male_nouns.extend(['he', 'him', 'himself', 'gentleman', 'gentlemen', 'man', 'men', 'male'])
male_nouns = [x.lower() for x in male_nouns]
female_nouns = nc.names.words('female.txt')
female_nouns.extend(['she', 'her', 'herself', 'lady', 'ladys', 'woman', 'women', 'female'])
female_nouns = [x.lower() for x in female_nouns]

sentence = "We need to protect women's rights. Men's health is as important. I can look after the Simpsons' cat. Japan's women live longest. Canada's John clinged a gold prize."



def gender_count(sentence):
    sentence = sentence.lower()
    sent_text = nltk.sent_tokenize(sentence)
    tot_count_female = 0
    tot_count_male = 0
    for sent in sent_text:
        tokens = nltk.word_tokenize(sent)
        for tok in tokens:
            if tok in female_nouns:
                tot_count_female += 1
            if tok in male_nouns:
                tot_count_male += 1

    return tot_count_female, tot_count_male


print(gender_count(sentence))