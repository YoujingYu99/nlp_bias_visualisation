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
import unittest
from parse_sentence import parse_sentence
from functions_files import save_obj_text, concat_csv_excel, save_obj, load_obj, load_obj_user_uploads
from PrecalculatedBiasCalculator import PrecalculatedBiasCalculator

import functions_analysis

sys.setrecursionlimit(10000)

calculator = PrecalculatedBiasCalculator()

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

neutral_words = [
    'is',
    'was',
    'who',
    'what',
    'where',
    'the',
    'it',
]

male_names = nc.names.words('male.txt')
male_names.extend(['he', 'him', 'himself', 'man', 'men', 'gentleman', 'gentlemen'])
female_names = nc.names.words('female.txt')
female_names.extend(['she', 'her', 'herself', 'woman', 'women', 'lady'])
neutral_sub_list = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours', 'it', 'its', 'they', 'them', 'their', 'theirs', 'neutral']

spec_chars = ['!', ''','#','%','&',''', '(', ')',
              '*', '+', ',', '-', '.', '/', ':', ';', '<',
              '=', '>', '?', '@', '[', '\\', ']', '^', '_',
              '`', '{', '|', '}', '~', '–']

input_data = "Women writers support male fighters. Male cleaners are not more careful. Lucy likes female dramas. Women do not like gloves. Lucy eats a tasty black bread. The elegant powerful woman wears shiny black glasses. The dark tall man drinks water. He adores vulnerable strong women. The kind beautiful girl picks a cup. Most writers are female. The majority are women. The suspect is a woman. Her father was a man who lived an extraordinary life. Women are victims. Men are not minority. The woman is a teacher. Sarah is an engineer. The culprit is not Linda.We need to protect women's rights. Men's health is as important. I can look after the Simpsons' cat. California's women live longest. Australia's John did not cling a gold prize. The world's women should unite together. Anna looks up a book. John asked Marilyn out. Steven did not take the coat off. Most writers are a woman. Most writers are not male. The teacher is not a man. The majority are women. The suspect is a woman. Her father was a man who lived an extraordinary life. Women are not victims. Men are minority. The woman isn't a teacher. Sarah is not a mathematician. Women pregnant should be carefully treated. Men generous are kind. John is a pilot. Steven is a fireman. Most nurses are male."

class TestAnalysis(unittest.TestCase):

    def test_determine_gender_SVO(self):
        expected = ['women writers', 'lucy', 'women', 'not', 'lucy', 'woman', 'man', 'he', 'girl', 'who', 'to', 'i', 'women', 'john', 'not', 'women', 'anna', 'john', 'steven', 'not', 'who']

        self.assertEqual(functions_analysis.determine_gender_SVO(input_data)['subject'].tolist(), expected)

    def test_determine_gender_premodifier(self):
        expected = ['elegant', 'powerful', 'vulnerable', 'strong', 'beautiful', 'california', 'world']

        self.assertEqual(functions_analysis.determine_gender_premodifier(input_data)['female_premodifier'].tolist(), expected)

    def test_determine_gender_postmodifier(self):
        expected = ['writer', 'drama', 'pregnant', 'nan', 'nan', 'nan']

        self.assertEqual(functions_analysis.determine_gender_postmodifier(input_data)['female_postmodifier'].tolist(), expected)
        
    def test_determine_gender_aux(self):
        expected = ['victim', 'teacher', 'engineer', '!victim', '!teacher', '!mathematician']

        self.assertEqual(functions_analysis.determine_gender_aux(input_data)['female_before_aux'].tolist(), expected)

    def test_determine_gender_possess(self):
        expected = ['right', 'nan']

        self.assertEqual(functions_analysis.determine_gender_possess(input_data)['female_possessive'].tolist(), expected)

    def test_gender_count(self):
        expected = [28]

        self.assertEqual(functions_analysis.gender_count(input_data)['female_count'].tolist(), expected)

    def test_determine_profession(self):
        data = {'token': ['teacher', 'cleaner', 'nurse', 'engineer'],
                'bias': [0.5, 0.7, 0.9, -0.8],
                'pos': ['NOUN', 'NOUN', 'NOUN', 'NOUN']}

        view_df = pd.DataFrame(data)

        expected = ['teacher', 'cleaner']

        self.assertEqual(functions_analysis.determine_gender_professions(view_df)['male_profession'].tolist(), expected)


if __name__ == '__main__':
    unittest.main()