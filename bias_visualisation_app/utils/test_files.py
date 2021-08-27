import unittest
import os
import glob
import pathlib as pl
import sys
from os import listdir
from io import open
from conllu import parse_incr
import csv
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import requests
from werkzeug.utils import secure_filename
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pandas as pd
import functions_files

class TestFiles(unittest.TestCase):

    def test_txt_list(self):
        path_parent = os.path.dirname(os.getcwd())
        expected = [['time', 'would', 'embraced', 'change', 'coming', 'youth', 'sought', 'adventure', 'unknown', 'years', 'ago', 'wished', 'could', 'go', 'back', 'learn', 'find', 'excitement', 'came', 'change', 'useless', 'curiosity', 'long', 'left', 'come', 'loathe', 'anything', 'put', 'comfort', 'zone']]
        self.assertEqual(functions_files.txt_list(os.path.join(path_parent, 'utils', 'test')), expected)

    def test_concat_csv_excel(self):
        parent_path = os.path.dirname(os.getcwd())
        csv_path = os.path.join(parent_path,  'static', 'user_downloads')
        functions_files.concat_csv_excel()
        xlsx_exist = True
        if not any(fname.endswith('.xlsx') for fname in os.listdir(csv_path)):
            xlsx_exist = False

        self.assertEquals(xlsx_exist, True)

    def test_load_obj(self):
        expected = ['support', 'careful', 'bread', 'powerful', 'dark', 'tall', 'drink', 'water', 'strong', 'beautiful', 'girl',
         'cup', 'need', 'protect', 'health', 'important', 'cat', 'live', 'world', 'together', 'book', 'asked', 'take',
         'coat', 'carefully', 'treated', 'pilot', 'still', 'bit', 'equation', 'matter', 'much', 'tried', 'positive',
         'anywhere', 'seen', 'coming', 'pretty', 'get', 'back', 'neck', 'sometimes', 'got', 'talking', 'believe',
         'always', 'good', 'bring', 'friend', 'happen', 'come', 'across', 'whole', 'worry', 'learn', 'living',
         'ordinary', 'even', 'explain', 'mere', 'fact', 'exist', 'make', 'existing', 'le', 'winning', 'let', 'question',
         'kept', 'asking', 'looked', 'around', 'daily', 'reached', 'goal', 'also', 'beginning', 'made', 'choice',
         'fine', 'willing', 'face', 'consequence']
        self.assertEqual(functions_files.load_obj(name='m_dic')['token'].tolist(), expected)

    def test_load_obj_user_uploads(self):
        expected = [30]
        self.assertEqual(functions_files.load_obj_user_uploads(name='gender_count_dataframe_user_uploads')['female_count'].tolist(), expected)


if __name__ == '__main__':
    unittest.main()