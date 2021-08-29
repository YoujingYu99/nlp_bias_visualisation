import unittest
import tempfile
import os

import pandas as pd

import functions_files

path_parent = os.path.dirname(os.getcwd())
test_path = os.path.join(path_parent, 'utils', 'test')


class TestFiles(unittest.TestCase):

    def test_txt_list(self):
        expected = [
            ['time', 'would', 'embraced', 'change', 'coming', 'youth', 'sought', 'adventure', 'unknown', 'years', 'ago',
             'wished', 'could', 'go', 'back', 'learn', 'find', 'excitement', 'came', 'change', 'useless', 'curiosity',
             'long', 'left', 'come', 'loathe', 'anything', 'put', 'comfort', 'zone']]
        self.assertEqual(functions_files.txt_list(test_path), expected)

    def test_concat_csv_excel(self):
        data1 = pd.DataFrame({'token': ['teacher', 'cleaner', 'nurse', 'engineer'],
                              'bias': [0.5, 0.7, 0.9, -0.8],
                              'pos': ['NOUN', 'NOUN', 'NOUN', 'NOUN']})

        save_name = os.path.join(test_path, '1')
        df_path_1 = save_name + '.csv'
        data1.to_csv(df_path_1, index=False)

        data2 = pd.DataFrame({'token': ['fighter', 'lecturer', 'butcher', 'baker'],
                              'bias': [0.3, 0.4, 0.8, -0.6],
                              'pos': ['NOUN', 'NOUN', 'NOUN', 'NOUN']})

        save_name = os.path.join(test_path, '2')
        df_path_2 = save_name + '.csv'
        data2.to_csv(df_path_2, index=False)

        functions_files.concat_csv_excel(csv_path=test_path)
        xlsx_exist = True
        if not any(fname.endswith('.xlsx') for fname in os.listdir(test_path)):
            xlsx_exist = False

        self.assertEquals(xlsx_exist, True)

    def test_load_obj(self):
        data = pd.DataFrame({'token': ['teacher', 'cleaner', 'nurse', 'engineer'],
                             'bias': [0.5, 0.7, 0.9, -0.8],
                             'pos': ['NOUN', 'NOUN', 'NOUN', 'NOUN']})

        save_name = os.path.join(test_path, 'load_obj_test')
        save_path = save_name + '.csv'
        data.to_csv(save_path, index=False)

        expected = ['teacher', 'cleaner', 'nurse', 'engineer']
        self.assertEqual(functions_files.load_obj(test_path, 'load_obj_test')['token'].tolist(), expected)

    def test_load_obj_user_uploads(self):
        data = pd.DataFrame({'female_count': [30],
                             'male_count': [40]})

        save_name = os.path.join(test_path, 'gender_count')
        save_path = save_name + '.csv'
        data.to_csv(save_path, index=False)

        expected = [30]

        self.assertEqual(functions_files.load_obj_user_uploads(test_path, name='gender_count')[
                             'female_count'].tolist(), expected)


if __name__ == '__main__':
    unittest.main()
