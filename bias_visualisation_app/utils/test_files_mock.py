import unittest
import os
import functions_files

class TestFiles(unittest.TestCase):

    def test_txt_list(self):
        path_parent = os.path.dirname(os.getcwd())
        expected = [['time', 'would', 'embraced', 'change', 'coming', 'youth', 'sought', 'adventure', 'unknown', 'years', 'ago', 'wished', 'could', 'go', 'back', 'learn', 'find', 'excitement', 'came', 'change', 'useless', 'curiosity', 'long', 'left', 'come', 'loathe', 'anything', 'put', 'comfort', 'zone']]
        self.assertEqual(functions_files.txt_list(os.path.join(path_parent, 'utils', 'test')), expected)

    def test_concat_csv_excel(self):
        parent_path = os.path.dirname(os.getcwd())
        csv_path = os.path.join(parent_path,  'static', 'user_downloads')
        functions_files.concat_csv_excel(csv_path)
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

        path_parent = os.path.dirname(os.getcwd())
        df_path = os.path.join(path_parent, 'visualising_data_bias', 'bias_visualisation_app', 'static')
        self.assertEqual(functions_files.load_obj(df_path, 'm_dic')['token'].tolist(), expected)

    def test_load_obj_user_uploads(self):
        expected = [30]
        path_parent = os.path.dirname(os.getcwd())
        df_path = os.path.join(path_parent, 'visualising_data_bias', 'bias_visualisation_app', 'static', 'user_uploads')
        self.assertEqual(functions_files.load_obj_user_uploads(df_path,name='gender_count_dataframe_user_uploads')['female_count'].tolist(), expected)


if __name__ == '__main__':
    unittest.main()