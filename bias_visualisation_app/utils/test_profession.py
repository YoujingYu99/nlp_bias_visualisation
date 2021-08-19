import os
import numpy as np
import pandas as pd



def txt_profession_list():
    """
    :param txt_dir: the path of the txt files to be extracted
    :return: a clean list containing the raw sentences
    """
    profession_list = []
    profession_txt = os.path.join(os.path.dirname(__file__), '..', 'data', 'professions.txt')
    with open(profession_txt, 'r', encoding='utf-8') as file_in:
        for line in file_in:
            profession_list.append(line.strip())
        profession_list = [x.lower() for x in profession_list]
    return profession_list


view_df = pd.DataFrame({'token': ['firefighter', 'engineer', 'writer'], 'bias': [0.5, -0.25, -0.125], 'pos':['NOUN', 'NOUN', 'NOUN']})

def determine_gender_professions(view_df):
    profession_list = txt_profession_list()
    view_dict = view_df.to_dict('records')
    female_professions = []
    female_professions_bias = []
    male_professions = []
    male_professions_bias = []
    for item in view_dict:
        if item['pos'] == 'NOUN':
            if item['token'] in profession_list:
                if item['bias'] < 0:
                    female_professions.append(item['token'])
                    female_professions_bias.append(item['bias'])
                elif item['bias'] > 0:
                    male_professions.append(item['token'])
                    male_professions_bias.append(item['bias'])

    #convert list to dataframe
    list_of_series = [pd.Series(female_professions), pd.Series(female_professions_bias), pd.Series(male_professions), pd.Series(male_professions_bias)]
    profession_df = pd.concat(list_of_series, axis=1)
    profession_df.columns = ['female_profession', 'female_bias', 'male_profession', 'male_bias']

    return profession_df

print(view_df)
print(determine_gender_professions(view_df))