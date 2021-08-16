import os
import numpy as np
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer

spec_chars = ['!', ''','#','%','&',''', '(', ')',
              '*', '+', ',', '-', '.', '/', ':', ';', '<',
              '=', '>', '?', '@', '[', '\\', ']', '^', '_',
              '`', '{', '|', '}', '~', '–']

data = {'subject':['tom', 'nick', 'mary', 'jack'],
        'verb':['!hit', 'find', 'walked', '!telling'],
        'object':['jane', 'susan', 'neutral_instransitive', 'linda']}
SVO_df = pd.DataFrame(data)

def clean_SVO_dataframe(SVO_df):
    for char in spec_chars:
        SVO_df['subject'] = SVO_df['subject'].str.replace(char, ' ')
        SVO_df['object'] = SVO_df['object'].str.replace(char, ' ')
        #SVO_df['verb'] = SVO_df['verb'].str.replace(char, ' ')


    # get base form of verb
    verb_list = SVO_df['verb'].to_list()
    verb_base_list = []
    for verb in verb_list:
        verb.strip()
        if '!' in verb:
            verb = verb.replace('!', '')
            verb.strip()
            try:
                main_verb, particle = verb.split()[0], verb.split()[1]
                base_word = WordNetLemmatizer().lemmatize(main_verb, 'v')
                base_word.strip()
                base_phrasal_verb = '!' + base_word + ' ' + particle
                verb_base_list.append(base_phrasal_verb)
            except:
                verb = verb.split()[0]
                base_word = WordNetLemmatizer().lemmatize(verb, 'v')
                base_word.strip()
                verb_base_list.append('!' + base_word)
        else:
            verb.strip()
            try:
                main_verb, particle = verb.split()[0], verb.split()[1]
                base_word = WordNetLemmatizer().lemmatize(main_verb, 'v')
                base_word.strip()
                base_phrasal_verb = base_word + ' ' + particle
                verb_base_list.append(base_phrasal_verb)
            except:
                verb = verb.split()[0]
                base_word = WordNetLemmatizer().lemmatize(verb, 'v')
                base_word.strip()
                verb_base_list.append(base_word)

    SVO_df['verb'] = verb_base_list
    SVO_df = SVO_df.apply(lambda x: x.astype(str).str.lower())

    return SVO_df

print(clean_SVO_dataframe(SVO_df))
