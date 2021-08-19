from bias_visualisation_app.utils.functions import gender_dataframe_from_tuple, parse_pos_dataframe, \
premodifier_analysis, postmodifier_analysis, aux_analysis, possess_analysis, profession_analysis, SVO_analysis

user_question1 = 'What nouns are women mostly associated with?'
user_question2 = 'What actions are usually performed by women?'
user_question3 = 'What are the auxiliary words that appear after male nouns?'

female_synonyms = ['woman', 'women', 'female', 'females']
male_synonyms = ['man', 'men', 'male', 'males']
# Nouns, adjectives, verbs, aux_before, aux_follow, possessor, possessive, postmodifier, premodifier, profession

# Python program to check
# if two lists have at-least
# one element common
# using set and property

def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return True
    else:
        return False

def analyse_question(input_question, view_df, input_SVO_dataframe, input_premodifier_dataframe,
                         input_postmodifier_dataframe, input_aux_dataframe, input_possess_dataframe, input_profession_dataframe):
    female_tot_df, male_tot_df = gender_dataframe_from_tuple(view_df)
    female_noun_df, female_adj_df, female_verb_df = parse_pos_dataframe(view_df)[:3]
    male_noun_df, male_adj_df, male_verb_df = parse_pos_dataframe(view_df)[-3:]
    female_sub_df, female_obj_df, female_intran_df, male_sub_df, male_obj_df, male_intran_df = SVO_analysis(
        input_SVO_dataframe)
    female_premodifier_df, male_premodifier_df = premodifier_analysis(input_premodifier_dataframe)
    female_postmodifier_df, male_postmodifier_df = postmodifier_analysis(input_postmodifier_dataframe)
    female_before_aux_df, male_before_aux_df, female_follow_aux_df, male_follow_aux_df = aux_analysis(
        input_aux_dataframe)
    female_possessive_df, male_possessive_df, female_possessor_df, male_possessor_df = possess_analysis(
        input_possess_dataframe)
    female_profession_df, male_profession_df = profession_analysis(input_profession_dataframe)

    user_question = input_question.lower()
    user_question_list = user_question.split()
    # first step is to determine gender
    if common_member(user_question_list, female_synonyms) is True and common_member(user_question_list, male_synonyms) is False:
        # second step is to parse the dataform
        if 'nouns' in user_question_list:
            return female_noun_df
        if 'adjectives' in user_question_list:
            return female_adj_df
        if 'verbs' in user_question_list and 'intransitive' not in user_question_list:
            return female_verb_df
        if 'auxiliary' in user_question_list and 'before' in user_question_list:
            return female_before_aux_df
        if 'auxiliary' in user_question_list and 'follow' in user_question_list:
            return female_follow_aux_df
        if 'possessors' in user_question_list:
            return female_possessor_df
        if 'possessives' in user_question_list:
            return female_possessive_df
        if 'premodifiers' in user_question_list:
            return female_premodifier_df
        if 'postmodifiers' in user_question_list:
            return female_postmodifier_df
        if 'professions' in user_question_list or 'jobs' in user_question_list:
            return female_profession_df
        if 'intransitive' in user_question_list:
            return female_intran_df
        if 'actions' in user_question_list and 'by' in user_question_list:
            return female_sub_df
        if 'actions' in user_question_list and 'against' in user_question_list:
            return female_obj_df




    elif common_member(user_question_list, female_synonyms) is False and common_member(user_question_list, male_synonyms) is True:
        if 'nouns' in user_question_list:
            return male_noun_df
        if 'adjectives' in user_question_list:
            return male_adj_df
        if 'verbs' in user_question_list and 'intransitive' not in user_question_list:
            return male_verb_df
        if 'auxiliary' in user_question_list and 'before' in user_question_list:
            return male_before_aux_df
        if 'auxiliary' in user_question_list and 'follow' in user_question_list:
            return male_follow_aux_df
        if 'possessors' in user_question_list:
            return male_possessor_df
        if 'possessives' in user_question_list:
            return male_possessive_df
        if 'premodifiers' in user_question_list:
            return male_premodifier_df
        if 'postmodifiers' in user_question_list:
            return male_postmodifier_df
        if 'professions' in user_question_list or 'jobs' in user_question_list:
            return male_profession_df
        if 'intransitive' in user_question_list:
            return male_intran_df
        if 'actions' in user_question_list and 'by' in user_question_list:
            return male_sub_df
        if 'actions' in user_question_list and 'against' in user_question_list:
            return male_obj_df
