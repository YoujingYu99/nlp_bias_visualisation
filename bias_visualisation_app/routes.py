"""
The interactive web interface for data bias visualisation
"""

from __future__ import unicode_literals
from flask_caching import Cache
from flask import redirect, render_template, url_for, request, send_from_directory
from bias_visualisation_app import app
from os import path
from bias_visualisation_app.utils.functions import get_text_url, get_text_file, generate_list,\
    SVO_analysis, premodifier_analysis, postmodifier_analysis, aux_analysis, possess_analysis,gender_count_analysis,\
    generate_bias_values, save_obj, save_obj_user_uploads, load_obj_user_uploads, \
    frame_from_file, bar_graph, specific_bar_graph, cloud_image, tsne_graph, tsne_graph_male, \
    tsne_graph_female, pca_graph, \
    pca_graph_male, pca_graph_female, gender_dataframe_from_tuple, parse_pos_dataframe, df_based_on_question
import werkzeug
import spacy
import pandas as pd
import sys

sys.setrecursionlimit(10000)
cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})

# from urllib.request import urlopen
# from urllib3 import urlopen

nlp = spacy.load('en_core_web_sm')


#
# # Sumy Pkg
# from sumy.parsers.plaintext import PlaintextParser
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.summarizers.lex_rank import LexRankSummarizer
#
#
# # Sumy
# def sumy_summary(docx):
#     parser = PlaintextParser.from_string(docx, Tokenizer('english'))
#     lex_summarizer = LexRankSummarizer()
#     summary = lex_summarizer(parser.document, 3)
#     summary_list = [str(sentence) for sentence in summary]
#     result = ' '.join(summary_list)
#     return result
#
#
# # Reading Time
# def readingTime(mytext):
#     total_words = len([token.text for token in nlp(mytext)])
#     estimatedTime = total_words / 200.0
#     return estimatedTime
#


@app.route('/')
def index():
    return render_template('index.html')


# @app.route('/analyze', methods=['GET', 'POST'])
# def analyze():
#     start = time.time()
#     if request.method == 'POST':
#         rawtext = request.form['rawtext']
#         final_reading_time = readingTime(rawtext)
#         final_summary = text_summarizer(rawtext)
#         summary_reading_time = readingTime(final_summary)
#         end = time.time()
#         final_time = end - start
#     return render_template('index.html', ctext=rawtext, final_summary=final_summary, final_time=final_time,
#                            final_reading_time=final_reading_time, summary_reading_time=summary_reading_time)


# @app.route('/analyze_url', methods=['GET', 'POST'])
# def analyze_url():
#     start = time.time()
#     if request.method == 'POST':
#         raw_url = request.form['raw_url']
#         rawtext = get_text(raw_url)
#         final_reading_time = readingTime(rawtext)
#         final_summary = text_summarizer(rawtext)
#         summary_reading_time = readingTime(final_summary)
#         end = time.time()
#         final_time = end - start
#     return render_template('index.html', ctext=rawtext, final_summary=final_summary, final_time=final_time,
#                            final_reading_time=final_reading_time, summary_reading_time=summary_reading_time)


@app.route('/visualisation')
def visualisation():
    view_df = load_obj_user_uploads(name='total_dataframe_user_uploads')
    token_list, value_list = generate_list(view_df)[0], generate_list(view_df)[1]

    # plot the bar graphs and word clouds
    plot_bar = bar_graph(view_df, token_list, value_list)
    plot_female_cloud, plot_male_cloud = cloud_image(token_list, value_list)
    # only perform tsne plot if more than 100 tokens
    if len(token_list) > 100:
        plot_tsne = tsne_graph(token_list)
        plot_tsne_male = tsne_graph_male(token_list, value_list)
        plot_tsne_female = tsne_graph_female(token_list, value_list)
        plot_pca = pca_graph(token_list)
        plot_pca_male = pca_graph_male(token_list, value_list)
        plot_pca_female = pca_graph_female(token_list, value_list)
    else:
        plot_tsne = url_for('static', filename='nothing_here.png')
        plot_tsne_male = url_for('static', filename='nothing_here.png')
        plot_tsne_female = url_for('static', filename='nothing_here.png')
        plot_pca = url_for('static', filename='nothing_here.png')
        plot_pca_male = url_for('static', filename='nothing_here.png')
        plot_pca_female = url_for('static', filename='nothing_here.png')

    return render_template('visualisation.html', bar_graph=plot_bar,
                           female_word_cloud=plot_female_cloud, male_word_cloud=plot_male_cloud,
                           tsne_graph=plot_tsne,
                           male_tsne_graph=plot_tsne_male, female_tsne_graph=plot_tsne_female, pca_graph=plot_pca,
                           male_pca_graph=plot_pca_male, female_pca_graph=plot_pca_female)


# @app.route('/comparer', methods=['GET', 'POST'])
# def comparer():
#     start = time.time()
#     if request.method == 'POST':
#         rawtext = request.form['rawtext']
#         final_reading_time = readingTime(rawtext)
#         final_summary_spacy = text_summarizer(rawtext)
#         summary_reading_time = readingTime(final_summary_spacy)
#         # Gensim Summarizer
#         final_summary_gensim = summarize(rawtext)
#         summary_reading_time_gensim = readingTime(final_summary_gensim)
#         # NLTK
#         final_summary_nltk = nltk_summarizer(rawtext)
#         summary_reading_time_nltk = readingTime(final_summary_nltk)
#         # Sumy
#         final_summary_sumy = sumy_summary(rawtext)
#         summary_reading_time_sumy = readingTime(final_summary_sumy)
#
#         end = time.time()
#         final_time = end - start
#     return render_template('visualisation.html', ctext=rawtext, final_summary_spacy=final_summary_spacy,
#                            final_summary_gensim=final_summary_gensim, final_summary_nltk=final_summary_nltk,
#                            final_time=final_time, final_reading_time=final_reading_time,
#                            summary_reading_time=summary_reading_time,
#                            summary_reading_time_gensim=summary_reading_time_gensim,
#                            final_summary_sumy=final_summary_sumy, summary_reading_time_sumy=summary_reading_time_sumy,
#                            summary_reading_time_nltk=summary_reading_time_nltk)


@app.route('/detect_text', methods=['GET', 'POST'])
def detect_text():
    if request.method == 'POST':
        input_data = request.form['rawtext']
        # sentence = request.args.get('sentence')
        if not input_data:
            raise werkzeug.exceptions.BadRequest('You must provide a paragraph')
        if len(input_data) > 50000:
            raise werkzeug.exceptions.BadRequest(
                'Input Paragraph must be at most 500000 characters long'
            )
        generate_bias_values(input_data)

    return render_template('index.html')


@app.route('/detect_url', methods=['GET', 'POST'])
def detect_url():
    if request.method == 'POST':
        raw_url = request.form['raw_url']
        input_data = get_text_url(raw_url)
        if not input_data:
            raise werkzeug.exceptions.BadRequest('You must provide a paragraph')
        if len(input_data) > 50000:
            raise werkzeug.exceptions.BadRequest(
                'Input Paragraph must be at most 500000 characters long'
            )
        generate_bias_values(input_data)

    return render_template('index.html')


@app.route('/detect_corpora', methods=['GET', 'POST'])
def detect_corpora():
    if request.method == 'POST':
        corpora_file = request.files['raw_file']
        input_data = get_text_file(corpora_file)
        if not input_data:
            raise werkzeug.exceptions.BadRequest('You must provide a paragraph')
        if len(input_data) > 50000:
            raise werkzeug.exceptions.BadRequest(
                'Input Paragraph must be at most 500000 characters long'
            )
        generate_bias_values(input_data)
    return render_template('index.html')


@app.route('/detect_dataframe', methods=['GET', 'POST'])
def detect_dataframe():
    if request.method == 'POST':
        complete_file = request.files['complete_file']
        dataframe_SVO = pd.read_excel(complete_file, sheet_name='SVO_dataframe')
        dataframe_premodifier = pd.read_excel(complete_file, sheet_name='premodifier_dataframe')
        dataframe_postmodifier = pd.read_excel(complete_file, sheet_name='postmodifier_dataframe')
        dataframe_aux = pd.read_excel(complete_file, sheet_name='aux_dataframe')
        dataframe_possess = pd.read_excel(complete_file, sheet_name='possess_dataframe')
        dataframe_gender_count = pd.read_excel(complete_file, sheet_name='gender_count_dataframe')
        dataframe_total = pd.read_excel(complete_file, sheet_name='total_dataframe')

        input_dataframe_total = dataframe_total
        save_obj_user_uploads(input_dataframe_total, name='total_dataframe_user_uploads')

        input_dataframe_SVO = dataframe_SVO
        save_obj_user_uploads(input_dataframe_SVO, name='SVO_dataframe_user_uploads')

        input_dataframe_premodifier = dataframe_premodifier
        save_obj_user_uploads(input_dataframe_premodifier, name='premodifier_dataframe_user_uploads')

        input_dataframe_postmodifier = dataframe_postmodifier
        save_obj_user_uploads(input_dataframe_postmodifier, name='postmodifier_dataframe_user_uploads')

        input_dataframe_aux = dataframe_aux
        save_obj_user_uploads(input_dataframe_aux, name='aux_dataframe_user_uploads')

        input_dataframe_possess = dataframe_possess
        save_obj_user_uploads(input_dataframe_possess, name='possess_dataframe_user_uploads')

        input_dataframe_gender_count = dataframe_gender_count
        save_obj_user_uploads(input_dataframe_gender_count, name='gender_count_dataframe_user_uploads')

        return redirect(url_for('visualisation'))


# . It works by looking at differences between male and female word pairs
#       like 'he' and 'she', or 'boy' and 'girl', and then comparing the
#       differences between those words to other word vectors in the word2vec
#       dataset.

# >0: is male biased
# <0: is female biased

@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    # open dataframe file
    view_df = load_obj_user_uploads(name='total_dataframe_user_uploads')
    input_SVO_dataframe = load_obj_user_uploads(name='SVO_dataframe_user_uploads')
    input_premodifier_dataframe = load_obj_user_uploads(name='premodifier_dataframe_user_uploads')
    input_postmodifier_dataframe = load_obj_user_uploads(name='postmodifier_dataframe_user_uploads')
    input_aux_dataframe = load_obj_user_uploads(name='aux_dataframe_user_uploads')
    input_possess_dataframe = load_obj_user_uploads(name='possess_dataframe_user_uploads')
    input_gender_count_dataframe = load_obj_user_uploads(name='gender_count_dataframe_user_uploads')

    # view_df = frame_from_file(input_dataframe)[0]
    female_tot_df, male_tot_df = gender_dataframe_from_tuple(view_df)
    female_noun_df, female_adj_df, female_verb_df = parse_pos_dataframe(view_df)[:3]
    male_noun_df, male_adj_df, male_verb_df = parse_pos_dataframe(view_df)[-3:]
    female_sub_df, female_obj_df, female_intran_df, male_sub_df, male_obj_df, male_intran_df = SVO_analysis(input_SVO_dataframe)
    female_premodifier_df, male_premodifier_df = premodifier_analysis(input_premodifier_dataframe)
    female_postmodifier_df, male_postmodifier_df = postmodifier_analysis(input_postmodifier_dataframe)
    female_before_aux_df, male_before_aux_df, female_follow_aux_df, male_follow_aux_df = aux_analysis(input_aux_dataframe)
    female_possessive_df, male_possessive_df, female_possessor_df, male_possessor_df = possess_analysis(input_possess_dataframe)
    female_count, male_count = gender_count_analysis(input_gender_count_dataframe)

    return render_template('analysis.html', female_count=female_count, male_count=male_count, data_fm_tot=female_tot_df, data_m_tot=male_tot_df,
                           data_fm_noun=female_noun_df, data_m_noun=male_noun_df, data_fm_adj=female_adj_df,
                           data_m_adj=male_adj_df, data_fm_verb=female_verb_df, data_m_verb=male_verb_df,
                           data_fm_intran_verb=female_intran_df,
                           data_fm_sub_verb=female_sub_df, data_fm_obj_verb=female_obj_df,
                           data_m_intran_verb=male_intran_df, data_m_sub_verb=male_sub_df,
                           data_m_obj_verb=male_obj_df, data_fm_premodifier=female_premodifier_df, data_m_premodifier=male_premodifier_df, data_fm_postmodifier=female_postmodifier_df, data_m_postmodifier=male_postmodifier_df, data_fm_before_aux=female_before_aux_df, data_m_before_aux=male_before_aux_df,
                           data_fm_follow_aux=female_follow_aux_df, data_m_follow_aux=male_follow_aux_df, data_fm_possessive=female_possessive_df, data_m_possessive=male_possessive_df, data_fm_possessor=female_possessor_df, data_m_possessor=male_possessor_df,
                           wordtype_data=[{'type': 'nouns'}, {'type': 'adjectives'}, {'type': 'intransitive_verbs'}, {'type': 'subject_verbs'},
                                          {'type': 'object_verbs'}, {'type': 'premodifiers'}, {'type': 'auxs'}, {'type': 'before_aux'}, {'type': 'follow_aux'}, {'type': 'possessives'}, {'type': 'possessors'}],
                           gender_data=[{'type': 'female'}, {'type': 'male'}])


@app.route('/query', methods=['GET', 'POST'])
def query():
    dataframe_to_display = None
    select_gender = None
    select_wordtype = None
    plot_bar = None
    if request.method == 'POST':
        # open dataframe file
        view_df = load_obj_user_uploads(name='total_dataframe_user_uploads')
        input_SVO_dataframe = load_obj_user_uploads(name='SVO_dataframe_user_uploads')
        input_premodifier_dataframe = load_obj_user_uploads(name='premodifier_dataframe_user_uploads')
        input_postmodifier_dataframe = load_obj_user_uploads(name='postmodifier_dataframe_user_uploads')
        input_aux_dataframe = load_obj_user_uploads(name='aux_dataframe_user_uploads')
        input_possess_dataframe = load_obj_user_uploads(name='possess_dataframe_user_uploads')

        select_wordtype = request.form.get('type_select')
        select_gender = request.form.get('gender_select')
        dataframe_to_display = df_based_on_question(str(select_wordtype), str(select_gender), view_df,
                                                    input_SVO_dataframe, input_premodifier_dataframe, input_postmodifier_dataframe, input_aux_dataframe, input_possess_dataframe)
        save_obj(dataframe_to_display, name='specific_df')
        plot_bar = specific_bar_graph(df_name='specific_df')

    return render_template('query.html', data_question=dataframe_to_display, gender_in_question=str(select_gender),
                           type_in_question=str(select_wordtype), bar_graph_specific=plot_bar)


# @app.route('/analyse_adj', methods=['GET', 'POST'])
# def analyse_adj():
#     if request.method == 'POST':
#         # rawtext = request.form['rawtext']
#         # female_dataframe_tot, male_dataframe_tot = gender_dataframe_from_dict(m_dic, fm_dic)
#         # if 'adjectives' in rawtext:
#         #     if 'female' in rawtext:
#         #         female_adjs = female_adjs()
#         #     elif 'male' in rawtext:
#         #         male_adjs = male_adjs()
#         #     else:
#         #         print('Please enter a valid question')
#
#     return render_template('query.html', ctext=rawtext, data_fm_tot=female_dataframe_tot, data_m_tot=male_dataframe_tot)
#
# @app.route('/analyse_question', methods=['GET', 'POST'])
# def analyse_question():
#     if request.method == 'POST':
#         select_wordtype = request.form.get('type_select')
#         select_gender = request.form.get('gender_select')
#         dataframe_to_display = df_based_on_question(str(select_wordtype), str(select_gender))
#
#         print(dataframe_to_display)
#
#     return render_template('query.html', data_question=dataframe_to_display)


@app.route('/download/<df_name>', methods=['GET', 'POST'])
def download(df_name):
    print('calling download_total_route')
    uploads = path.join(path.dirname(__file__), 'static', 'user_downloads', df_name)
    total_data = pd.read_excel(uploads)
    print(total_data)
    return send_from_directory(directory=app.config['DOWNLOAD_FOLDER'], filename=df_name, as_attachment=True)
    #return send_file(uploads, as_attachment=True)


# @app.route('/uploads/<path:filename>', methods=['GET', 'POST'])
# def download_total(filename):
#     print('calling download_total_route')
#     uploads = path.join(path.dirname(__file__), 'static', filename)
#     total_data = pd.read_csv(uploads)
#     print(total_data)
#     return send_from_directory(directory=app.config['UPLOAD_FOLDER'], filename=filename, as_attachment=True)
#     #return send_file(uploads, as_attachment=True)
#
#
# @app.route('/uploads/<path:filename>', methods=['GET', 'POST'])
# def download_SVO(filename='SVO_dataframe.csv'):
#     uploads = path.join(path.dirname(__file__), 'static', filename)
#
#     return send_from_directory(directory=app.static_folder, filename=filename, as_attachment=True)
#
# @app.route('/uploads/<path:filename>', methods=['GET', 'POST'])
# def download_premodifier(filename='premodifier_dataframe.csv'):
#     uploads = path.join(path.dirname(__file__), 'static', filename)
#     print(uploads)
#     # return send_from_directory(directory=uploads, filename=filename)
#     return send_from_directory(directory=app.static_folder, filename=filename, as_attachment=True)
#
# @app.route('/uploads/<path:filename>', methods=['GET', 'POST'])
# def download_aux(filename='aux_dataframe.csv'):
#     uploads = path.join(path.dirname(__file__), 'static', filename)
#     print(uploads)
#     # return send_from_directory(directory=uploads, filename=filename)
#     return send_from_directory(directory=app.static_folder, filename=filename, as_attachment=True)

@app.route('/about')
def about():
    return render_template('about.html')


# text for testing functions
# "Women writers support male fighters. Male cleaners are not more careful. Lucy likes female dramas. Women do not like sunglasses. Lucy eats a tasty black bread. The elegant powerful woman wears shiny black glasses. The dark tall man drinks water. He admires vulnerable strong women. The kind beautiful girl picks a cup. Most writers are female. The majority are women. The suspect is a woman. Her father was a man who lived an extraordinary life. Women are victims. Men are not minority. The woman is a teacher. Sarah is an engineer. The culprit is not Linda.We need to protect women's rights. Men's health is as important. I can look after the Simpsons' cat. Japan's women live longest. Canada's John did not cling a gold prize. The world's women should unite together. Anna looks up a book. John asked Marilyn out. Steven did not take the coat off. Most writers are a woman. Most writers are not male. The teacher is not a man. The majority are women. The suspect is a woman. Her father was a man who lived an extraordinary life. Women are not victims. Men are minority. The woman isn't a teacher. Sarah is not a nurse."