"""
The interactive web interface for data bias visualisation
"""

from __future__ import unicode_literals
from flask_caching import Cache
from flask import redirect, render_template, url_for, request, send_from_directory, flash
from bias_visualisation_app import app
from bias_visualisation_app.utils.functions_files import get_text_url, get_text_file, save_user_file_text,  save_obj_user_uploads, load_obj_user_uploads
from bias_visualisation_app.utils.functions_analysis import  generate_list,\
    SVO_analysis, premodifier_analysis, postmodifier_analysis, aux_analysis, possess_analysis, profession_analysis, gender_count_analysis,\
    generate_bias_values, save_obj, gender_dataframe_from_tuple, parse_pos_dataframe, analyse_question, debiased_file, style_dataframe, df_tot
from bias_visualisation_app.utils.functions_graphs import bar_graph, specific_bar_graph, cloud_image, tsne_graph, tsne_graph_male, \
    tsne_graph_female, pca_graph, \
    pca_graph_male, pca_graph_female
import werkzeug
import spacy
import pandas as pd
import sys
import os

sys.setrecursionlimit(10000)
cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})


nlp = spacy.load('en_core_web_sm')
nlp.max_length = 10**10


@app.route('/')
def index():
    return render_template('index.html')




@app.route('/visualisation')
def visualisation():
    path_parent = os.path.dirname(os.path.abspath(__file__))
    df_path = os.path.join(path_parent, 'static', 'user_uploads')
    view_df = load_obj_user_uploads(df_path, name='total_dataframe_user_uploads')
    token_list, value_list = generate_list(view_df)[0], generate_list(view_df)[1]

    # plot the bar graphs and word clouds
    plot_bar = bar_graph(view_df, token_list, value_list)
    plot_female_cloud, plot_male_cloud = cloud_image(token_list, value_list)
    # only perform tsne plot if more than 100 tokens
    if len(token_list) > 100:
        plot_tsne = tsne_graph(token_list, 'rgba(91,221,191,0.5)')
        plot_tsne_male = tsne_graph_male(token_list, value_list, 'rgba(72,176,152,0.5)')
        plot_tsne_female = tsne_graph_female(token_list, value_list, 'rgba(57,140,121,0.5)')
        plot_pca = pca_graph(token_list, 'rgba(91,221,191,0.5)')
        plot_pca_male = pca_graph_male(token_list, value_list, 'rgba(72,176,152,0.5)')
        plot_pca_female = pca_graph_female(token_list, value_list, 'rgba(57,140,121,0.5)')
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



@app.route('/detect_text', methods=['GET', 'POST'])
def detect_text():
    if request.method == 'POST':
        try:
            input_data = request.form['rawtext']
            save_user_file_text(input_data)
            if not input_data:
                raise werkzeug.exceptions.BadRequest('You must provide a paragraph')
            if len(input_data) > 1500000:
                raise werkzeug.exceptions.BadRequest(
                    'Input Paragraph must be at most 1500000 characters long'
                )
            generate_bias_values(input_data)
            flash('Your file is ready for download!', 'info')
        except:
            flash('Please enter a valid text.', 'danger')

    return render_template('index.html')


@app.route('/detect_url', methods=['GET', 'POST'])
def detect_url():
    if request.method == 'POST':
        try:
            raw_url = request.form['raw_url']
            input_data = get_text_url(raw_url)
            save_user_file_text(input_data)
            if not input_data:
                raise werkzeug.exceptions.BadRequest('You must provide a paragraph')
            if len(input_data) > 1500000:
                raise werkzeug.exceptions.BadRequest(
                    'Input Paragraph must be at most 1500000 characters long'
                )
            generate_bias_values(input_data)
            flash('Your file is ready for download!', 'info')
        except:
            flash('Please enter a valid URL.', 'danger')

    return render_template('index.html')


@app.route('/detect_corpora', methods=['GET', 'POST'])
def detect_corpora():
    if request.method == 'POST':
        corpora_file = request.files['raw_corpora']
        input_data = get_text_file(corpora_file)
        save_user_file_text(input_data)
        if not input_data:
            raise werkzeug.exceptions.BadRequest('You must provide a paragraph')
        if len(input_data) > 1500000:
            raise werkzeug.exceptions.BadRequest(
                'Input Paragraph must be at most 1500000 characters long'
            )
        generate_bias_values(input_data)
        flash('Your file is ready for download!', 'info')

    return render_template('index.html')

@app.route('/detect_dataframe', methods=['GET', 'POST'])
def detect_dataframe():
    if request.method == 'POST':
        try:
            complete_file = request.files['complete_file']
            dataframe_SVO = pd.read_excel(complete_file, sheet_name='SVO_dataframe', engine='openpyxl')
            dataframe_premodifier = pd.read_excel(complete_file, sheet_name='premodifier_dataframe', engine='openpyxl')
            dataframe_postmodifier = pd.read_excel(complete_file, sheet_name='postmodifier_dataframe', engine='openpyxl')
            dataframe_aux = pd.read_excel(complete_file, sheet_name='aux_dataframe', engine='openpyxl')
            dataframe_possess = pd.read_excel(complete_file, sheet_name='possess_dataframe', engine='openpyxl')
            dataframe_profession = pd.read_excel(complete_file, sheet_name='profession_dataframe', engine='openpyxl')
            dataframe_gender_count = pd.read_excel(complete_file, sheet_name='gender_count_dataframe', engine='openpyxl')
            dataframe_total = pd.read_excel(complete_file, sheet_name='total_dataframe', engine='openpyxl')

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

            input_dataframe_profession = dataframe_profession
            save_obj_user_uploads(input_dataframe_profession, name='profession_dataframe_user_uploads')

            input_dataframe_gender_count = dataframe_gender_count
            save_obj_user_uploads(input_dataframe_gender_count, name='gender_count_dataframe_user_uploads')

            return redirect(url_for('visualisation'))

        except:
            flash('Please input the complete excel file.', 'danger')
            return redirect(url_for('index'))

@app.route('/sample_dataframe_ANC', methods=['GET', 'POST'])
def sample_dataframe_ANC():
    if request.method == 'POST':
        try:
            path_parent = os.path.dirname(os.path.abspath(__file__))
            df_path = os.path.join(path_parent, 'resources', 'sample_dataframe_ANC.xlsx', engine='openpyxl')
            dataframe_SVO = pd.read_excel(df_path, sheet_name='SVO_dataframe', engine='openpyxl')
            dataframe_premodifier = pd.read_excel(df_path, sheet_name='premodifier_dataframe', engine='openpyxl')
            dataframe_postmodifier = pd.read_excel(df_path, sheet_name='postmodifier_dataframe', engine='openpyxl')
            dataframe_aux = pd.read_excel(df_path, sheet_name='aux_dataframe', engine='openpyxl')
            dataframe_possess = pd.read_excel(df_path, sheet_name='possess_dataframe', engine='openpyxl')
            dataframe_profession = pd.read_excel(df_path, sheet_name='profession_dataframe', engine='openpyxl')
            dataframe_gender_count = pd.read_excel(df_path, sheet_name='gender_count_dataframe', engine='openpyxl')
            dataframe_total = pd.read_excel(df_path, sheet_name='total_dataframe', engine='openpyxl')

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

            input_dataframe_profession = dataframe_profession
            save_obj_user_uploads(input_dataframe_profession, name='profession_dataframe_user_uploads')

            input_dataframe_gender_count = dataframe_gender_count
            save_obj_user_uploads(input_dataframe_gender_count, name='gender_count_dataframe_user_uploads')

            return redirect(url_for('visualisation'))

        except:
            flash('Sample file not found!', 'danger')
            return redirect(url_for('index'))

@app.route('/sample_dataframe_enwiki', methods=['GET', 'POST'])
def sample_dataframe_enwiki():
    if request.method == 'POST':
        # try:
        path_parent = os.path.dirname(os.path.abspath(__file__))
        df_path = os.path.join(path_parent, 'resources', 'sample_dataframe_enwiki.xlsx', engine='openpyxl')
        dataframe_SVO = pd.read_excel(df_path, sheet_name='SVO_dataframe')
        dataframe_premodifier = pd.read_excel(df_path, sheet_name='premodifier_dataframe')
        dataframe_postmodifier = pd.read_excel(df_path, sheet_name='postmodifier_dataframe')
        dataframe_aux = pd.read_excel(df_path, sheet_name='aux_dataframe')
        dataframe_possess = pd.read_excel(df_path, sheet_name='possess_dataframe')
        dataframe_profession = pd.read_excel(df_path, sheet_name='profession_dataframe')
        dataframe_gender_count = pd.read_excel(df_path, sheet_name='gender_count_dataframe')
        dataframe_total = pd.read_excel(df_path, sheet_name='total_dataframe')

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

        input_dataframe_profession = dataframe_profession
        save_obj_user_uploads(input_dataframe_profession, name='profession_dataframe_user_uploads')

        input_dataframe_gender_count = dataframe_gender_count
        save_obj_user_uploads(input_dataframe_gender_count, name='gender_count_dataframe_user_uploads')

        return redirect(url_for('visualisation'))

        # except:
        #     flash('Sample file not found!', 'danger')
        #     return redirect(url_for('index'))


@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    path_parent = os.path.dirname(os.path.abspath(__file__))
    df_path = os.path.join(path_parent, 'static', 'user_uploads')
    # open dataframe file
    view_df = load_obj_user_uploads(df_path, name='total_dataframe_user_uploads')
    input_SVO_dataframe = load_obj_user_uploads(df_path, name='SVO_dataframe_user_uploads')
    input_premodifier_dataframe = load_obj_user_uploads(df_path, name='premodifier_dataframe_user_uploads')
    input_postmodifier_dataframe = load_obj_user_uploads(df_path, name='postmodifier_dataframe_user_uploads')
    input_aux_dataframe = load_obj_user_uploads(df_path, name='aux_dataframe_user_uploads')
    input_possess_dataframe = load_obj_user_uploads(df_path, name='possess_dataframe_user_uploads')
    input_profession_dataframe = load_obj_user_uploads(df_path, name='profession_dataframe_user_uploads')
    input_gender_count_dataframe = load_obj_user_uploads(df_path, name='gender_count_dataframe_user_uploads')

    female_tot_df, male_tot_df = gender_dataframe_from_tuple(view_df)
    tot_df = df_tot(female_tot_df, male_tot_df)
    tot_df = style_dataframe(tot_df, 'tot_df', ['token', 'pos'])

    female_noun_df, female_adj_df, female_verb_df = parse_pos_dataframe(view_df)[:3]
    male_noun_df, male_adj_df, male_verb_df = parse_pos_dataframe(view_df)[-3:]

    noun_df, adj_df, verb_df = df_tot(female_noun_df, male_noun_df), df_tot(female_adj_df, male_adj_df), df_tot(female_verb_df, male_verb_df)
    noun_df, adj_df, verb_df = style_dataframe(noun_df, 'noun_df', ['token', 'pos']), style_dataframe(adj_df, 'adj_df', ['token', 'pos']), style_dataframe(verb_df, 'verb_df', ['token', 'pos'])

    female_sub_df, female_obj_df, female_intran_df, male_sub_df, male_obj_df, male_intran_df = SVO_analysis(input_SVO_dataframe)

    sub_df, obj_df, intran_df = df_tot(female_sub_df, male_sub_df), df_tot(female_obj_df, male_obj_df), df_tot(female_intran_df, male_intran_df)
    sub_df, obj_df, intran_df = style_dataframe(sub_df, 'sub_df', ['verb', 'gender']), style_dataframe(obj_df, 'obj_df', ['verb', 'gender']), style_dataframe(intran_df, 'intran_df', ['verb', 'gender'])

    female_premodifier_df, male_premodifier_df = premodifier_analysis(input_premodifier_dataframe)
    premodifier_df = df_tot(female_premodifier_df, male_premodifier_df)
    premodifier_df = style_dataframe(premodifier_df, 'premodifier_df', ['word', 'gender'])


    female_postmodifier_df, male_postmodifier_df = postmodifier_analysis(input_postmodifier_dataframe)
    postmodifier_df = df_tot(female_postmodifier_df, male_postmodifier_df)
    postmodifier_df = style_dataframe(postmodifier_df, 'postmodifier_df', ['word', 'gender'])


    female_before_aux_df, male_before_aux_df, female_follow_aux_df, male_follow_aux_df = aux_analysis(input_aux_dataframe)
    before_aux_df, follow_aux_df = df_tot(female_before_aux_df, male_before_aux_df), df_tot(female_follow_aux_df, male_follow_aux_df)
    before_aux_df, follow_aux_df = style_dataframe(before_aux_df, 'before_aux_df', ['word', 'gender']), style_dataframe(follow_aux_df, 'follow_aux_df', ['word', 'gender'])


    female_possessive_df, male_possessive_df, female_possessor_df, male_possessor_df = possess_analysis(input_possess_dataframe)
    possessive_df, possessor_df = df_tot(female_possessive_df, male_possessive_df), df_tot(female_possessor_df, male_possessor_df)
    possessive_df, possessor_df = style_dataframe(possessive_df, 'possessive_df', ['word', 'gender']), style_dataframe(possessor_df, 'possessor_df', ['word', 'gender'])

    female_profession_df, male_profession_df = profession_analysis(input_profession_dataframe)
    profession_df = df_tot(female_profession_df, male_profession_df)
    profession_df = style_dataframe(profession_df, 'profession_df', ['token', 'gender'])


    female_count, male_count = gender_count_analysis(input_gender_count_dataframe)


    return render_template('analysis.html', female_count=female_count, male_count=male_count, data_tot=tot_df,
                           data_noun=noun_df, data_profession=profession_df,
                           data_adj=adj_df,
                           data_verb=verb_df,
                           data_intran_verb=intran_df,
                           data_sub_verb=sub_df, data_obj_verb=obj_df,
                           data_premodifier=premodifier_df,
                           data_postmodifier=postmodifier_df,
                           data_before_aux=before_aux_df,
                           data_follow_aux=follow_aux_df,
                           data_possessive=possessive_df,
                           data_possessor=possessor_df)

@app.route('/query', methods=['GET', 'POST'])
def query():
    path_parent = os.path.dirname(os.path.abspath(__file__))
    df_path = os.path.join(path_parent, 'static', 'user_uploads')

    dataframe_to_display = None
    select_gender = None
    select_wordtype = None
    plot_bar = None
    try:
        if request.method == 'POST':
            # open dataframe file
            view_df = load_obj_user_uploads(df_path, name='total_dataframe_user_uploads')
            input_SVO_dataframe = load_obj_user_uploads(df_path, name='SVO_dataframe_user_uploads')
            input_premodifier_dataframe = load_obj_user_uploads(df_path, name='premodifier_dataframe_user_uploads')
            input_postmodifier_dataframe = load_obj_user_uploads(df_path, name='postmodifier_dataframe_user_uploads')
            input_aux_dataframe = load_obj_user_uploads(df_path, name='aux_dataframe_user_uploads')
            input_possess_dataframe = load_obj_user_uploads(df_path, name='possess_dataframe_user_uploads')
            input_profession_dataframe = load_obj_user_uploads(df_path, name='profession_dataframe_user_uploads')


            input_question = request.form['user_question']
            select_gender, select_wordtype, dataframe_to_display = analyse_question(input_question, view_df, input_SVO_dataframe, input_premodifier_dataframe,
                             input_postmodifier_dataframe, input_aux_dataframe, input_possess_dataframe, input_profession_dataframe)

            save_obj(dataframe_to_display, name='specific_df')
            plot_bar = specific_bar_graph(df_name='specific_df')


        return render_template('query.html', data_question=dataframe_to_display, gender_in_question=str(select_gender),
                               type_in_question=str(select_wordtype), bar_graph_specific=plot_bar)
    except:

        flash('Please enter a valid question!', 'danger')

        return redirect(url_for('analysis'))


@app.route('/debiase', methods=['GET', 'POST'])
def debiase():
    if request.method == 'POST':
        user_threshold = request.form['user_threshold']
        try:
            user_threshold = float(user_threshold)
            if 0 <= user_threshold and user_threshold <= 1:
                debiased_file(user_threshold)
                flash('You can download the debiased file now!', 'info')
            else:
                flash('Please enter a number between 0 and 1!', 'danger')
        except:
            flash('Please enter a valid number!', 'danger')

    return render_template('debiase.html')



@app.route('/download/<df_name>', methods=['GET', 'POST'])
def download(df_name):
    return send_from_directory(directory=app.config['DOWNLOAD_FOLDER'], filename=df_name, as_attachment=True)


@app.route('/download_debiased_file/<txt_name>', methods=['GET', 'POST'])
def download_debiased_file(txt_name):
    return send_from_directory(directory=app.config['DEBIAS_FOLDER'], filename=txt_name, as_attachment=True)


@app.route('/about')
def about():
    return render_template('about.html')


