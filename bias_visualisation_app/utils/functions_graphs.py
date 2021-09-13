import os
import sys
import string
from os import path
from os import listdir
from io import open
from conllu import parse_incr
import csv
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import requests
import werkzeug
from werkzeug.utils import secure_filename
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from flask import url_for
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
from wordcloud import WordCloud
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from .functions_files import load_obj
from .functions_analysis import token_by_gender
from .parse_sentence import parse_sentence
from .PrecalculatedBiasCalculator import PrecalculatedBiasCalculator

def bar_graph(dataframe, token_list, value_list):
    # set minus sign
    df = dataframe

    # save file to static
    bar_name = token_list[0] + token_list[-2]
    save_img_path = path.join(path.dirname(__file__), "..", "static", bar_name)

    bar_name_ex = bar_name + '.html'
    bar_path = save_img_path + '.html'
    fig = px.bar(df, x='token', y='bias', color='bias', color_continuous_scale=px.colors.sequential.Cividis_r)
    fig.update_xaxes(title='words', visible=True, showticklabels=False)
    fig.update_yaxes(title='bias value', visible=True, showticklabels=False)
    pio.write_html(fig, file=bar_path, auto_open=False)
    plot_bar = url_for('static', filename=bar_name_ex)
    return plot_bar


def specific_bar_graph(df_name='specific_df'):
    # set minus sign
    try:
        # set minus
        path_parent = os.path.dirname(os.getcwd())
        df_path = os.path.join(path_parent, 'visualising_data_bias', 'bias_visualisation_app', 'static')
        # mpl.rcParams['axes.unicode_minus'] = False
        # np.random.seed(12345)
        df = load_obj(df_path, name=df_name)
        # set_x_tick = True

        # plt.style.use('ggplot')
        # plt.rcParams['font.family'] = ['sans-serif']
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # fig, ax = plt.subplots()
        #
        # # set up the colors
        # cmap = mpl.colors.LinearSegmentedColormap.from_list('green_to_red', ['darkgreen', 'darkred'])
        # df_mean = df.mean(axis=1)
        # norm = plt.Normalize(df_mean.min(), df_mean.max())
        # colors = cmap(norm(df_mean))

        # ax.barh(
        #     df['token'],
        #     df['bias'],
        #     yerr=df.std(axis=1) / np.sqrt(len(df.columns)),
        #     color=colors)
        # fig.colorbar(ScalarMappable(cmap=cmap))
        #
        # ax.set_title('Specific Word Bias', fontsize=12)
        # ax.set_xlabel('Bias Value')
        # ax.xaxis.set_visible(set_x_tick)
        #
        # ax.set_ylabel('Word')
        # plt.tight_layout()

        # save file to static
        bar_name = df['token'].iloc[0] + df['token'].iloc[-2]
        bar_name_ex = bar_name + '.html'
        save_img_path = os.path.join(path.dirname(__file__), '..', 'static', bar_name)
        bar_path = save_img_path + '.html'
        # plt.savefig(bar_path)

        fig = px.bar(df, x='token', y='bias', color='bias', orientation='h', color_continuous_scale=px.colors.sequential.Cividis_r)
        fig.update_xaxes(title='words', visible=True, showticklabels=False)
        fig.update_yaxes(title='bias value', visible=True, showticklabels=False)
        pio.write_html(fig, file=bar_path, auto_open=False)
        plot_bar = url_for('static', filename=bar_name_ex)

        return plot_bar


    except:
        try:
            # mpl.rcParams['axes.unicode_minus'] = False
            # np.random.seed(12345)
            df = load_obj(df_path, name=df_name)
            # set_x_tick = True
            #
            # plt.style.use('ggplot')
            # plt.rcParams['font.family'] = ['sans-serif']
            # plt.rcParams['font.sans-serif'] = ['SimHei']
            # fig, ax = plt.subplots()

            # set up the colors
            # cmap = mpl.colors.LinearSegmentedColormap.from_list('green_to_red', ['darkgreen', 'darkred'])
            # df_mean = df.mean(axis=1)
            # norm = plt.Normalize(df_mean.min(), df_mean.max())
            # colors = cmap(norm(df_mean))
            #
            # ax.barh(
            #     df['verb'],
            #     df['Frequency'],
            #     yerr=df.std(axis=1) / np.sqrt(len(df.columns)),
            #     color=colors)
            # fig.colorbar(ScalarMappable(cmap=cmap))
            #
            # ax.set_title('Specific Word Frequency', fontsize=12)
            # ax.set_xlabel('Frequency')
            # ax.xaxis.set_visible(set_x_tick)
            #
            # ax.set_ylabel('Word')
            # plt.tight_layout()

            # save file to static
            bar_name = df['verb'].iloc[0] + df['verb'].iloc[1]
            bar_name_ex = bar_name + '.html'
            save_img_path = path.join(path.dirname(__file__), "..", "static", bar_name)
            bar_path = save_img_path + '.html'

            fig = px.bar(df, x='verb', y='Frequency', orientation='h', color='Frequency', color_continuous_scale=px.colors.sequential.Cividis_r)
            fig.update_xaxes(title='verb', visible=True, showticklabels=False)
            fig.update_yaxes(title='frequency', visible=True, showticklabels=False)
            pio.write_html(fig, file=bar_path, auto_open=False)
            plot_bar = url_for('static', filename=bar_name_ex)

            return plot_bar

        except:
            try:
                # mpl.rcParams['axes.unicode_minus'] = False
                # np.random.seed(12345)
                df = load_obj(df_path, name=df_name)
                # set_x_tick = True
                #
                # plt.style.use('ggplot')
                # plt.rcParams['font.family'] = ['sans-serif']
                # plt.rcParams['font.sans-serif'] = ['SimHei']
                # fig, ax = plt.subplots()

                # set up the colors
                # cmap = mpl.colors.LinearSegmentedColormap.from_list('green_to_red', ['darkgreen', 'darkred'])
                # df_mean = df.mean(axis=1)
                # norm = plt.Normalize(df_mean.min(), df_mean.max())
                # colors = cmap(norm(df_mean))
                #
                # ax.barh(
                #     df['word'],
                #     df['Frequency'],
                #     yerr=df.std(axis=1) / np.sqrt(len(df.columns)),
                #     color=colors)
                # fig.colorbar(ScalarMappable(cmap=cmap))
                #
                # ax.set_title('Specific Word Frequency', fontsize=12)
                # ax.set_xlabel('Frequency')
                # ax.xaxis.set_visible(set_x_tick)
                #
                # ax.set_ylabel('Word')
                # plt.tight_layout()

                # save file to static
                bar_name = df['word'].iloc[0] + df['word'].iloc[1]
                bar_name_ex = bar_name + '.html'
                save_img_path = path.join(path.dirname(__file__), '..', 'static', bar_name)
                bar_path = save_img_path + '.html'

                fig = px.bar(df, x='word', y='Frequency', orientation='h', color='Frequency', color_continuous_scale=px.colors.sequential.Cividis_r)
                fig.update_xaxes(title='word', visible=True, showticklabels=False)
                fig.update_yaxes(title='frequency', visible=True, showticklabels=False)
                pio.write_html(fig, file=bar_path, auto_open=False)
                plot_bar = url_for('static', filename=bar_name_ex)

                return plot_bar

            except:

                print('Not enough words for Plotting a bar chart')
                plot_bar = url_for('static', filename='nothing_here.jpg')

                return plot_bar


# def bar_graph(token_list, value_list):
#     plt.rcParams['axes.unicode_minus'] = False
#
#
#     def autolable(rects):
#         for rect in rects:
#             height = rect.get_height()
#             if height >= 0:
#                 plt.text(rect.get_x() + rect.get_width() / 2.0 - 0.3, height + 0.02, '%.3f' % height)
#             else:
#                 plt.text(rect.get_x() + rect.get_width() / 2.0 - 0.3, height - 0.06, '%.3f' % height)
#                 plt.axhline(y=0, color='black')
#
#     # normalise
#     norm = plt.Normalize(-1, 1)
#     norm_values = norm(value_list)
#     map_vir = cm.get_cmap(name='viridis')
#     colors = map_vir(norm_values)
#     fig = plt.figure()
#     plt.subplot(111)
#     ax = plt.bar(token_list, value_list, width=0.5, color=colors, edgecolor='black')
#
#     sm = cm.ScalarMappable(cmap=map_vir, norm=norm)
#     sm.set_array([])
#     plt.colorbar(sm)
#     autolable(ax)
#
#     # save file to static
#     bar_name = token_list[0]
#     bar_name_ex = bar_name + '.png'
#     save_img_path = path.join(path.dirname(__file__), '..\\static\\', bar_name)
#     bar_path = save_img_path + '.png'
#     plt.savefig(bar_path)
#     plot_bar = url_for('static', filename=bar_name_ex)
#
#     return plot_bar


def transform_format(val):
    if val.any() == 0:
        return 255
    else:
        return val


def cloud_image(token_list, value_list):
    # data
    # to convert lists to dictionary
    data = dict(zip(token_list, value_list))
    data = {k: v or 0 for (k, v) in data.items()}

    # separate into male and female dictionaries
    male_data = {k: v for (k, v) in data.items() if v > 0}
    female_data = {k: v for (k, v) in data.items() if v < 0}

    # cloud
    cloud_color = 'magma'
    cloud_bg_color = 'white'
    # cloud_custom_font = False

    # transform mask
    # female_mask_path = path.join(path.dirname(__file__), '..\\static\\images', 'female_symbol.png')
    # male_mask_path = path.join(path.dirname(__file__), '..\\static\\images', 'male_symbol.png')
    #
    # female_cloud_mask = np.array(Image.open(female_mask_path))
    # male_cloud_mask = np.array(Image.open(male_mask_path))

    cloud_scale = 0.1
    cloud_horizontal = 1
    bigrams = True

    # Setting up wordcloud from previously set variables.
    female_wordcloud = WordCloud(collocations=bigrams, regexp=None,
                                 relative_scaling=cloud_scale, width=1000,
                                 height=500, background_color=cloud_bg_color, max_words=10000,
                                 contour_width=0,
                                 colormap=cloud_color)

    male_wordcloud = WordCloud(collocations=bigrams, regexp=None, relative_scaling=cloud_scale,
                               width=1000,
                               height=500, background_color=cloud_bg_color, max_words=10000,
                               contour_width=0,
                               colormap=cloud_color)

    try:
        female_wordcloud.generate_from_frequencies(female_data)

        # save file to static
        female_cloud_name = 'femalecloud'
        female_cloud_name_ex = female_cloud_name + '.png'
        save_img_path = path.join(path.dirname(__file__), '..', 'static', female_cloud_name)
        img_path = save_img_path + '.png'
        female_wordcloud.to_file(img_path)
        female_wordcloud.to_file(img_path)

        # female_cloud_name = 'femalecloud'
        # female_cloud_name_ex = female_cloud_name + '.html'
        # save_img_path = path.join(path.dirname(__file__), '..', 'static', female_cloud_name)
        # img_path = save_img_path + '.html'
        # with open(img_path, 'w') as f:
        #     f.write(female_wordcloud.to_html())

        plot_female_cloud = url_for('static', filename=female_cloud_name_ex)

    except:
        # https: // www.wattpad.com / 729617965 - there % 27s - nothing - here - 3
        # https://images-na.ssl-images-amazon.com/images/I/41wjfr0wSsL.png
        print('Not enough words for female cloud!')
        plot_female_cloud = url_for('static', filename='nothing_here.jpg')

    try:
        male_wordcloud.generate_from_frequencies(male_data)

        # save file to static
        male_cloud_name = 'malecloud'
        male_cloud_name_ex = male_cloud_name + '.png'
        save_img_path = path.join(path.dirname(__file__),  '..', 'static', male_cloud_name)
        img_path = save_img_path + '.png'
        male_wordcloud.to_file(img_path)

        # male_cloud_name = 'malecloud'
        # male_cloud_name_ex = male_cloud_name + '.html'
        # save_img_path = path.join(path.dirname(__file__), '..', 'static', male_cloud_name)
        # img_path = save_img_path + '.html'
        # with open(img_path, 'w') as f:
        #     f.write(male_wordcloud.to_html())

        plot_male_cloud = url_for('static', filename=male_cloud_name_ex)

    except:
        print('Not enough words for male cloud!')
        plot_male_cloud = url_for('static', filename='nothing_here.jpg')

    return plot_female_cloud, plot_male_cloud


N = 10 # divide the dataframe into 10 animation frames
import math
def interactive_scatter(token_df, name, path, kind):
    # we need to make the token_df have points divided into N animation frames
    # first we need to divide the df into 10 groups with NO 0-9
    n = len(token_df.index)
    groupsize = math.ceil(n/N)
    groupno = []
    for i in range(N):
        if i==N-1:
            if n%groupsize:
                grouplist = [N-1]*(n%groupsize)
            else:
                grouplist = [N-1]*groupsize
        else:
            grouplist = [i]*groupsize
        groupno.extend(grouplist)
    token_df['groupNO'] = groupno
    animationno = groupno
    for i in range(N-1): # except for the last group
        # for groupNO x, we need to duplicate N-1-x times
        df_try = token_df[token_df['groupNO'] == i]
        token_df = token_df.append([df_try]*(N-1-i))
        for x in range(i+1, N):
            animationno.extend([x]*groupsize)
    for y, no in enumerate(animationno):
        if no==N-1:
            animationno[y] = n
        else:
            animationno[y] = (no + 1) * groupsize

    token_df['NO of words shown'] = animationno

    fig = px.scatter(token_df, x='x', y='y', animation_frame='NO of words shown', text='word')
    fig.update_traces(marker={'size': 14, 'line': dict(width=2,
                                                       color='DarkSlateGrey')},
                      selector=dict(mode='markers'))
    fig.update_traces(textposition='top center', textfont_size=12)
    fig["layout"].pop("updatemenus")
    fig.update_xaxes(title='{} Latent Dimentsion 1'.format(kind), visible=True, showticklabels=True)
    fig.update_yaxes(title='{} Latent Dimentsion 2'.format(kind), visible=True, showticklabels=True)
    pio.write_html(fig, file=path, auto_open=False)
    plot_scatter = url_for('static', filename=name)

    return plot_scatter


def tsne_graph(token_list, iterations=3000, seed=20, title="TSNE Visualisation of Word-Vectors"):
    """Creates a TSNE model and plots it"""

    # define word2vec model
    model_path = path.join(path.dirname(__file__), '..','resources','gum_word2vec.model')
    print(model_path)
    w2vmodel = Word2Vec.load(model_path)

    # manually define which words we want to explore
    my_word_list = []
    my_word_vectors = []

    words_to_explore = token_list

    for i in words_to_explore:
        try:
            if my_word_list not in my_word_list:
                my_word_vectors.append(w2vmodel.wv[i])
                my_word_list.append(i)
        except KeyError:
            continue

    tsne_model = TSNE(perplexity=5, n_components=2, init='pca', n_iter=iterations,
                      random_state=seed)
    new_values = tsne_model.fit_transform(my_word_vectors)
    new_values = np.array(new_values)
    # new_values is a list of dots coordinates and we need to make it a dataframe with the word_name on it
    word_df = pd.DataFrame({'word': my_word_list, 'x': new_values[:, 0], 'y': new_values[:, 1]})

    tsne_name = token_list[0] + token_list[-2] + 'tsne'
    tsne_name_ex = tsne_name + '.html'
    save_img_path = path.join(path.dirname(__file__), "..", "static", tsne_name)
    tsne_path = save_img_path + '.html'

    return interactive_scatter(word_df, tsne_name_ex, tsne_path, 'TSNE')


def tsne_graph_male(token_list, value_list, iterations=3000, seed=20, title="TSNE Visualisation(Male)"):
    """Creates a TSNE model and plots it"""

    # define word2vec model
    model_path = path.join(path.dirname(__file__), '..','resources','gum_word2vec.model')
    w2vmodel = Word2Vec.load(model_path)

    # manually define which words we want to explore
    my_word_list = []
    my_word_vectors = []

    words_to_explore = token_by_gender(token_list, value_list)[1]

    for i in words_to_explore:
        try:
            if my_word_list not in my_word_list:
                my_word_vectors.append(w2vmodel.wv[i])
                my_word_list.append(i)
        except KeyError:
            continue

    tsne_model = TSNE(perplexity=5, n_components=2, init='pca', n_iter=iterations,
                      random_state=seed)
    new_values = tsne_model.fit_transform(my_word_vectors)
    word_df = pd.DataFrame({'word': my_word_list, 'x': new_values[:, 0], 'y': new_values[:, 1]})

    tsne_name = token_list[0] + token_list[-2] + 'tsne_male'
    tsne_name_ex = tsne_name + '.html'
    save_img_path = path.join(path.dirname(__file__), "..", "static", tsne_name)
    tsne_path = save_img_path + '.html'

    return interactive_scatter(word_df, tsne_name_ex, tsne_path, 'TSNE')


def tsne_graph_female(token_list, value_list, iterations=3000, seed=20, title="TSNE Visualisation (Female)"):
    """Creates a TSNE model and plots it"""

    # define word2vec model
    model_path = path.join(path.dirname(__file__), '..','resources','gum_word2vec.model')
    w2vmodel = Word2Vec.load(model_path)

    # manually define which words we want to explore
    my_word_list = []
    my_word_vectors = []

    words_to_explore = token_by_gender(token_list, value_list)[1]

    for i in words_to_explore:
        try:
            if my_word_list not in my_word_list:
                my_word_vectors.append(w2vmodel.wv[i])
                my_word_list.append(i)
        except KeyError:
            continue

    tsne_model = TSNE(perplexity=5, n_components=2, init='pca', n_iter=iterations,
                      random_state=seed)
    new_values = tsne_model.fit_transform(my_word_vectors)
    word_df = pd.DataFrame({'word': my_word_list, 'x': new_values[:, 0], 'y': new_values[:, 1]})

    tsne_name = token_list[0] + token_list[-2] + 'tsne_female'
    tsne_name_ex = tsne_name + '.html'
    save_img_path = path.join(path.dirname(__file__), "..", "static", tsne_name)
    tsne_path = save_img_path + '.html'

    return interactive_scatter(word_df, tsne_name_ex, tsne_path, 'TSNE')


def pca_graph(token_list, title="PCA Visualisation of Word-Vectors for Amalgum"):
    """Creates a PCA model and plots it"""

    # define word2vec model
    model_path = path.join(path.dirname(__file__), '..','resources','gum_word2vec.model')
    w2vmodel = Word2Vec.load(model_path)

    # manually define which words we want to explore
    my_word_list = []
    my_word_vectors = []

    words_to_explore = token_list

    for i in words_to_explore:
        try:
            if my_word_list not in my_word_list:
                my_word_vectors.append(w2vmodel.wv[i])
                my_word_list.append(i)
        except KeyError:
            continue

    pca_model = PCA(n_components=2, svd_solver='full')
    new_values = pca_model.fit_transform(my_word_vectors)
    word_df = pd.DataFrame({'word': my_word_list, 'x': new_values[:, 0], 'y': new_values[:, 1]})

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    # save file to static
    pca_name = token_list[0] + token_list[-2] + 'pca'
    pca_name_ex = pca_name + '.html'
    save_img_path = path.join(path.dirname(__file__), "..", "static", pca_name)
    pca_path = save_img_path + '.html'

    return interactive_scatter(word_df, pca_name_ex, pca_path, 'PCA')


def pca_graph_male(token_list, value_list, title="PCA Visualisation(Male)"):
    """Creates a PCA model and plots it"""

    # define word2vec model
    model_path = path.join(path.dirname(__file__), '..','resources','gum_word2vec.model')
    w2vmodel = Word2Vec.load(model_path)

    # manually define which words we want to explore
    my_word_list = []
    my_word_vectors = []

    words_to_explore = token_by_gender(token_list, value_list)[1]

    for i in words_to_explore:
        try:
            if my_word_list not in my_word_list:
                my_word_vectors.append(w2vmodel.wv[i])
                my_word_list.append(i)
        except KeyError:
            continue

    pca_model = PCA(n_components=2, svd_solver='full')
    new_values = pca_model.fit_transform(my_word_vectors)
    word_df = pd.DataFrame({'word': my_word_list, 'x': new_values[:, 0], 'y': new_values[:, 1]})

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    # save file to static
    pca_name = token_list[0] + token_list[-2] + 'pca_male'
    pca_name_ex = pca_name + '.html'
    save_img_path = path.join(path.dirname(__file__), "..", "static", pca_name)
    pca_path = save_img_path + '.html'

    return interactive_scatter(word_df, pca_name_ex, pca_path, 'PCA')


def pca_graph_female(token_list, value_list, title="PCA Visualisation(Female)"):
    """Creates a PCA model and plots it"""

    # define word2vec model
    model_path = path.join(path.dirname(__file__), '..','resources','gum_word2vec.model')
    w2vmodel = Word2Vec.load(model_path)

    # manually define which words we want to explore
    my_word_list = []
    my_word_vectors = []

    words_to_explore = token_by_gender(token_list, value_list)[1]

    for i in words_to_explore:
        try:
            if my_word_list not in my_word_list:
                my_word_vectors.append(w2vmodel.wv[i])
                my_word_list.append(i)
        except KeyError:
            continue

    pca_model = PCA(n_components=2, svd_solver='full')
    new_values = pca_model.fit_transform(my_word_vectors)
    word_df = pd.DataFrame({'word': my_word_list, 'x': new_values[:, 0], 'y': new_values[:, 1]})

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    # save file to static
    pca_name = token_list[0] + token_list[-2] + 'pca_female'
    pca_name_ex = pca_name + '.html'
    save_img_path = path.join(path.dirname(__file__), "..", "static", pca_name)
    pca_path = save_img_path + '.html'

    return interactive_scatter(word_df, pca_name_ex, pca_path, 'PCA')



# def df_based_on_question(select_wordtype, select_gender, view_df, input_SVO_dataframe, input_premodifier_dataframe,
#                          input_postmodifier_dataframe, input_aux_dataframe, input_possess_dataframe, input_profession_dataframe):
#     female_tot_df, male_tot_df = gender_dataframe_from_tuple(view_df)
#     female_noun_df, female_adj_df, female_verb_df = parse_pos_dataframe(view_df)[:3]
#     male_noun_df, male_adj_df, male_verb_df = parse_pos_dataframe(view_df)[-3:]
#     female_sub_df, female_obj_df, female_intran_df, male_sub_df, male_obj_df, male_intran_df = SVO_analysis(
#         input_SVO_dataframe)
#     female_premodifier_df, male_premodifier_df = premodifier_analysis(input_premodifier_dataframe)
#     female_postmodifier_df, male_postmodifier_df = postmodifier_analysis(input_postmodifier_dataframe)
#     female_before_aux_df, male_before_aux_df, female_follow_aux_df, male_follow_aux_df = aux_analysis(
#         input_aux_dataframe)
#     female_possessive_df, male_possessive_df, female_possessor_df, male_possessor_df = possess_analysis(
#         input_possess_dataframe)
#     female_profession_df, male_profession_df = profession_analysis(input_profession_dataframe)
#
#     if select_gender == 'female':
#         if select_wordtype == 'nouns':
#             return female_noun_df
#         if select_wordtype == 'adjectives':
#             return female_adj_df
#         if select_wordtype == 'intransitive_verbs':
#             return female_intran_df
#         if select_wordtype == 'subject_verbs':
#             return female_sub_df
#         if select_wordtype == 'object_verbs':
#             return female_obj_df
#         if select_wordtype == 'premodifiers':
#             return female_premodifier_df
#         if select_wordtype == 'postmodifiers':
#             return female_postmodifier_df
#         if select_wordtype == 'before_aux':
#             return female_before_aux_df
#         if select_wordtype == 'follow_aux':
#             return female_follow_aux_df
#         if select_wordtype == 'possessives':
#             return female_possessive_df
#         if select_wordtype == 'possessors':
#             return female_possessor_df
#         if select_wordtype == 'professions':
#             return female_profession_df
#         else:
#             raise werkzeug.exceptions.BadRequest(
#                 'Please recheck your question'
#             )
#     if select_gender == 'male':
#         if select_wordtype == 'nouns':
#             return male_noun_df
#         if select_wordtype == 'adjectives':
#             return male_adj_df
#         if select_wordtype == 'intransitive_verbs':
#             return male_intran_df
#         if select_wordtype == 'subject_verbs':
#             return male_sub_df
#         if select_wordtype == 'object_verbs':
#             return male_obj_df
#         if select_wordtype == 'premodifiers':
#             return male_premodifier_df
#         if select_wordtype == 'postmodifiers':
#             return male_postmodifier_df
#         if select_wordtype == 'before_aux':
#             return male_before_aux_df
#         if select_wordtype == 'follow_aux':
#             return male_follow_aux_df
#         if select_wordtype == 'possessives':
#             return male_possessive_df
#         if select_wordtype == 'possessors':
#             return male_possessor_df
#         if select_wordtype == 'professions':
#             return male_profession_df
#         else:
#             raise werkzeug.exceptions.BadRequest(
#                 'Please recheck your question'
#             )