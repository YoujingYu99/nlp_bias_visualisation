import os
from os import path
from flask import url_for
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.io as pio
from wordcloud import WordCloud
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from .functions_files import load_obj
from .functions_analysis import token_by_gender

def bar_graph(dataframe, token_list, value_list):
    # set minus sign
    df = dataframe

    # save file to static
    bar_name = 'bar_graph'
    save_img_path = path.join(os.path.dirname(os.path.abspath(__file__)), '..', "static", bar_name)

    bar_name_ex = bar_name + '.html'
    bar_path = save_img_path + '.html'
    fig = px.bar(df, x='token', y='bias', color='bias', color_continuous_scale=px.colors.sequential.Darkmint)
    fig.update_xaxes(title='words', visible=True, showticklabels=False)
    fig.update_yaxes(title='bias value', visible=True, showticklabels=False)
    pio.write_html(fig, file=bar_path, auto_open=False)
    plot_bar = url_for('static', filename=bar_name_ex)
    return plot_bar


def specific_bar_graph(df_name='specific_df'):
    path_parent = os.path.dirname(os.path.abspath(__file__))
    df_path = os.path.join(path_parent, '..', 'static')

    try:
        # set minus
        df = load_obj(df_path, name=df_name)

        # save file to static
        bar_name = 'specific_bar_graph'
        bar_name_ex = bar_name + '.html'
        save_img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static', bar_name)
        bar_path = save_img_path + '.html'


        fig = px.bar(df, x='token', y='bias', color='bias', orientation='h', color_continuous_scale=px.colors.sequential.Darkmint)
        fig.update_xaxes(title='words', visible=True, showticklabels=False)
        fig.update_yaxes(title='bias value', visible=True, showticklabels=False)
        pio.write_html(fig, file=bar_path, auto_open=False)
        plot_bar = url_for('static', filename=bar_name_ex)

        return plot_bar


    except:
        try:
            df = load_obj(df_path, name=df_name)

            # save file to static
            bar_name = 'specific_bar_graph'
            bar_name_ex = bar_name + '.html'
            save_img_path = path.join(os.path.dirname(os.path.abspath(__file__)), '..', "static", bar_name)
            bar_path = save_img_path + '.html'

            fig = px.bar(df, x='verb', y='Frequency', orientation='h', color='Frequency', color_continuous_scale=px.colors.sequential.Cividis_r)
            fig.update_xaxes(title='verb', visible=True, showticklabels=False)
            fig.update_yaxes(title='frequency', visible=True, showticklabels=False)
            pio.write_html(fig, file=bar_path, auto_open=False)
            plot_bar = url_for('static', filename=bar_name_ex)

            return plot_bar

        except:
            try:
                df = load_obj(df_path, name=df_name)
                # save file to static
                bar_name = 'specific_bar_graph'
                bar_name_ex = bar_name + '.html'
                save_img_path = path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static', bar_name)
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
    cloud_color = 'summer'
    cloud_bg_color = '#F0FFFF'



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
        save_img_path = path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static', female_cloud_name)
        img_path = save_img_path + '.png'
        female_wordcloud.to_file(img_path)
        female_wordcloud.to_file(img_path)


        plot_female_cloud = url_for('static', filename=female_cloud_name_ex)

    except:

        print('Not enough words for female cloud!')
        plot_female_cloud = url_for('static', filename='nothing_here.jpg')

    try:
        male_wordcloud.generate_from_frequencies(male_data)

        # save file to static
        male_cloud_name = 'malecloud'
        male_cloud_name_ex = male_cloud_name + '.png'
        save_img_path = path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static', male_cloud_name)
        img_path = save_img_path + '.png'
        male_wordcloud.to_file(img_path)


        plot_male_cloud = url_for('static', filename=male_cloud_name_ex)

    except:
        print('Not enough words for male cloud!')
        plot_male_cloud = url_for('static', filename='nothing_here.jpg')

    return plot_female_cloud, plot_male_cloud

# divide the dataframe into 10 animation frames
N = 10
def interactive_scatter(token_df, name, path, kind, markercolor):
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

    pio.templates.default = "ggplot2"

    fig = px.scatter(token_df, x='x', y='y', animation_frame='NO of words shown', text='word')

    fig.update_traces(textposition='top center', textfont_size=12)
    fig["layout"].pop("updatemenus")
    fig.update_xaxes(title='{} Latent Dimentsion 1'.format(kind), visible=True, showticklabels=True)
    fig.update_yaxes(title='{} Latent Dimentsion 2'.format(kind), visible=True, showticklabels=True)
    pio.write_html(fig, file=path, auto_play=False, auto_open=False)
    plot_scatter = url_for('static', filename=name)

    return plot_scatter


def tsne_graph(token_list, markercolor, iterations=3000, seed=20):
    """Creates a TSNE model and plots it"""

    # define word2vec model
    model_path = path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'resources','gum_word2vec.model')
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

    tsne_name = 'tsne'
    tsne_name_ex = tsne_name + '.html'
    save_img_path = path.join(os.path.dirname(os.path.abspath(__file__)), '..', "static", tsne_name)
    tsne_path = save_img_path + '.html'

    return interactive_scatter(word_df, tsne_name_ex, tsne_path, 'TSNE', markercolor)


def tsne_graph_male(token_list, value_list, markercolor, iterations=3000, seed=20):
    """Creates a TSNE model and plots it"""

    # define word2vec model
    model_path = path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'resources','gum_word2vec.model')
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

    tsne_name = 'tsne_male'
    tsne_name_ex = tsne_name + '.html'
    save_img_path = path.join(os.path.dirname(os.path.abspath(__file__)), '..', "static", tsne_name)
    tsne_path = save_img_path + '.html'

    return interactive_scatter(word_df, tsne_name_ex, tsne_path, 'TSNE', markercolor)


def tsne_graph_female(token_list, value_list, markercolor, iterations=3000, seed=20):
    """Creates a TSNE model and plots it"""

    # define word2vec model
    model_path = path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'resources','gum_word2vec.model')
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

    tsne_name = 'tsne_female'
    tsne_name_ex = tsne_name + '.html'
    save_img_path = path.join(os.path.dirname(os.path.abspath(__file__)), '..', "static", tsne_name)
    tsne_path = save_img_path + '.html'

    return interactive_scatter(word_df, tsne_name_ex, tsne_path, 'TSNE', markercolor)


def pca_graph(token_list, markercolor):
    """Creates a PCA model and plots it"""

    # define word2vec model
    model_path = path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'resources','gum_word2vec.model')
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
    pca_name = 'pca'
    pca_name_ex = pca_name + '.html'
    save_img_path = path.join(os.path.dirname(os.path.abspath(__file__)), '..', "static", pca_name)
    pca_path = save_img_path + '.html'

    return interactive_scatter(word_df, pca_name_ex, pca_path, 'PCA', markercolor)


def pca_graph_male(token_list, value_list, markercolor):
    """Creates a PCA model and plots it"""

    # define word2vec model
    model_path = path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'resources','gum_word2vec.model')
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
    pca_name = 'pca_male'
    pca_name_ex = pca_name + '.html'
    save_img_path = path.join(os.path.dirname(os.path.abspath(__file__)), '..', "static", pca_name)
    pca_path = save_img_path + '.html'

    return interactive_scatter(word_df, pca_name_ex, pca_path, 'PCA', markercolor)


def pca_graph_female(token_list, value_list, markercolor):
    """Creates a PCA model and plots it"""

    # define word2vec model
    model_path = path.join(os.path.dirname(os.path.abspath(__file__)), '..','resources','gum_word2vec.model')
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
    pca_name = 'pca_female'
    pca_name_ex = pca_name + '.html'
    save_img_path = path.join(os.path.dirname(os.path.abspath(__file__)), '..', "static", pca_name)
    pca_path = save_img_path + '.html'

    return interactive_scatter(word_df, pca_name_ex, pca_path, 'PCA', markercolor)


