"""
The interactive web interface for data bias visualisation
"""

from __future__ import unicode_literals

import sys
from flask import Flask, render_template, url_for, request, jsonify
from bias_visualisation_app import app
from os import environ
import os
from bias_visualisation_app.utils.parse_sentence import parse_sentence, textify_tokens
from bias_visualisation_app.utils.PcaBiasCalculator import PcaBiasCalculator
from bias_visualisation_app.utils.PrecalculatedBiasCalculator import PrecalculatedBiasCalculator
from bias_visualisation_app.utils.functions import get_text_url, get_text_file, generate_list, list_to_dataframe, generate_bias_values, bar_graph, cloud_image, tsne_graph, tsne_graph_male, tsne_graph_female, pca_graph, pca_graph_male, pca_graph_female, gender_dataframe_from_tuple
import werkzeug
import spacy
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable


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
#     parser = PlaintextParser.from_string(docx, Tokenizer("english"))
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
    return render_template('visualisation.html')


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



@app.route("/detect_text", methods=['GET', 'POST'])
def detect_text():
    if request.method == "POST":
        input_data = request.form['rawtext']
        # sentence = request.args.get("sentence")
        if not input_data:
            raise werkzeug.exceptions.BadRequest("You must provide a paragraph")
        if len(input_data) > 50000:
            raise werkzeug.exceptions.BadRequest(
                "Input Paragraph must be at most 500000 characters long"
            )
        view_results = generate_bias_values(input_data)[0]
        view_df = generate_bias_values(input_data)[1]
        token_list, value_list = generate_bias_values(input_data)[2]


        #plot the bar graphs and word clouds
        plot_bar = bar_graph(view_df, token_list, value_list)
        plot_female_cloud, plot_male_cloud = cloud_image(token_list, value_list)
        #only perform tsne plot if more than 100 tokens
        if len(token_list) > 100:
            plot_tsne = tsne_graph(token_list)
            plot_tsne_male = tsne_graph_male(token_list, value_list)
            plot_tsne_female = tsne_graph_female(token_list, value_list)
            plot_pca = pca_graph(token_list)
            plot_pca_male = pca_graph_male(token_list, value_list)
            plot_pca_female = pca_graph_female(token_list, value_list)
        else:
            plot_tsne = url_for('static', filename="nothing_here.png")
            plot_tsne_male = url_for('static', filename="nothing_here.png")
            plot_tsne_female = url_for('static', filename="nothing_here.png")
            plot_pca = url_for('static', filename="nothing_here.png")
            plot_pca_male = url_for('static', filename="nothing_here.png")
            plot_pca_female = url_for('static', filename="nothing_here.png")

    return render_template('visualisation.html', ctext=input_data, bias_description=view_results, bar_graph=plot_bar, female_word_cloud=plot_female_cloud, male_word_cloud=plot_male_cloud,tsne_graph=plot_tsne, male_tsne_graph=plot_tsne_male, female_tsne_graph=plot_tsne_female, pca_graph=plot_pca, male_pca_graph=plot_pca_male, female_pca_graph=plot_pca_female)
 #he is a nurse


@app.route("/detect_url", methods=['GET', 'POST'])
def detect_url():
    if request.method == "POST":
        raw_url = request.form['raw_url']
        input_data = get_text_url(raw_url)
        if not input_data:
            raise werkzeug.exceptions.BadRequest("You must provide a paragraph")
        if len(input_data) > 5000:
            raise werkzeug.exceptions.BadRequest(
                "Input Paragraph must be at most 5000 characters long"
            )
        view_results = generate_bias_values(input_data)[0]
        view_df = generate_bias_values(input_data)[1]
        token_list, value_list = generate_bias_values(input_data)[2]

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
            plot_tsne = url_for('static', filename="nothing_here.png")
            plot_tsne_male = url_for('static', filename="nothing_here.png")
            plot_tsne_female = url_for('static', filename="nothing_here.png")
            plot_pca = url_for('static', filename="nothing_here.png")
            plot_pca_male = url_for('static', filename="nothing_here.png")
            plot_pca_female = url_for('static', filename="nothing_here.png")

        return render_template('visualisation.html', ctext=input_data, bias_description=view_results,
                               bar_graph=plot_bar, female_word_cloud=plot_female_cloud, male_word_cloud=plot_male_cloud,
                               tsne_graph=plot_tsne, male_tsne_graph=plot_tsne_male, female_tsne_graph=plot_tsne_female,
                               pca_graph=plot_pca, male_pca_graph=plot_pca_male, female_pca_graph=plot_pca_female)


@app.route("/detect_corpora", methods=['GET', 'POST'])
def detect_corpora():
    if request.method == "POST":
        try:
            corpora_file = request.files['raw_file']
        except:
            print("error with this line!")
            print(sys.exc_info()[0])
        input_data = get_text_file(corpora_file)
        if not input_data:
            raise werkzeug.exceptions.BadRequest("You must provide a paragraph")
        if len(input_data) > 50000:
            raise werkzeug.exceptions.BadRequest(
                "Input Paragraph must be at most 500000 characters long"
            )
        view_results = generate_bias_values(input_data)[0]
        view_df = generate_bias_values(input_data)[1]
        token_list, value_list = generate_bias_values(input_data)[2]

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
            plot_tsne = url_for('static', filename="nothing_here.png")
            plot_tsne_male = url_for('static', filename="nothing_here.png")
            plot_tsne_female = url_for('static', filename="nothing_here.png")
            plot_pca = url_for('static', filename="nothing_here.png")
            plot_pca_male = url_for('static', filename="nothing_here.png")
            plot_pca_female = url_for('static', filename="nothing_here.png")

        return render_template('visualisation.html', ctext=input_data, bias_description=view_results,
                               bar_graph=plot_bar, female_word_cloud=plot_female_cloud, male_word_cloud=plot_male_cloud,
                               tsne_graph=plot_tsne, male_tsne_graph=plot_tsne_male, female_tsne_graph=plot_tsne_female,
                               pca_graph=plot_pca, male_pca_graph=plot_pca_male, female_pca_graph=plot_pca_female)

# . It works by looking at differences between male and female word pairs
#       like "he" and "she", or "boy" and "girl", and then comparing the
#       differences between those words to other word vectors in the word2vec
#       dataset.

# >0: is male biased
# <0: is female biased

@app.route('/query')
def query():
    female_dataframe_tot, male_dataframe_tot = gender_dataframe_from_tuple()
    return render_template('query.html', data_fm=female_dataframe_tot, data_m=male_dataframe_tot)

@app.route('/analyse_adj', methods=['GET', 'POST'])
def analyse_adj():
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        female_dataframe_tot, male_dataframe_tot = gender_dataframe_from_dict(m_dic, fm_dic)
        if "adjectives" in rawtext:
            if "female" in rawtext:
                female_adjs = female_adjs()
            elif "male" in rawtext:
                male_adjs = male_adjs()
            else:
                print("Please enter a valid question")

    return render_template('query.html', ctext=rawtext, data_fm=female_dataframe_tot, data_m=male_dataframe_tot)



















@app.route('/about')
def about():
    return render_template('about.html')
