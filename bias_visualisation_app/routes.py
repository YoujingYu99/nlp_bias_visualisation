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
from bias_visualisation_app.utils.functions import get_text_url, get_text_file, token_value_lists, bar_graph, cloud_image, tsne_graph
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


# NLP bias detection
# if environ.get("USE_PRECALCULATED_BIASES", "").upper() == "TRUE":
#     print("using precalculated biases")
#     calculator = PrecalculatedBiasCalculator()
# else:
#     calculator = PcaBiasCalculator()

calculator = PrecalculatedBiasCalculator()

neutral_words = [
    "is",
    "was",
    "who",
    "what",
    "where",
    "the",
    "it",
]


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
        objs = parse_sentence(input_data)
        results = []
        view_results = []
        for obj in objs:
            token_result = {
                "token": obj["text"],
                "bias": calculator.detect_bias(obj["text"]),
                "parts": [
                    {
                        #"whitespace": token.whitespace_,
                        "pos": token.pos_,
                        "dep": token.dep_,
                        "ent": token.ent_type_,
                        "skip": token.pos_
                                in ["AUX", "ADP", "PUNCT", "SPACE", "DET", "PART", "CCONJ"]
                                or len(token) < 2
                                or token.text.lower() in neutral_words,
                    }
                    for token in obj["tokens"]
                ],
            }
            results.append(token_result)
        #copy results and only keep the word and the bias value
        token_result2 = results.copy()
        for item in token_result2:
            if "parts" in item.keys():
                del item['parts']
            else:
                continue
            view_results.append(item)
        view_df = list_to_dataframe(view_results)


        #plot the graphs
        plot_bar = bar_graph(view_df)
        plot_female_cloud, plot_male_cloud = cloud_image(token_list, value_list)
        #only perform tsne plot if more than 100 tokens
        if len(token_list) > 100:
            plot_tsne = tsne_graph(token_list)
        else:
            plot_tsne = url_for('static', filename="nothing_here.jpg")

    return render_template('visualisation.html', ctext=input_data, bias_description=view_df, bar_graph=plot_bar, female_word_cloud=plot_female_cloud, male_word_cloud=plot_male_cloud, tsne_graph=plot_tsne)
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
        objs = parse_sentence(input_data)
        results = []
        view_results = []
        for obj in objs:
            token_result = {
                "token": obj["text"],
                "bias": calculator.detect_bias(obj["text"]),
                "parts": [
                    {
                        "whitespace": token.whitespace_,
                        "pos": token.pos_,
                        "dep": token.dep_,
                        "ent": token.ent_type_,
                        "skip": token.pos_
                                in ["AUX", "ADP", "PUNCT", "SPACE", "DET", "PART", "CCONJ"]
                                or len(token) < 2
                                or token.text.lower() in neutral_words,
                    }
                    for token in obj["tokens"]
                ],
            }
            results.append(token_result)
            # copy results and only keep the word and the bias value
            token_result2 = results.copy()
            for item in token_result2:
                if "parts" in item.keys():
                    del item['parts']
                else:
                    continue
                view_results.append(item)
            token_list, value_list = token_value_lists(view_results)

            # plot the graphs
            plot_bar = bar_graph(token_list, value_list)
            plot_female_cloud, plot_male_cloud = cloud_image(token_list, value_list)
            # only perform tsne plot if more than 100 tokens
            if len(token_list) > 100:
                plot_tsne = tsne_graph(token_list)
            else:
                plot_tsne = url_for('static', filename="nothing_here.jpg")

        return render_template('visualisation.html', ctext=input_data, bias_description=view_results,
                               bar_graph=plot_bar, female_word_cloud=plot_female_cloud, male_word_cloud=plot_male_cloud,
                               tsne_graph=plot_tsne)



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
        if len(input_data) > 5000:
            raise werkzeug.exceptions.BadRequest(
                "Input Paragraph must be at most 5000 characters long"
            )
        objs = parse_sentence(input_data)
        results = []
        view_results = []
        for obj in objs:
            token_result = {
                "token": obj["text"],
                "bias": calculator.detect_bias(obj["text"]),
                "parts": [
                    {
                        "whitespace": token.whitespace_,
                        "pos": token.pos_,
                        "dep": token.dep_,
                        "ent": token.ent_type_,
                        "skip": token.pos_
                                in ["AUX", "ADP", "PUNCT", "SPACE", "DET", "PART", "CCONJ"]
                                or len(token) < 2
                                or token.text.lower() in neutral_words,
                    }
                    for token in obj["tokens"]
                ],
            }
            results.append(token_result)
            # copy results and only keep the word and the bias value
            token_result2 = results.copy()
            for item in token_result2:
                if "parts" in item.keys():
                    del item['parts']
                else:
                    continue
                view_results.append(item)
            token_list, value_list = token_value_lists(view_results)

            # plot the graphs
            plot_bar = bar_graph(token_list, value_list)
            plot_female_cloud, plot_male_cloud = cloud_image(token_list, value_list)
            # only perform tsne plot if more than 100 tokens
            if len(token_list) > 100:
                plot_tsne = tsne_graph(token_list)
            else:
                plot_tsne = url_for('static', filename="nothing_here.jpg")

        return render_template('visualisation.html', ctext=input_data, bias_description=view_results,
                               bar_graph=plot_bar, female_word_cloud=plot_female_cloud, male_word_cloud=plot_male_cloud,
                               tsne_graph=plot_tsne)

# . It works by looking at differences between male and female word pairs
#       like "he" and "she", or "boy" and "girl", and then comparing the
#       differences between those words to other word vectors in the word2vec
#       dataset.

# >0: is male biased
# <0: is female biased


@app.route('/about')
def about():
    return render_template('about.html')
