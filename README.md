# Visualising_data_bias

## Goal
This project is aimed at creating a web-based interface for Natural Language Processing(NLP) text corpora that enables different kinds of bias to be revealed visually and interactively prior to the data being used to train systems. The concept of dynamic data statement will be incorporated in the interface. 

Interactive data visualisation is defined as the “use of computer-supported, interactive, visual representations of data to amplify cognition, or the acquisition and use of knowledge”. To date, there is little to no research into the benefits of implementing interactive visualisation into data statements, but ample research into the benefits of visualisation in other areas. Hence, our project aims to combine both NLP analysis and interactive user-based data visualisation, to help user obtain a better understanding of the biases of their dataset before the next stage of implementation.


The aim of the project is to be able to give an interactive and dynamic bias statement for any English text that the user wishes to analyse, irrespective of its genre or size. In this way, the user can get a basic understanding and take note of possible biases before they proceed onto the next stage of implementing the dataset for their own purposes.

## Data
The data used in this project is the Amalgum dataset(https://github.com/gucorpling/amalgum). AMALGUM is a machine annotated multilayer corpus following the same design and annotation layers as GUM, but substantially larger (around 4M tokens). The goal of this corpus is to close the gap between high quality, richly annotated, but small datasets, and the larger but shallowly annotated corpora that are often scraped from the Web.

## Installation
If you are running locally on your laptop in developer mode, please install download the spacy packages:

```python
python -m spacy download en_core_web_sm
```
You can also increase the size of the corpus to be analysed by increasing the limit nlp.max_length in functions_analysis.py and routes.py:

```python
nlp.max_length = 10**10
```
And setting length of input_data to a higher value in routes.py:
```python
if len(input_data) > 1200000:
    raise werkzeug.exceptions.BadRequest(
        'Input Paragraph must be at most 250000 words long'
    )
```

## Requirements

```python
plotly==5.1.0
flask_unittest==0.1.2
Flask_Caching==1.10.1
pivottablejs==0.9.0
selenium==3.141.0
requests==2.24.0
pandas==1.1.3
Flask_Cors==3.0.10
conllu==4.4
Flask==1.1.2
gensim==3.8.3
matplotlib==3.3.4
numpy==1.18.5
spacy==3.0.7
dtale==1.56.0
Werkzeug==1.0.1
wordcloud==1.8.1
nltk==3.6.2
beautifulsoup4==4.10.0
scikit_learn==0.24.2

```



## Bias Calculation
The bias score calculation is based on the paper,  Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings, and the algorithm is adapted from the work of Chanind at (https://github.com/chanind/word2vec-gender-bias-explorer). Each token is assigned a score between -1 and 1, where a more positive number indicates male-biased and the more negative number indicates female-biased. 
Another algorithm that we wrote is to identify specific sentence structures and word types associated with each gender. There are, in total, seven such structures that we bring to the user's attention: Subject-Verb-Object(SVO) pairs, intransitive verbs, auxiliary terms, possession, possessives, premodifiers and postmodifiers.
Both results are saved to an xlsx file that the user can download for their own analysis. Then they can upload them again for visualisation, query and debiasing.

## Results:

### Graphs
Both interactive graphs and pivot tables are provided. The graphs include bar graphs, word clouds, PCA graphs and TSNE graphs. The pivot tables show the phrases identified with each gender and their Part-of-Speech(POS) tagging. Examples are shown below:

### Query
The user is able to input a natural language question (e.g. 'What actions do women usually perform?') and the answer will be presented to them in a table and a bar graph.

### Debias
All tokens are labeled with a bias value and the total score for each sentence is calculated and normalised to -1 and 1. The more positive number indicates male-biased and the more negative number indicates female-biased. The user can input a threshold, where all sentences more biased than the number will be discarded and the user can download the debiased file.



## Contributors
This project is developed by Youjing YU and Xiaoqiao Hu. The web development is inspired from the project (https://github.com/Jcharis/NLP-Web-Apps/tree/master/Summaryzer_Text_Summarization_App) by JCharis.

## License
[MIT](https://choosealicense.com/licenses/mit/)
