#!/usr/bin/env python
#-*- coding: UTF-8 -*-
import numpy as np
import streamlit as st
import nltk
import pickle
import re
from bs4 import BeautifulSoup
from numpy import ceil, array
from sklearn.base import RegressorMixin
from pandas import read_pickle
import matplotlib.pyplot as plt
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
path = "examples/question_tagging/"


class SimpleModel(RegressorMixin):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        
    def fit(self, X, Y):
        if isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
            self.A_ = X.T.dot(Y)
        else:
            self.A_ = X.T * Y
        
    def predict(self, X):
        if isinstance(X, np.ndarray) and isinstance(self.A_, np.ndarray):
            continuous = X.dot(self.A_)
        else:
            continuous = X * self.A_
            
        normalized = (continuous.T / continuous.max(1)).T
        return normalized > self.threshold


@st.cache
def encode_decode(x):
    return str(x).encode("utf-8", errors='surrogatepass').decode("ISO-8859-1", errors='surrogatepass')


@st.cache
def clean_text(text):
    text = re.sub(r"\'", "'", text) # match all literal apostrophe pattern then replace them by a single whitespace
    text = re.sub(r"\n", " ", text) # match all literal Line Feed (New line) pattern then replace them by a single whitespace
    text = re.sub(r"\xa0", " ", text) # match all literal non-breakable space pattern then replace them by a single whitespace
    text = re.sub('\s+', ' ', text) # match all one or more whitespace then replace them by a single whitespace
    text = text.strip(' ')
    return text


@st.cache
def expand_contractions(s, contractions_dict, contractions_re):
    def replace(match):
        return contractions_dict[match.group(0)]

    return contractions_re.sub(replace, s)


@st.cache
def remove_single_letter(text):
    """remove single alphabetical character"""    

    rules = [
        "\. |\.$|\n|\t",                      # points de fin, retours à la ligne, tabulations
        "w*?\.(?!net)(\w*?)",                 # points entres mots fichier.truc exception .net
        " [-+#]+\w*?|\w*?[-]+ ",              # mots commençant par des +-# ou terminant par +-
        "__",                                 # caractères de fonctions cachées 
        "\\x80",
        "[ 0-9][-+\*/^_]+[ 0-9]",             # operations entre nombres
        " [0-9]*|\.[0-9]+",                   # nombres seuls ou après un point
        '[!?\'"`^~$%&@,;:(){}[\]\|<=>/\\\\]', # caractères spécifiques
        " . ",                                # caractère seul
        "^ | $",                              # espace de début et de fin
    ]
    for rule in rules *2:
        text = re.sub(rule, " ", text) 

    while "  " in text:
        text = text.replace("  ", " ")
    return text


@st.cache
def remove_stopwords(text, token, stop_words):
    """remove common words in english by using nltk.corpus's list"""
    words = token.tokenize(text)
    filtered = [w for w in words if not w in stop_words]

    return ' '.join(map(str, filtered)) # Return the text untokenize


@st.cache
def remove_by_tag(text, token, undesired_tag):
    """remove all words by using ntk tag (adjectives, verbs, etc.)"""
    words = token.tokenize(text)
    # Tag each words and return a list of tuples (e.g. ("have", "VB"))
    words_tagged = nltk.pos_tag(tokens=words, tagset=None, lang='eng')
    # Select all words that don't have the undesired tags
    filtered = [w[0] for w in words_tagged if w[1] not in undesired_tag]

    return ' '.join(map(str, filtered)) # Return the text untokenize


@st.cache
def stem_text(text, token, stemmer):
    """Stem the text"""
    words = token.tokenize(text.replace('-',''))  # Suppression de - pour les mots combinés "c-language"
    stem_text = []
    for word in words:
        stem_text.append(stemmer.stem(word)) # Stem each words
    return " ".join(stem_text) # Return the text untokenize


def load_model():
    with open(path+'model.pkl', 'rb') as file:
        model, multilabel_binarizer, tfidfX = pickle.load(file)
    with open(path+'contractions.pkl', 'rb') as file:
        contractions = pickle.load(file)
    return model, multilabel_binarizer, tfidfX, contractions

    
def grid_checkbox(values, nrow=None, ncol=None, title=None):
    N = len(values)
    
    if nrow == None:
        if ncol == None:
            return st.columns(N)
        else:
            nrow = int(ceil(N / ncol))
    elif ncol == None:
        ncol = int(ceil(N / nrow))
        
    cont = st.sidebar.container()
    if title is not None: 
        cont.text(title)
        
    grid = [cont.columns(ncol) for i in range(nrow)]
    i = 0
    checkboxes = []
    for row in range(nrow):
        for col in range(ncol):
            if i < N:
                checkboxes.append(grid[row][col].checkbox(values[i], key=values[i], value=True))
            i += 1
    return checkboxes

     
def model():
    # Visuel
    st.write("# Labélisation de question")
    title_input = st.text_input("Titre : ", value='Programming')
    question_input = st.text_area("Question : ", value='Python', height=300)
    
    # Outils
    model, multilabel_binarizer, tfidfX, contractions = load_model()    
    contractions_re = re.compile('(%s)' % '|'.join(contractions.keys()))
    stop_words = set(nltk.corpus.stopwords.words("english"))
    adjective_tag_list = set(['JJ','JJR', 'JJS', 'RBR', 'RBS'])
    token = nltk.tokenize.ToktokTokenizer()
    stemmer = nltk.stem.snowball.EnglishStemmer()
    # Traitements    
    x = ' + '.join([title_input, title_input, question_input])
    x = encode_decode(x)
    x = BeautifulSoup(x, 'html.parser').get_text()
    x = clean_text(x)
    x = expand_contractions(x, contractions, contractions_re)
    x = x.lower()
    x = remove_single_letter(x)
    x = remove_stopwords(x, token, stop_words)
    x = remove_by_tag(x, token, adjective_tag_list)
    x = stem_text(x, token, stemmer)

    # Tranformation / Prediction
    X = tfidfX.transform(np.array([x]))
    y = model.predict(X)
    output = multilabel_binarizer.inverse_transform(y)[0]

    # Visuel
    st.multiselect("Tags : ", multilabel_binarizer.classes_, output)


# @st.cache
def results():
    plt.style.use('dark_background')
    plt.rcParams.update({
        "figure.facecolor":  (0.0, 0.0, 0.0, 0.), 
        "axes.facecolor":    (0.0, 0.0, 0.0, 0.), 
        "savefig.facecolor": (0.0, 0.0, 0.0, 0.)})
    
    # Import et constantes
    df = read_pickle(path+'results.pkl')
    df.drop(columns=['Hamming L', 'Speed'], inplace=True)
    all_models = sorted(set(df.index.get_level_values(0).values))
    all_metrics = sorted(df.columns) 

    # Sidebar
    checkboxes1 = grid_checkbox(all_models, ncol=2, title='Modèles :')
    models = array(all_models)[checkboxes1]
    # models = st.sidebar.multiselect("Sélection de modèles :", all_models, all_models)
    checkboxes2 = grid_checkbox(all_metrics, ncol=2, title='Métriques :')
    metrics = array(all_metrics)[checkboxes2]
    # metrics = st.sidebar.multiselect("Sélection de métriques :", all_metrics, all_metrics)
    train_test = st.sidebar.radio('Jeu :', ['de Test','d\'Entrainement'])
    jeu = 'Test' if train_test == 'de Test' else'Train'
    st.sidebar.caption('Made by Woodian')

    # Visualisation
    st.write(f'# Résultats des différents modèles pour le jeu {train_test}')
    st.write((df.loc[models, jeu, :].droplevel(1).loc[:, metrics].sort_index() *100).astype(int).style.background_gradient())
    # axes = df.loc[(models, jeu),  metrics].droplevel(level=1, axis=0).plot.bar(figsize=(8,15), subplots=True)
    # for ax in axes:
    #     st.pyplot(ax.figure)
    st.pyplot(df.loc[(models, jeu),  metrics].droplevel(level=1, axis=0).plot.bar().figure)
    
    with st.container():
        col1, col2 = st.columns([1,7])
        desired_model = col1.selectbox('Modèle :', all_models, index=3)
        col2.write(df.loc[desired_model].loc[:, metrics].style.background_gradient(axis=1))
    

def main():
    menu = st.sidebar.radio('Partie', ['Modèle', 'Résultats'] , index=0)
    if menu == 'Modèle': model()
    elif menu == 'Résultats': results()
    
if __name__ == '__main__':
    path = './'
    main()