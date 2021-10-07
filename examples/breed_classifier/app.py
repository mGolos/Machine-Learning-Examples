# Librairies
import urllib
import pickle
import cv2
import numpy as np
import pandas as pd
import keras as kr
import streamlit as st
import streamlit.components.v1 as components
from tensorflow.keras import applications
from PIL import Image
path = "examples/breed_classifier/"


def load_from_url(address):
    resp = urllib.request.urlopen(address)
    img = np.array(bytearray(resp.read()), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_image(image_file):
    img = Image.open(image_file)
    img = np.array(img)
    return img


def preprocessing_image(img, dim):
    '''Temps de traitement approximatif de 7ms.'''
    res_img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
    img = applications.mobilenet_v2.preprocess_input(res_img)
    return img


def procedure(img, model, label_encoder, dim):
    img = preprocessing_image(img, dim)
    prediction = model.predict(img[None])[0]
    return label_encoder.classes_[prediction.argmax()], prediction.max()


def model():
    # Paramètres
    sample_img_url = 'https://s2.qwant.com/thumbr/474x263/4/d/508f009a12d962d40317b593a2fbeb5727e8a03dfaae00579394cf43ef9e81/th.jpg?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.NkTudYarFapDC5xnxMv9lQHaEH%26pid%3DApi&q=0&b=1&p=0&a=0'
    model = kr.models.load_model(path+'MobileNetV2.h5')
    dim = model.input_shape[1:-1]
    with open(path+'label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file)
        
    # Visuel
    st.title("Détermination de la race d'un chien")
    c1, c2 = st.columns(2)
    address = c1.text_input("Entrez une url d'image :", value=sample_img_url)
    image_file = c2.file_uploader('ou chargez en une :')

    # Chargement et Procédure
    try:
        if image_file is not None:
            img = load_image(image_file)
        else:
            img = load_from_url(address)
    except:
        st.info('Veuillez charger une image ou entrer une adresse URL valide')
    
    st.image(img, use_column_width=True)   
    race, rate = procedure(img, model, label_encoder, dim)
    
    # Visuel
    if rate > 0.8:   st.success(f"Race prédite : {race} à {100*rate:.2f}%")
    elif rate > 0.4: st.warning(f"Race prédite : {race} à {100*rate:.2f}%")
    else:             st.error(f"Pas de solution pertinente (Précision<40%)")
    if rate > 0.4:
        with st.expander(f"Exemples de {race}..."):
            components.iframe(f'https://www.picsearch.com/index.cgi?q=dog+breed+{race.replace(" ","+").lower()}', height=900, scrolling=True)
            
    with st.expander("Races possibles..."):
        st.multiselect("", label_encoder.classes_, label_encoder.classes_)


def results():
    st.title("Résultats")
    tmp = pd.read_csv(path+'results.csv', index_col=0)
    breed_results, avgs = tmp.iloc[:-3], tmp.iloc[-2:]
    avgs.index.name= 'Moyennes'
    avgs.index = ['Moyennes macro', 'Moyennes pondérées']
    st.write('### Moyennes')
    st.write(avgs)
    st.write('### Races')
    st.table(breed_results.sort_values(by='Précision', ascending=False)
             .style.bar(['Précision'], align='mid', color='steelblue')
             .background_gradient(subset=['Rappel','F1-Score','Occurences']))
    
    
def main():
    menu = st.sidebar.radio('Partie', ['Modèle', 'Résultats'] , index=0)
    if menu == 'Modèle': model()
    elif menu == 'Résultats': results()
    
if __name__ == '__main__':
    path = './'
    main()