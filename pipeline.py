
#%%
from sklearn import pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import create_dataset
import pandas as pd
import numpy as np
import re
import create_dataset


class MyTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, X, y=None):
        pass

#%%
acteurs, votes, scrutins = create_dataset.load_data()
df = scrutins.merge(votes, on='vote_uid')
df['vote_date'] = pd.to_datetime(df['vote_date'])

df['vote_date'] = pd.to_datetime(df['vote_date'])
df['position'] = (df['pour'] > df['contre']).astype(int)

y = df['position']
X =  df[["parti","abstention","pour","contre","vote_objet","vote_demandeur","vote_date","vote_uid"]]

# Encodage Parti

#Encodage demandeur
def find_parti_demandeur(txt):
    if type(txt) == str:
        groupe = re.findall('"(.*?)"', txt)
        #if len(groupe) > 1:
        #    print(groupe)
    else:
        # NaN value for txt
        groupe = []
    groupe = [clean_groupe_name(name) for name in groupe if clean_groupe_name(name) != ""]
    return groupe

def clean_groupe_name(txt):
    txt = txt.strip()
    # Add missing accents
    txt = txt.replace('Les Republicains', 'Les Républicains')
    txt = txt.replace("democrate et republicaine", "démocrate et républicaine")
    txt = txt.replace("Republique", "République")
    # Remove déterminants
    txt = txt.replace("de la Gauche démocrate et républicaine", "Gauche démocrate et républicaine")
    txt = txt.replace("du Mouvement Démocrate et apparentés", "Mouvement Démocrate et apparentés")
    # Add capital letter
    txt = txt.replace("UDI, Agir et indépendants", "UDI, Agir et Indépendants")
    # Remove non relevant text
    txt = txt.replace("President(e) du groupe", "")
    txt = txt.replace("\xa0", " ")
    
    return txt

X['demandeur_parti'] = X['vote_demandeur'].apply(find_parti_demandeur)


# Encodage de vote objet 
weird_type_votes = []

# Liste établie "à la main"
types_votes = ["l'amendement", "le sous-amendement", "l'article", "l'ensemble du projet de loi", "l'ensemble de la proposition de loi", "la proposition de résolution", "l'ensemble de la proposition de résolution", "les crédits", "la motion référendaire", "la motion de renvoi en commission", "la motion de rejet préalable", "la motion d'ajournement", "la motion de censure", "la déclaration", "la première partie du projet de loi de finances", "la demande de"]

# TODO: (peut-être ??)
# Grouper : les motions 
# Grouper : amendement et sous-amendement 
# Grouper : l'ensemble du projet du loi, l'ensemble de la proposition de loi
# Grouper : la proposition de résolution, l'ensemble de la proposition de résolution

def find_type_vote(txt: str) -> str:
    if type(txt) == str:
        type_vote = "?"
        # Fix common typos
        txt = txt.replace("le sous-amendment", "le sous-amendement")
        txt = txt.replace("declaration", "déclaration")
        txt = txt.replace("la motion de renvoi en commision", "la motion de renvoi en commission")
        txt = txt.replace("le demande de", "la demande de")
        for t in types_votes:
            if txt.startswith(t):
                type_vote = t
                break
        if type_vote == "?":
            weird_type_votes.append(txt)
    else:
        type_vote = "?"
    return type_vote



X['vote_objet_type'] = X['vote_objet'].apply(find_type_vote)
X = X[["parti","vote_objet","vote_demandeur","demandeur_parti","vote_objet_type"]]





# %%
