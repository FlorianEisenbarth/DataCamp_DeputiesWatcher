
#%%
from sklearn import pipeline, preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
import create_dataset
import pandas as pd
import numpy as np
import re
import create_dataset



#%%
def get_train_data():

    acteurs, votes, scrutins = create_dataset.load_data()
    df = scrutins.merge(votes, on='vote_uid')
    df['vote_date'] = pd.to_datetime(df['vote_date'])
    df['vote_date'] = pd.to_datetime(df['vote_date'])
    df['position'] = (df['pour'] > df['contre']).astype(int)

    y = df['position']
    X = df[["parti","vote_objet", "vote_demandeur", 
            "vote_date", "vote_uid"]]

    # Encodage Parti
    # Encodage demandeur
    def find_parti_demandeur(txt):
        if type(txt) == str:
            groupe = re.findall('"(.*?)"', txt)
            # if len(groupe) > 1:
            # print(groupe)
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
    types_votes = ["l'amendement", "le sous-amendement", "l'article",
                   "l'ensemble du projet de loi", "l'ensemble de la proposition de loi", 
                   "la proposition de résolution", "l'ensemble de la proposition de résolution", 
                   "les crédits", "la motion référendaire", "la motion de renvoi en commission",
                   "la motion de rejet préalable", "la motion d'ajournement", "la motion de censure",
                   "la déclaration", "la première partie du projet de loi de finances", "la demande de"]

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
    
    
   
    return X, y
# %%


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



X, y = get_train_data()
X["target"] = y
X = X.explode("demandeur_parti")
y = X["target"]
X = X.drop("target", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

types_votes = ["l'amendement","amendements","le sous-amendement", "l'article",
                "l'ensemble du projet de loi", "l'ensemble de la proposition de loi", 
                "la proposition de résolution", "l'ensemble de la proposition de résolution", 
                "les crédits", "la motion référendaire", "la motion de renvoi en commission",
                "la motion de rejet préalable", "la motion d'ajournement", "la motion de censure",
                "la déclaration", "la première partie du projet de loi de finances", "la demande de"]


class Vote_objet_transformer(BaseEstimator, TransformerMixin):

    def __init__(self,types_votes, ):
        self.types_votes = types_votes

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()

        def del_vote_type(x):
            l = x.split()
            res = [words for words in l if words not in self.types_votes]
            return " ".join(res)
        X_["vote_objet"] = X_["vote_objet"].apply(del_vote_type)
        return X_

            

enc_vote_objet = Vote_objet_transformer(types_votes)

preprocessing = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="unknow"),
    OneHotEncoder()
        )

#simple vectorizer encoding (All the challenge is here !!)
vote_objet_encoding = make_pipeline(
    CountVectorizer(),
    TfidfTransformer()
)

transform = make_column_transformer(
    (preprocessing, ["parti","vote_demandeur","demandeur_parti"]),
    (vote_objet_encoding, "vote_objet"),
    ("drop",["vote_date","vote_uid","vote_objet"])
)

model = Pipeline([
    ("clean_vote_objet",Vote_objet_transformer(types_votes)),
    ("Preprocessing",transform),
    ("estimator",DecisionTreeClassifier(min_samples_leaf=10))
])

model.fit(X_train,y_train)
y_pred = model.predict(X_test)


# %%
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import f1_score

print("sklearn score :{} ".format(model.score(X_test,y_test)))
    
cm_df = pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    columns=['N', 'P'],
    index=['N', 'P']
)
sns.heatmap(cm_df, annot=True,
            cmap='Oranges',)
_ = plt.ylim(2, 0)
print( "f1_score: {}".format(f1_score(y_test, y_pred)))

plot_roc_curve(model,X_test,y_test)


# %%
