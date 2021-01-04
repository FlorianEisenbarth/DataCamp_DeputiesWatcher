# %%

from problem import get_train_data, get_test_data, get_actor_party_data

import re, unidecode
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier


class FindGroupVoteDemandeurTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        """Through regex, find the groups mentioned in column vote_demandeur.
        There is often more than one group mentioned, so this returns a list.
        """
        pass

    def fit(self, X, y, **params):
        return self

    def transform(self, X, y=None, **params):
        X["demandeur_parti"] = X["demandeur"].apply(
            self.find_parti_demandeur
        )
        X['demandeur_parti'].fillna('[UNK]', inplace=True)
        X['demandeur_parti'] = X['demandeur_parti'].apply(lambda x: np.str_(x))
        return X

    def find_parti_demandeur(self, txt: str) -> list:
        def clean_groupe_name(txt: str) -> str:
            # TODO: rendre obsolète ce genre de remplacement
            # A mettre dans problem.py !
            txt = txt.strip()
            # Add missing accents
            txt = txt.replace("Les Republicains", "Les Républicains")
            txt = txt.replace(
                "democrate et republicaine", "démocrate et républicaine"
            )
            txt = txt.replace("Republique", "République")
            # Remove déterminants
            txt = txt.replace(
                "de la Gauche démocrate et républicaine",
                "Gauche démocrate et républicaine",
            )
            txt = txt.replace(
                "du Mouvement Démocrate et apparentés",
                "Mouvement Démocrate et apparentés",
            )
            # Add capital letter
            txt = txt.replace(
                "UDI, Agir et indépendants", "UDI, Agir et Indépendants"
            )
            # Remove non relevant text
            txt = txt.replace("President(e) du groupe", "")
            txt = txt.replace("\xa0", " ")
            return txt

        if type(txt) == str:
            groupe = re.findall('"(.*?)"', txt)
            # if len(groupe) > 1:
            # print(groupe)
        else:
            # NaN value for txt
            groupe = []
        groupe = [
            clean_groupe_name(name)
            for name in groupe
            if clean_groupe_name(name) != ""
        ]
        return groupe


class DecomposeVoteObjetTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        """In column libelle, there are several info.
        We use this tranformer to extract relevant ones :
        - libelle_type: the string with type of vote (look at self.types_votes)
        - libelle_desc: the string with a description of the object vote (ex: loi bioéthique)
        - libelle_auteur: the list of actors mentioned (ex: M. Melenchon)
        """
        # Liste établie "à la main"
        self.types_votes = [
            "l'amendement",
            "le sous-amendement",
            "l'article",
            "l'ensemble du projet de loi",
            "l'ensemble de la proposition de loi",
            "la proposition de résolution",
            "l'ensemble de la proposition de résolution",
            "les crédits",
            "la motion référendaire",
            "la motion de renvoi en commission",
            "la motion de rejet préalable",
            "la motion d'ajournement",
            "la motion de censure",
            "la déclaration",
            "la première partie du projet de loi de finances",
            "la demande de",
        ]
        # TODO: (peut-être ??)
        # Grouper : les motions
        # Grouper : amendement et sous-amendement
        # Grouper : l'ensemble du projet du loi, l'ensemble de la proposition de loi
        # Grouper : la proposition de résolution, l'ensemble de la proposition de résolution

    def fit(self, X, y, **params):
        return self

    def transform(self, X: pd.DataFrame, y=None, **params) -> pd.DataFrame:
        X["libelle_type"] = X["libelle"].apply(self.find_type_vote)
        X["libelle_desc"] = X["libelle"].apply(self.find_descriptif)
        X["libelle_auteur"] = X["libelle"].apply(self.find_auteur_loi)
        return X

    def find_type_vote(self, txt: str) -> str:
        self.weird_type_votes_ = []
        if type(txt) == str:
            type_vote = "[UNK]"
            # Fix common typos
            txt = txt.replace("le sous-amendment", "le sous-amendement")
            txt = txt.replace("declaration", "déclaration")
            txt = txt.replace(
                "la motion de renvoi en commision",
                "la motion de renvoi en commission",
            )
            txt = txt.replace("le demande de", "la demande de")
            for t in self.types_votes:
                if txt.startswith(t):
                    type_vote = t
                    break
            if type_vote == "[UNK]":
                self.weird_type_votes_.append(txt)
        else:
            type_vote = "[UNK]"
        return type_vote

    def find_descriptif(self, txt: str) -> str:
        self.weird_descriptifs_ = []
        if type(txt) == str:
            # Correct typo
            txt = txt.replace("  ", " ")
            descriptif = re.findall(
                "[le projet|du projet|de la proposition|au projet|à la proposition] de loi (.*?).?$",
                txt,
            )
            if descriptif != []:
                descriptif = descriptif[0]
            else:
                # Try another regex
                descriptif = re.findall(
                    "[la proposition de résolution|du Gouvernement sur|du Gouvernement relative|projet de loi, adopté par le Sénat,] (.*?).?$",
                    txt,
                )
                if descriptif != []:
                    descriptif = descriptif[0]
                else:
                    self.weird_descriptifs_.append(txt)
                    descriptif = "[UNK]"
        else:
            print(txt)
            descriptif = "[UNK]"
        return descriptif

    def find_auteur_loi(self, txt: str) -> list:
        """
        Returns a list of slugs of the auteurs found inside libelle.
        """
        self.weird_auteurs_loi_ = []

        def cut_at_stop_word(auteur):
            stop_words = ["et", "à", "après", "", "avant", "sur"]
            for i, l in enumerate(auteur):
                if l in stop_words:
                    auteur = auteur[:i]
                    break
            return auteur

        def merge_auteur(auteur):
            return " ".join([a.strip() for a in auteur])

        if type(txt) == str:
            # Match the two next word after "de M. XXX XXX"
            auteur = re.findall(
                "[du?e?|par] (Mm?e?\.?) ([A-Za-zÀ-ú]*) ([A-Za-zÀ-ú]*) ([A-Za-zÀ-ú]*)",
                txt,
            )
            if auteur != []:
                if len(auteur) == 1 or (
                    len(auteur) == 2 and txt.startswith("le sous-amend")
                ):
                    # In the case of sous-amendement, you may have 2 auteurs. Then, pick only the first auteur found.
                    auteur = auteur[0]
                    # Sometimes you have (first_name, last_name), sometimes (last_name, stop_word). We keep only last names.
                    auteur = [cut_at_stop_word(auteur)]
                else:
                    # You may have several auteurs for motions as well. In this case, keep all the auteurs.
                    auteur = [cut_at_stop_word(a) for a in auteur]
                auteur = [merge_auteur(a) for a in auteur]
            elif "du Gouvernement" in txt:
                auteur = ["Gouvernement"]
            else:
                auteur = ["Autre"]
                self.weird_auteurs_loi_.append(txt)
        else:
            # NaN value for txt
            auteur = ["Autre"]
        return auteur


class FindPartyActorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, actors: pd.DataFrame):
        """The column libelle_auteur is a list of peoples' names with irregularities.
        This transformer find the party of these people, as they are stored in the
        dataframe actors, accounting for those irregularities.

        Args:
            actors (pd.DataFrame): df with external info about deputies.
        """
        self.actors = self._create_slug_actors(actors.copy())

    def _normalize_txt(self, txt: str) -> str:
        """Remove accents and lowercase text."""
        if type(txt) == str:
            return unidecode.unidecode(txt).lower()
        else:
            return txt

    def _create_slug_actors(self, actors):
        """
        Create several slug columns for each actor.
        The slug is for example "m. melenchon" ou "m. jean-luc melenchon".
        """
        actors["slug_1"] = actors.apply(
            lambda x: x["membre_civ"] + " " + x["membre_nom"].replace("'", ""),
            axis=1,
        ).apply(self._normalize_txt)
        actors["slug_2"] = actors.apply(
            lambda x: x["membre_civ"] + " " + x["membre_fullname"], axis=1
        ).apply(self._normalize_txt)
        actors["slug_3"] = actors.apply(
            lambda x: x["membre_civ"]
            + " "
            + x["membre_prenom"]
            + " "
            + x["membre_nom"],
            axis=1,
        ).apply(self._normalize_txt)
        return actors

    def fit(self, X, y, **params):
        return self

    def transform(self, X, y=None, **params):
        X_ = X.copy()
        X_ = X_.explode("libelle_auteur")

        # Normalize libelle_auteur on specific autor names
        def replace_batch_auteur(list_of_s: list, replace: str):
            for s in list_of_s:
                X_["libelle_auteur"] = X_["libelle_auteur"].apply(
                    lambda x: x.replace(s, replace)
                )

        replace_batch_auteur(
            ["M. Édouard Philippe", "M. Edouard Philippe", "M. Jean Castex"],
            "Gouvernement",
        )
        replace_batch_auteur(["Mme x", "M. XXX"], "Anonyme")
        # Add a slug column by removing accents and setting it lowercase
        X_["slug"] = X_["libelle_auteur"].apply(self._normalize_txt)
        # Try to merge with self.actors on several version of the lusgs
        va_merge_1 = X_.merge(
            self.actors, how="inner", left_on="slug", right_on="slug_1"
        )
        va_merge_2 = X_.merge(
            self.actors, how="inner", left_on="slug", right_on="slug_2"
        )
        va_merge_3 = X_.merge(
            self.actors, how="inner", left_on="slug", right_on="slug_3"
        )
        # Special case with "Gouvernement", that is not in self.actors
        va_merge_4 = X_.merge(
            pd.DataFrame(
                {"slug": ["gouvernement"], "membre_parti": ["Gouvernement"]}
            ),
            on="slug",
        )
        # Merge all the joins together
        va_merge = (
            va_merge_1.append(va_merge_2).append(va_merge_3).append(va_merge_4)
        )
        va_merge.rename({"membre_parti": "auteur_parti"}, axis=1, inplace=True)
        # Reverse the explosion made over X, using a groupby.
        X_ = (
            va_merge.groupby("vote_uid")
            .agg({"auteur_parti": lambda x: x.tolist()})
            .reset_index()
        )
        # Drop non-relevant column
        X_ = X_[["vote_uid", "auteur_parti"]]
        # print(X_.head(5))
        # print(X.head(5))
        # Join with the original dataframe
        X = X.merge(X_, how="left", on="vote_uid")
        X['auteur_parti'].fillna("[NAN]", inplace=True)
        X['auteur_parti'] = X['auteur_parti'].apply(lambda x: np.str_(x))
        return X

class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()


# %%

from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.utils import class_weight


def get_estimator():
    # TODO : check mçi c'est ok de faire ça.
    # Si c'est pas ok, ajouter un fichier actors.csv au dossier de estimator.py
    actors = get_actor_party_data()  # Additional data about deputies

    find_group_vote_demandeur = FindGroupVoteDemandeurTransformer()
    decompose_vote_object = DecomposeVoteObjetTransformer()
    find_party_actor = FindPartyActorTransformer(actors)

    encode_category = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=["unknown"])
    )
    text_vectorizer = make_pipeline(CountVectorizer(), TfidfTransformer())
    idty = lambda x: x

    def encode_party_presence(x):
        y = x.iloc[:,0].apply(pd.Series)
        #y[y> 0] = 1
        return y

    vectorize_vote = make_column_transformer(
        (OneHotEncoder(), ["libelle_type"]),
        (
            CountVectorizer(binary=True, preprocessor=idty, tokenizer=idty),
            "demandeur_parti",
        ),
        (
            CountVectorizer(binary=True, preprocessor=idty, tokenizer=idty),
            "auteur_parti",
        ),
        (FunctionTransformer(func=encode_party_presence), ["presence_per_party"]),
        (text_vectorizer, "libelle_desc"),
    )

    #forest = LogisticRegression()
    #multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)

    def create_nn_model(): 
        nn = Sequential()
        nn.add(Dense(64, activation='relu', input_shape=(1161,)))
        nn.add(Dropout(0.2))
        nn.add(Dense(10, activation= "sigmoid"))
        nn.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=["accuracy"])
        return nn

    model = Pipeline(
        [
            ("find_group_vote_demandeur", find_group_vote_demandeur),
            ("decompose_vote_object", decompose_vote_object),
            ("find_party_actor", find_party_actor),
            ("vectorize_vote", vectorize_vote),
            ("densify", DenseTransformer()),
            ("nn", KerasClassifier(create_nn_model))
        ]
    )
    return model



# %%
model = get_estimator()

# %%
#X_train, y_train = get_train_data()
#X_test, y_test = get_test_data()

# %%
from sklearn.utils import class_weight

weights = np.mean(np.sum(y_train, axis=0))/np.sum(y_train, axis=0)
weights = dict(enumerate(weights))

# %%

model.fit(X_train, y_train.to_numpy(), 
    nn__batch_size=4096, 
    nn__epochs=500, 
    nn__class_weight=weights,
    nn__verbose=0)
model.score(X_test, y_test.to_numpy())
# %%
