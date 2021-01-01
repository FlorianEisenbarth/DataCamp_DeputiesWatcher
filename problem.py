import pandas as pd
import json
import numpy as np
from dataclasses import dataclass
import os
from os.path import join, splitext
import unidecode



DATA_HOME = "data"


@dataclass
class Vote:
    ''' Base class containing all relevant basis information of the dataset
    '''
    id: str
    code_type_vote: str
    libelle_type_vote: str
    demandeur: str
    libelle: str
    nb_votants: int
    date: str           # en faire un datetime ce serait bien ; à regarder
    vote_counts: pd.DataFrame

    @classmethod
    def load_from_files(cls, id, data_home=DATA_HOME, train_or_test='train'):
        f_name = join(data_home, train_or_test, id)
        with open(f_name + ".json", "r") as f:
            vote_metadata = json.load(f)

        vote_counts = (pd.read_csv(f_name + ".csv", sep=",").
                       rename(columns={'Unnamed: 0': 'party'}).
                       # renommer la première colonne (partis)
                       set_index('party'))

        vote = cls(id=id,
                   code_type_vote=vote_metadata['code_type_vote'],
                   libelle_type_vote=vote_metadata['libelle_type_vote'],
                   demandeur=vote_metadata['demandeur'],
                   libelle=vote_metadata['libelle'],
                   nb_votants=vote_metadata['nb_votants'],
                   date=vote_metadata['date'],
                   vote_counts=vote_counts)

        return vote

    def to_X_y(self):
        ''' Transform a Vote object into an observation X of features (dictionnary)
            and a label y
        '''
        number_of_dpt_per_party = {party: sum(self.vote_counts.loc[party])
                                    for party in self.vote_counts.index}
        X = {'code_type_vote': self.code_type_vote,
             'libelle_type_vote': self.libelle_type_vote,
             'demandeur': self.demandeur,
             'libelle': self.libelle,
             'nb_votants': self.nb_votants,
             'date': self.date,
             'presence_per_party': number_of_dpt_per_party}
        
        vote_columns = self.vote_counts.columns
        y = {}
        for party in self.vote_counts.index:
            major_position = vote_columns[np.argmax(self.vote_counts.loc[party])]
            y[party] = 1. * (major_position == 'pours')

        return X, y

def _read_data(path, train_or_test='train'):
    ''' Return the features dataset X and the labels dataset y for either the train or the test
    '''
    directory = join(path, DATA_HOME, train_or_test)
    votes_names = os.listdir(directory)
    votes_names = [splitext(vote)[0] for vote in votes_names if vote.endswith('.json')]
    votes_names.sort(key=lambda name: int(splitext(name)[0][10:]))
    
    for i, f_name in enumerate(votes_names):
        vote = Vote.load_from_files(f_name, train_or_test=train_or_test)
        features, label = vote.to_X_y()
        if i == 0:
            X = pd.DataFrame(columns=[key for key in features.keys()])
            y = pd.DataFrame(columns=[key for key in label.keys()])
        X.loc[f_name] = features
        y.loc[f_name] = label

    # Add a column equal to the index
    X['vote_uid'] = X.index 

    return X, y

def get_actor_party_data():
    """
    Returns general information about deputies and parties.
    To be used for creating features.
    Returns:
        actors: pd.DataFrame with info about actors.
    """
    try:
        actors = pd.read_csv("data/acteurs.csv")
    except:
        actors = _read_all_actors()
        actors.to_csv("data/acteurs.csv")

    actors_info = _read_info_actors()
    actors["membre_fullname"] = actors.apply(
        lambda x: x["membre_prenom"] + " " + x["membre_nom"], axis=1
    )
    actors["slug"] = actors["membre_fullname"].apply(_normalize_txt)
    actors.drop(["membre_fullname"], axis=1, inplace=True)
    actors_info.drop(["membre_prenom", "membre_nom"], axis=1, inplace=True)
    actors_info["slug"] = actors_info["membre_fullname"].apply(_normalize_txt)
    actors_merge = pd.merge(actors, actors_info, on="slug")

    return actors_merge


def _read_info_actors():
    filename = "data/nosdeputes.fr_synthese_2020-11-21.csv"
    df = pd.read_csv(filename, sep=";")
    old_cols = [
        "id",
        "nom",
        "prenom",
        "nom_de_famille",
        "date_naissance",
        "sexe",
        "parti_ratt_financier",
    ]
    new_cols = [
        "custom_id",
        "membre_fullname",
        "membre_prenom",
        "membre_nom",
        "membre_birthDate",
        "membre_sex",
        "membre_parti",
    ]
    df.rename(
        dict(zip(old_cols, new_cols)),
        axis=1,
        inplace=True,
    )
    df = df[new_cols]
    return df


def _read_actor(filename):
    acteur = pd.read_csv(filename, sep=";")
    id = acteur["uid[1]"]
    civ = acteur["etatCivil[1]/ident[1]/civ[1]"]
    prenom = acteur["etatCivil[1]/ident[1]/prenom[1]"]
    nom = acteur["etatCivil[1]/ident[1]/nom[1]"]
    output = pd.DataFrame(
        {
            "membre_acteurRef": id,
            "membre_civ": civ,
            "membre_prenom": prenom,
            "membre_nom": nom,
        }
    )
    return output


def _read_all_actors():
    all_acteur_filenames = os.listdir("data/acteur")
    output = pd.DataFrame()
    for filename in all_acteur_filenames:
        acteur = _read_actor("data/acteur/" + filename)
        # Update
        if not output.empty:
            output = output.append(acteur)
        else:
            output = acteur
    return output


def _normalize_txt(txt: str) -> str:
    """Remove accents and lowercase text."""
    if type(txt) == str:
        return unidecode.unidecode(txt).lower()
    else:
        return txt

def get_train_data(path='.'):
    return _read_data(path=path, train_or_test='train')

def get_test_data(path='.'):
    return _read_data(path=path, train_or_test='test')

