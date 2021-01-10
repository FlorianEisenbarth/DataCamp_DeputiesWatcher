import pandas as pd
import json
import numpy as np
from dataclasses import dataclass
import os
from os.path import join, splitext
import unidecode
import pickle as pkl
import sys
from sklearn.model_selection import KFold

import rampwf
from rampwf.prediction_types.base import BasePrediction
from rampwf.score_types import BaseScoreType
from rampwf.workflows import Estimator


##### KEEP OFF ! DO NOT TOUCH !
PARTIES_SIGLES = ['SOC', 'FI', 'Dem', 'LT', 'GDR', 'LaREM', 'Agir ens', 'UDI-I', 'LR', 'NI']
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

# ----------
# score type
# ----------

def _custom_precision(party_pos_array_true, party_pos_array_pred):
    ''' Precision is the number of correct predictions divided by the number of predictions.
        In our problem, one prediction is a vector of binaries entries zeros and ones. Thus,
        a prediction set takes the form of a 2-dimensional array and the precision is
        computed column-wise (there is one precision value per party position).

        Returns:
            a vector of precisions values between 0.0 and 1.0
    '''
    assert party_pos_array_true.shape == party_pos_array_pred.shape, ('The '
            'true labels array and the prediction array should have same shape '
            'but have shape ' + str(party_pos_array_true.shape) + ' and shape '
            + str(party_pos_array_pred.shape) + ' respectively')
    
    if len(party_pos_array_pred) == 0:  # empty prediction
        return 0.0

    error_matrix = (party_pos_array_pred == party_pos_array_true)
    pred_number = len(party_pos_array_pred)
    return np.sum(error_matrix, axis=0) / pred_number

def _custom_recall(party_pos_array_true, party_pos_array_pred):
    ''' Here, recall is the number of correctly predicted 'pour' party major position divided
        by the actual number of 'pour' party major position (annoted as 1, versus 'not pour',
        annoted as 0). In our problem, one prediction is a vector of binaries entries zeros and
        ones. Thus, a prediction set takes the form of a 2-dimensional array and the recall is
        computed column-wise (there is one recall value per party position).

        Returns:
            a vector of recall values between 0.0 and 1.0        
    '''
    assert party_pos_array_true.shape == party_pos_array_pred.shape, ('The '
            'true labels array and the prediction arry should have same shape '
            'but have shape ' + str(party_pos_array_true.shape) + ' and shape '
            + str(party_pos_array_pred.shape) + ' respectively')

    if len(party_pos_array_pred) == 0:  # empty prediction
        return 0.0

    n_pours_per_party = np.sum(party_pos_array_true, axis=0)

    correct_pred_pour = np.zeros(party_pos_array_pred.shape[1])
    for i in range(party_pos_array_pred.shape[0]):
        pred = party_pos_array_pred[i]
        for j in range(party_pos_array_pred.shape[1]):
            if (party_pos_array_true[i,j] == 1) and (pred[j] == 1):
                correct_pred_pour[j] += 1

    recall = np.zeros(party_pos_array_pred.shape[1])
    for i in range(party_pos_array_pred.shape[1]):
        if n_pours_per_party[i] == 0:
            recall[i] = 1.
        else:
            recall[i] = correct_pred_pour[i] / n_pours_per_party[i]

    return recall

class CustomFScore(BaseScoreType):

    weights_type = 'log'    # whether to use mere proportion ('linear') 
                            # or log-proportion ('log') of deputies

    def __init__(self, name="F-score (party position detection)", precision=3):
        path = '\\'.join(os.path.abspath(__file__).split('\\')[:-1])
        self.name = name
        self.precision = precision
        self.weights = self.get_parties_weights(path=path, type=CustomFScore.weights_type)

    def __call__(self, y_true, y_pred) -> float:
        w = self.weights
        prec = _custom_precision(y_true, y_pred)
        rec = _custom_recall(y_true, y_pred)
        F_score = np.zeros(y_pred.shape[1])
        not_zero_idx = np.where(prec + rec != 0)
        F_score[not_zero_idx] = (
            2 * prec[not_zero_idx] * rec[not_zero_idx] / (prec[not_zero_idx] + rec[not_zero_idx])
        )
        return np.average(F_score, weights=w)
    
    def get_parties_weights(self, path, type='linear'):
        ''' Return the weights associated to each party. The default weight for a party
            (type='linear') is the mere proportion of deputies in the party among all the
            deputies. if type='log', the weight is passed through natural logartihm.
        '''
        file_name = join(path, 'dpt_data', 'liste_deputes_excel.csv')
        dpt_data = pd.read_csv(file_name, sep=';')
        groups_column_name = dpt_data.columns[-1]
        counts = dpt_data.groupby(groups_column_name).nunique()['identifiant'].to_dict()
        if type == 'linear':
            list_count = np.array([counts[key] for key in PARTIES_SIGLES])
        elif type == 'log':
            list_count = np.log(
                np.array([counts[key] for key in PARTIES_SIGLES])
            )
        else:
            raise ValueError('Unknown value for argument \'type\' :', type)
        weights = list_count / np.sum(list_count)

        return weights


# -----------------------
# A little bit of reading
# -----------------------


def _read_data(path, train_or_test='train', save=True):
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

    if save:
        file_name = join(path, DATA_HOME, train_or_test, train_or_test + '_data.pkl')
        with open(file_name, 'wb') as f:
            pkl.dump((X, y), f)

    return X, y

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

def _normalize_txt(txt: str) -> str:
    """Remove accents and lowercase text."""
    if type(txt) == str:
        return unidecode.unidecode(txt).lower()
    else:
        return txt


# -----------------------
# Ramp problem definition
# -----------------------


problem_title = "Deputy Watchers"
Predictions = rampwf.prediction_types.make_multiclass(label_names=PARTIES_SIGLES)
workflow = Estimator()
score_types = [CustomFScore()]


def get_train_data(path='.'):
    file_name = join(path, DATA_HOME, 'train', 'train_data.pkl')
    if os.path.isfile(file_name):
        with open(file_name, 'rb') as f:
            X, y = pkl.load(f)
        return X, y
    try:
        X, y = _read_data(path=path, train_or_test='train', save=True)
    except FileNotFoundError:
        print('Data files not created yet. Run \'create_files.py\' first.')
        sys.exit(0)

    return X, y

def get_test_data(path='.'):
    file_name = join(path, DATA_HOME, 'test', 'test_data.pkl')
    if os.path.isfile(file_name):
        with open(file_name, 'rb') as f:
            X, y = pkl.load(f)
        return X, y
    try:
        X, y = _read_data(path=path, train_or_test='test', save=True)
    except FileNotFoundError:
        print('Data files not created yet. Run \'create_files.py\' first.')
        sys.exit(0)
    
    return X, y

def get_cv(X, y):
    cv = KFold(n_splits=5)
    return cv.split(X, y)