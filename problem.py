import pandas as pd
import json
import numpy as np
from dataclasses import dataclass
import os
from os.path import join, splitext



DATA_HOME = "data"


@dataclass
class Vote:
    ''' Base class containing all relevant basis information of the dataset
    '''
    id: str
    code_type_vote: str
    libelle_type_vote: str
    sort: str
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
                       ######## renommer la première colonne (partis)
                       set_index('party'))

        vote = cls(id=id,
                   code_type_vote=vote_metadata['code_type_vote'],
                   libelle_type_vote=vote_metadata['libelle_type_vote'],
                   sort=vote_metadata['sort'],
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
             'sort': self.sort,
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

    return X, y

def get_train_data(path='.'):
    return _read_data(path=path, train_or_test='train')

def get_test_data(path='.'):
    return _read_data(path=path, train_or_test='test')


X, y = get_test_data()
print(y)