import pandas as pd
import json
import numpy as np
from dataclasses import dataclass
import os
from os.path import join, splitext



DATA_HOME = "data"


@dataclass
class Vote:
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
    def load_from_files(cls, id, data_home=DATA_HOME):
        f_name = join(data_home, id)
        with open(f_name + ".json", "r") as f:
            vote_metadata = json.load(f)

        vote_counts = (pd.read_csv(f_name + ".csv", sep=","))
                       #rename({'Unnamed: 0': 'parties'}).
                       ######## renommer la première colonne (partis)
                       #set_index('parties'))

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


votes_names = os.listdir('data')
votes_names = [splitext(name)[0] for name in votes_names if name.endswith('.json')]
print(votes_names[0])
votes_names.sort(key=lambda name: int(splitext(name)[0][10:]))
for i, name in enumerate(votes_names[:100]):
    vote = Vote.load_from_files(name)
    print('vote', i, ':')
    print('demandeur :', vote.demandeur)
    print('libelle :', vote.libelle)
    print()
