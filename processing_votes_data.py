import pandas as pd
import os
import json
import numpy as np


# ----------- load the votes data ------------

votes_names = os.listdir('votes')
votes_names.sort(key=lambda name: int(os.path.splitext(name[10:])[0]))
max_votes = len(votes_names)

list_scrutin = []
for f_name in votes_names[:max_votes]:
    with open('votes/' + f_name, 'r') as f:
        content = json.load(f)
    list_scrutin.append(content)

# ----------- extract interesting facts, fill a dataFrame -------------

vote_data = pd.DataFrame(columns=['id',
                                  'libelle',
                                  'demandeur',
                                  'sort',
                                  'nombre_votants',
                                  'libelle_type_vote',
                                  'code_type_vote']
)

for i in range(len(list_scrutin)):
    scrutin = list_scrutin[i]['scrutin']
    id = scrutin['uid']
    code_type_vote = scrutin['typeVote']['codeTypeVote']
    libelle_type_vote = scrutin['typeVote']['libelleTypeVote']
    sort = scrutin['sort']['code']
    demandeur = str(scrutin['demandeur']['texte']).replace(',', '').replace('\r', ' ')
    libelle = str(scrutin['titre']).replace(',', '').replace('\r', ' ')
    nbre_votants = scrutin['syntheseVote']['nombreVotants']

    line = {'id': id,
            'code_type_vote': code_type_vote,
            'libelle_type_vote': libelle_type_vote,
            'sort': sort,
            'demandeur': demandeur,
            'libelle': libelle,
            'nombre_votants': nbre_votants}

    vote_data.loc[i] = line

vote_data.set_index('id', inplace=True)

with open('votes_data/votes_data.csv', 'w', encoding='utf8') as f:
    vote_data.to_csv(f, sep=',', line_terminator='\n')
