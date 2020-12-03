import pandas as pd
import json
import os
import matplotlib.pyplot as plt


def remove_special_chars(string):
    new_str = ''
    for char in string:
        if 'a' <= char <= 'z' or char == ' ':
            new_str += char 
    return new_str


list_dep = pd.read_csv('dpt_data/liste_deputes_excel.csv', sep=';')
dep_infos = pd.read_csv('dpt_data/nosdeputes.fr_synthese_2020-11-21.csv', sep=';')
list_dep = list_dep[
    ['identifiant', 
     'Prenom',
     'Nom',
     'Numéro de circonscription']
]
dep_infos = dep_infos[
    ['nom',
     'groupe_sigle',
     'parti_ratt_financier',
     'semaines_presence',
     'commission_presences',
     'amendements_signes',
     'amendements_adoptes',
     ]
]

list_dep['Prenom'] = list_dep['Prenom'].apply(str.lower)
list_dep['Nom'] = list_dep['Nom'].apply(str.lower)
dep_infos['nom'] = dep_infos['nom'].apply(str.lower)

list_dep['nom_id'] = (list_dep['Prenom'] + ' ' + list_dep['Nom']).apply(remove_special_chars)
dep_infos['nom_id'] = dep_infos['nom'].apply(remove_special_chars)
list_dep['identifiant'] = list_dep['identifiant'].apply(lambda x: 'PA' + str(x))

dpt_data = list_dep.merge(dep_infos, on='nom_id').set_index('identifiant')

dpt_data = dpt_data.drop(columns=['Prenom', 'Nom', 'nom_id'])
dpt_data.rename(
    columns={'Numéro de circonscription': 'circonscription',
             'parti_ratt_financier': 'parti',
             'groupe_sigle': 'parti_sigle'}, 
    inplace=True
)

with open('dpt_data/dpt_data.csv', 'w', encoding='utf8') as f:
    dpt_data.to_csv(f, line_terminator='\n')