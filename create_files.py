import json
import os
from os.path import join, splitext
import pandas as pd
import numpy as np



def get_releveant_informations_from_json_vote(file_name):
    ''' Returns a dictionnary with all the information that shall be used in the Vote class,
        except the party votes that are processed in **another function**
    '''
    with open(file_name, 'r') as f:
        scrutin = json.load(f)['scrutin']
    dictionnary = {
        'id': scrutin['uid'],
        'code_type_vote': scrutin['typeVote']['codeTypeVote'],
        'libelle_type_vote': scrutin['typeVote']['libelleTypeVote'],
        'sort': scrutin['sort']['code'],
        'demandeur': str(scrutin['demandeur']['texte']).replace(',', '').replace('\r', ' '),
        'libelle': str(scrutin['titre']).replace(',', '').replace('\r', ' '),
        'nb_votants': scrutin['syntheseVote']['nombreVotants'],
        'date': scrutin['dateScrutin']
    }

    return dictionnary

def create_json_vote_files(directory='votes', output_directory='data'):
    ''' For each "vote" json file in the directory, create a json file in the output_directory
        containing relevant data for the Vote objects
    '''
    votes_names = os.listdir(directory)
    votes_names.sort(key=lambda name: int(splitext(name)[0][10:]))

    for f_name in votes_names:
        dictionnary = get_releveant_informations_from_json_vote(join(directory, f_name))
        with open(join(output_directory, f_name), 'w') as f:
            json.dump(dictionnary, f)

def get_vote_informations_from_json_vote(scrutin):
    ''' Extract the vote ('nonVotants', 'pours', 'contres' or 'abstentions') of each
        actor in the scrutin, which is a vote dictionnary contained in a json file
    '''
    vote_keys = ['nonVotants', 'pours', 'contres', 'abstentions']
    votes = dict()

    list_of_groups = scrutin['ventilationVotes']['organe']['groupes']['groupe']
    for group in list_of_groups:
        for key in vote_keys:
            if group['vote']['decompteNominatif'][key] is not None:
                voters = group['vote']['decompteNominatif'][key]['votant']
                if type(voters) == dict: voters = [voters]
                for voter in voters:
                    votes[voter['acteurRef']] = key

    return votes

def count_votes_per_party(file_name, input_actors_data='dpt_data/dpt_data.csv'):
    ''' Returns a DataFrame indexed by parties, and with columns corresponding to
        each type of vote ('nonVotants', 'pours', 'contres', 'abstentions'), for the 
        json file corresponding to file_name
    '''
    with open(file_name, 'r') as f:
        scrutin = json.load(f)['scrutin']

    vote_columns = ['nonVotants', 'pours', 'contres', 'abstentions', 'total_count']
    votes = get_vote_informations_from_json_vote(scrutin)
    dpt_data = pd.read_csv(input_actors_data, sep=';').set_index('identifiant')
    parties = set(dpt_data['parti_sigle'])
    parties_count_df = pd.DataFrame(
        np.zeros(shape=(len(parties), len(vote_columns)), dtype=int),
        columns=vote_columns,
        index=parties
    )

    for voter, vote in votes.items():
        if voter in dpt_data.index:
            voter_party = dpt_data.loc[voter, 'parti_sigle']
        else:           # the deputy is not in the dataframe ; we don't include him/her in the ratio
            continue
        parties_count_df.loc[voter_party, vote] += 1            
        parties_count_df.loc[voter_party, 'total_count'] += 1

    return parties_count_df

def create_vote_count_csv_files(directory='votes', output_directory='data'):
    ''' For each "vote" json file in the directory, create a csv file in the output_directory
        containing vote informations for each party (number of 'abstentions', 'pours'...)
    '''
    votes_names = os.listdir(directory)
    votes_names.sort(key=lambda name: int(splitext(name)[0][10:]))

    for f_name in votes_names:
        df = count_votes_per_party(directory + '/' + f_name)
        name_export = join(output_directory, splitext(f_name)[0]) + '.csv'
        df.to_csv(name_export)

if __name__ == "__main__":
    print('creating json files...')
    create_json_vote_files()
    print('creating csv files...')
    create_vote_count_csv_files()
    print('done.')

