import json
import os
import pandas as pd
import pickle as pkl


votes_names = os.listdir('votes')
votes_names = [os.path.splitext(x)[0] for x in votes_names]
votes_names.sort(key=lambda name: int(name[10:]))
max_votes = len(votes_names)

list_scrutin = []
for f_name in votes_names[:max_votes]:
    with open('votes/' + f_name + '.json', 'r') as f:
        content = json.load(f)
    list_scrutin.append(content)

votes_data = pd.read_csv('votes_data/votes_data.csv', sep=',').set_index('id')

class Vote:

    def __init__(self, scrutin_id):
        scrutin = votes_data.loc[scrutin_id]
        self.id = scrutin_id
        self.libelle = scrutin['libelle']
        self.nombre_votants = scrutin['nombre_votants']
    
    def set_demandeur(self):
        scrutin = votes_data.loc[self.id]
        demandeur = scrutin['demandeur']
        if demandeur is not None and demandeur.count('"') == 2:
            idx1 = demandeur.find('"')
            idx2 = demandeur.find('"', idx1 + 1)
        else:
            idx1 = -1
        
        if idx1 != -1:
            self.demandeur = demandeur[idx1+1:idx2]
        else:
            self.demandeur = None

    def set_voters(self):

        ''' For a given scrutin, returns the list of actors who voted
            'for', the list of actors who voted 'against', the list
            of those who abstained, and the list of non-voters.
        '''

        vote_keys = {'nonVotants': 'NV', 'pours': 'P', 'contres': 'C', 'abstentions': 'A'}
        votes = dict()

        try:
            ind = votes_names.index(self.id)
        except ValueError:
            print(self.id, 'is not a valid scrutin id (not found in the list)')
            raise KeyError
        
        scrutin = list_scrutin[ind]['scrutin']
        list_of_groups = scrutin['ventilationVotes']['organe']['groupes']['groupe']
        for group in list_of_groups:
            for key in vote_keys.keys():
                if group['vote']['decompteNominatif'][key] is not None:
                    voters = group['vote']['decompteNominatif'][key]['votant']
                    if type(voters) == dict: voters = [voters]
                    for voter in voters:
                        votes[voter['acteurRef']] = vote_keys[key]

        self.voters = votes
    
    def compute_vote_pour_ratios_per_party(self, dpt_data):
        ''' Returns a dictionnary containing the vote 'pour' ratio, along 
            with the total vote count, for each party to the vote given 
            in argument

            arguments:
             - dpt_data is a DataFrame formatted as dpt_data.csv

            returns :
             - a tuple of the form (total vote count, 'pour' votes ratio)
        '''

        voters = self.voters
        parties = set(dpt_data['parti'])

        votes_per_party = {party: [0,0] for party in parties}
        for voter, vote in voters.items():
            if voter in dpt_data.index:
                voter_party = dpt_data.loc[voter]['parti']
            else:           # the deputy is not in the dataframe ; we don't include him/her in the ratio
                continue
            votes_per_party[voter_party][0] += 1        # counts voters per party
            if vote == 'P':
                votes_per_party[voter_party][1] += 1        # counts 'pour' voters per party
            
        self.vote_ratios = {party: (count[0], round(count[1] / count[0], 2)) 
                                for party, count in votes_per_party.items() 
                                if count[0] != 0
                            }

dpt_data = pd.read_csv('dpt_data/dpt_data.csv', sep=';').set_index('identifiant')

votes = []
for scrutin_id in votes_names:
    vote = Vote(scrutin_id)
    vote.set_demandeur()
    vote.set_voters()
    vote.compute_vote_pour_ratios_per_party(dpt_data=dpt_data)
    votes.append(vote)

with open('votes_data/vote_objects.pkl', 'wb') as f:
    pkl.dump(votes, f)
