import os
from vote_class import Vote
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl

'''
votes_names = os.listdir('votes')
votes_names = [os.path.splitext(x)[0] for x in votes_names]
votes_names.sort(key=lambda name: int(name[10:]))
max_votes = len(votes_names)

dpt_data = pd.read_csv('dpt_data/dpt_data.csv', sep=';').set_index('identifiant')
'''
with open('votes_data/vote_objects.pkl', 'rb') as f:
    votes = pkl.load(f)

vote1 = votes[0]
vote2 = votes[1]
vote3 = votes[2]
print('id :', vote2.id)
print('demandeur :', vote2.demandeur)
print('libelle :', vote2.libelle)
print('ratios :')
for parti, ratio in vote2.vote_ratios.items():
    print(parti, ':', ratio * 100, '% pour')