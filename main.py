import os
from vote_class import Vote
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl




print('loading data...')
with open('votes_data/vote_objects.pkl', 'rb') as f:
    votes = pkl.load(f)
    print('data loaded.')
    print()

for i, vote in enumerate(votes[:50]):
    print('vote ' + str(i) + ' :', vote.libelle)
    print('demandeur :', vote.demandeur )
    print('sort :', vote.sort)
    print('sort :', vote.vote_ratios)
    print()
