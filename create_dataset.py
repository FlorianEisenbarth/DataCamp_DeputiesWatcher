import pandas as pd
import numpy as np
import json
import os


def read_vote(filename):
    with open(filename, "r") as f:
        vote = json.load(f)

    # Info about the vote
    id_vote = vote["scrutin"]["uid"]
    objet_vote = vote["scrutin"]["objet"]["libelle"]
    demandeur_vote = vote["scrutin"]["demandeur"]["texte"]
    date_vote = vote["scrutin"]["dateScrutin"]
    jourseance_vote = vote["scrutin"]["quantiemeJourSeance"]

    # Info about voting procedure
    type_vote = vote["scrutin"]["typeVote"]
    type_vote_code = type_vote["codeTypeVote"]
    type_vote_libelle = type_vote["libelleTypeVote"]
    type_vote_majorite = type_vote["typeMajorite"]

    # Info about members positions
    scrutin = vote["scrutin"]["ventilationVotes"]["organe"]["groupes"][
        "groupe"
    ]
    scrutin_membres = []
    position_membres = []
    for organe in scrutin:
        decomptes = organe["vote"]["decompteNominatif"]
        for position in ["pours", "contres", "abstentions"]:
            if decomptes[position]:
                votants = decomptes[position]["votant"]
                for membre in votants:
                    if type(membre) == dict:
                        membre_acteurRef = membre["acteurRef"]
                        scrutin_membres.append(membre_acteurRef)
                        position_membres.append(position[:-1])

    df_vote = pd.DataFrame(
        {
            "vote_uid": [id_vote],
            "vote_objet": [objet_vote],
            "vote_data": [date_vote],
            "vote_jourSeance": [jourseance_vote],
            "vote_typeCode": [type_vote_code],
            "vote_typeName": [type_vote_libelle],
            "vote_typeMajorite": [type_vote_majorite],
        }
    )
    n_membres = len(scrutin_membres)
    df_votants = pd.DataFrame(
        {
            # "vote_uid": np.repeat(id_vote, n_membres),
            "membre_acteurRef": scrutin_membres,
            "membre_position": position_membres,
        }
    )
    return df_vote, df_votants


def read_all_votes(acteurs):
    all_vote_filenames = os.listdir("data/vote")
    all_votes = pd.DataFrame()
    all_partis = pd.DataFrame()
    for filename in all_vote_filenames:
        vote, votants = read_vote("data/vote/" + filename)
        # Group votants by parti
        votants = votants.merge(acteurs, on="membre_acteurRef")
        partis = pd.pivot_table(
            votants,
            values="membre_acteurRef",
            index="membre_parti",
            columns="membre_position",
            aggfunc="count",
            # fill_value=0,
        )
        partis = partis.reset_index().rename({"membre_parti": "parti"}, axis=1)
        partis["vote_uid"] = vote["vote_uid"].iloc[0]
        # Update
        if not all_votes.empty:
            all_votes = all_votes.append(vote)
        else:
            all_votes = vote
        if not all_partis.empty:
            all_partis = all_partis.append(partis)
        else:
            all_partis = partis
    # If no votes found, we say it counts as 0 votes
    all_partis.fillna(0, inplace=True)
    return all_votes, all_partis


def read_acteur(filename):
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


def read_all_acteurs():
    all_acteur_filenames = os.listdir("data/acteur")
    output = pd.DataFrame()
    for filename in all_acteur_filenames:
        acteur = read_acteur("data/acteur/" + filename)
        # Update
        if not output.empty:
            output = output.append(acteur)
        else:
            output = acteur
    return output


def read_info_acteurs():
    filename = "data/nosdeputes.fr_synthese_2020-11-21.csv"
    df = pd.read_csv(filename, sep=";")
    old_cols = [
        "id",
        "prenom",
        "nom_de_famille",
        "date_naissance",
        "sexe",
        "parti_ratt_financier",
    ]
    new_cols = [
        "custom_id",
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


def load_data():
    """Loads all data

    Returns:
        acteurs: df with info about party members
        votes: df with info about votes
        scrutins: df with party votes for each vote
    """
    try:
        acteurs = pd.read_csv("data/acteurs.csv")
    except:
        acteurs = read_all_acteurs()
        acteurs.to_csv("data/acteurs.csv")

    acteurs_info = read_info_acteurs()

    acteurs_merge = pd.merge(
        acteurs, acteurs_info, on=["membre_prenom", "membre_nom"]
    )

    try:
        votes = pd.read_csv("data/votes.csv")
        scrutins_parti = pd.read_csv("data/scrutins_parti.csv")
    except:
        votes, scrutins_parti = read_all_votes(acteurs_merge)
        votes.to_csv("data/votes.csv")
        scrutins_parti.to_csv("data/scrutins_parti.csv")

    return acteurs_merge, votes, scrutins_parti


if __name__ == "__main__":
    acteurs, votes, scrutins = load_data()