import pandas as pd
import numpy as np
import json
import os
import unidecode


def get_train_data():
    """Loads training data.

    Returns:
        votes: pd.DataFrame Features about the vote.
        resultats: pd.DataFrame Results of the vote for each party. This is what needs to be predicted.
    """
    # TODO: Train test split en amont !
    votes = pd.DataFrame(
        {
            "vote_uid": ["VTANR5L15V2827"],
            "vote_objet": [
                "l'amendement n° 2166 rectifié du Gouvernement et les amendements identiques suivants à l'article premier du projet de loi relatif à la bioéthique (deuxième lecture)."
            ],
            "vote_demandeur": ['Président du groupe "UDI et Indépendants"'],
        }
    )
    results = pd.DataFrame(
        {
            "vote_uid": [
                "VTANR5L15V2827",
                "VTANR5L15V2827",
                "VTANR5L15V2827",
                "VTANR5L15V2827",
                "VTANR5L15V2827",
                "VTANR5L15V2827",
                "VTANR5L15V2827",
                "VTANR5L15V2827",
                "VTANR5L15V2827",
            ],
            "parti": [
                "La France Insoumise",
                "La République en Marche",
                "Les Républicains",
                "Mouvement Démocrate",
                "Indépendant",
                "Parti communiste français",
                "Parti socialiste",
                "Rassemblement national",
                "Union des démocrates, radicaux et libéraux",
            ],
            "abstention": [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "contre": [2.0, 11.0, 2.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            "pour": [0.0, 41.0, 13.0, 6.0, 1.0, 2.0, 0.0, 1.0, 3.0],
        }
    )

    return votes, results


def get_test_data():
    """Loads test data."""
    votes = pd.DataFrame(columns=["vote_id", "features"])
    results = pd.DataFrame(columns=["parti", "result_vote"])
    return votes, results


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


def _normalize_txt(txt: str) -> str:
    """Remove accents and lowercase text."""
    if type(txt) == str:
        return unidecode.unidecode(txt).lower()
    else:
        return txt