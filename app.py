### PARTIE DATASET

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
from sklearn.neighbors import KDTree
import requests
import json
import Flask
from sklearn.ensemble import GradientBoostingClassifier
import shap
import streamlit.components.v1 as components
import plotly.express as px
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold


# Fct de chargement des infos clients (en réorganisant les colonnes et en convertissant certaines unités)
n_rows=1000
def transform_raw_data(path):

    raw_data = pd.read_csv(path, nrows=n_rows, index_col='SK_ID_CURR')

    raw_data['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
    raw_data = raw_data[raw_data['CODE_GENDER'] != 'XNA']
    raw_data = raw_data[raw_data['AMT_INCOME_TOTAL'] < 100000000]

    good_cols = ['CODE_GENDER', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
                 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'OCCUPATION_TYPE']
    infos = raw_data.loc[:,good_cols]

    infos['AGE'] = (infos['DAYS_BIRTH']/-365).astype(int)
    infos['YEARS EMPLOYED'] = round((infos['DAYS_EMPLOYED']/-365), 2)
    infos.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1, inplace=True)

    infos = infos[[ 'AGE', 'CODE_GENDER','NAME_FAMILY_STATUS',
                   'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE','YEARS EMPLOYED',
                   'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
                 ]]

    infos.columns = [ 'AGE', 'GENDER','FAMILY STATUS',
                   'EDUCATION TYPE', 'OCCUPATION TYPE','YEARS EMPLOYED',
                   'YEARLY INCOME', 'AMOUNT CREDIT', 'AMOUNT ANNUITY', 'GOODS PRICE',
                 ]
    return infos




# Fct de chargement du CSV nettoyé en enlevant la colonne 'TARGET'
def load_app_train_clean():
    app_train_clean = pd.read_csv('https://raw.githubusercontent.com/MohamedBouzidGit/web-app-pret/master/app_train_clean.csv', nrows=n_rows, index_col=0)
    return app_train_clean.drop('TARGET', axis=1)


# Appel à la fct de chargement du CSV infos clients (via données brutes)
infos = transform_raw_data('https://raw.githubusercontent.com/MohamedBouzidGit/web-app-pret/master/application_train.csv')

# Appel à la fct de chargement du CSV nettoyé
data_processed = load_app_train_clean()
data_processed.index = data_processed.index.astype(int) # homonogéise l'index pour les différents dataframes

# Conversion des critères journaliers en valeurs positives pour des graphiques plus cohérents
data_processed['DAYS_ID_PUBLISH'] = (data_processed['DAYS_ID_PUBLISH'] * -1)
data_processed['DAYS_REGISTRATION'] = (data_processed['DAYS_REGISTRATION'] * -1)


# Chargement d'un CSV avec données moyennes pour chaque target
moyennes_tmp = pd.read_csv('https://raw.githubusercontent.com/MohamedBouzidGit/web-app-pret/master/application_train.csv', index_col=0)
moyennes_tmp2 = moyennes_tmp[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'DAYS_EMPLOYED', 'DAYS_BIRTH']]
moyennes_tmp2 = moyennes_tmp2.fillna(moyennes_tmp2.mean())

# Il existe des anomalies de jours (> 365243), donc on retire ces valeurs
moyennes_tmp2['DAYS_EMPLOYED'] = moyennes_tmp2['DAYS_EMPLOYED'].replace({365243: np.nan})
moyennes_tmp2['DAYS_EMPLOYED'] = moyennes_tmp2['DAYS_EMPLOYED'].fillna(moyennes_tmp2['DAYS_EMPLOYED'].mean())

# On ajoute deux colonnes de pourcentage par rapport aux revenus et à l'ancienneté
moyennes_tmp2['CREDIT_INCOME_PERCENT'] = moyennes_tmp2['AMT_CREDIT'] / moyennes_tmp2['AMT_INCOME_TOTAL']
moyennes_tmp2['DAYS_EMPLOYED_PERCENT'] = (moyennes_tmp2['DAYS_EMPLOYED'] / -1) / (moyennes_tmp2['DAYS_BIRTH'] / -1)
moyennes_tmp2['CREDIT_TERM'] = moyennes_tmp2['AMT_ANNUITY'] / moyennes_tmp2['AMT_CREDIT']


# Puis on supprime les colonnes non nécessaires
moyennes_tmp2 = moyennes_tmp2.drop(columns = ['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'DAYS_EMPLOYED', 'DAYS_BIRTH', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'CREDIT_INCOME_PERCENT', 'DAYS_EMPLOYED_PERCENT'])

# Enfin, on fait les moyennes de ces colonnes pour chaque classe de target
moyennes = moyennes_tmp2.groupby(moyennes_tmp2['TARGET']).mean()


# IMPORT ALGO ENTRAÎNÉ
with open('lgbc.pickle', 'rb') as file :
    LGB = pickle.load(file)


# Définition d'un df "voisins" 'df_vois' encodant les 6 premières colonnes du des infos client (infos.iloc[:,:6])
# L'encodage est fait car on a des colonnes de type classes et que l'on veut utiliser KNN dessus
df_vois = pd.get_dummies(infos.iloc[:,:6])
df_vois = df_vois.fillna(df_vois.mean())


# Définition d'une var 'tree' qui fait un KNN type 'KDTree' sur 'df_vois' (la structure de df_vois reste inchangée)
# On utilise tree pour la suite (faire une 'query' pour connaître la distance entre les plus proches voisins)
tree = KDTree(df_vois)


## PARTIE STATISTIQUES CLIENTS

# Streamlit : ajoute un titre principal
st.title('PRET A DEPENSER - Dashboard clients')
# Streamlit : affichage d'un titre
st.header('1. Statistiques client :')

# Streamlit : variable sélectionnant le client (infos.index donne la réf client)
client_id = st.sidebar.selectbox('Select ID Client :', infos.index)

# Streamlit : affiche une barre latérale avec les infos clients (via infos)
st.sidebar.table(infos.loc[client_id][:6])


# Streamlit : définir un bar chart sur les revenus/crédit/mensualités/prix du bien demandé par le client/moyenne
# sélection des colonnes revenus/crédit/mensualités/prix du bien dans une var 'bar_cols'
bar_cols = infos.columns[6:10]

# ajoute dans 'infos' une ligne 'Moyenne clients' = la moyenne pour chaque colonne de 'bar_cols' (le reste = NaN)
infos.at['Moyenne clients', bar_cols] = infos.loc[:,bar_cols].mean()

# Plotly : go.Bar affiche les barres (ici les colonnes voulues et la moyenne des colonnes concernées)
fig = go.Figure(data=[
    go.Bar(name='Client sélectionné', x=bar_cols, y=infos.loc[client_id, bar_cols].values, marker_color='rgba(116, 246, 69, 0.61)'),
    go.Bar(name='Moyenne des clients', x=bar_cols, y=infos.loc['Moyenne clients', bar_cols].values, marker_color='rgba(69, 205, 246, 0.61)')
])

# Plotly : met à jour le graphique en groupant les colonnes voulues avec les moyennes concernées et les couleurs
fig.update_layout(title_text=f'Montants des revenus et du crédit demandé pour le client {client_id}', paper_bgcolor = 'rgba(247, 247, 247, 0.55)', template='ggplot2')

# Streamlit : affiche le graphique
st.plotly_chart(fig, use_container_width=True)



## PARTIE ESTIMATION DE L'ALGORITHME
# Streamlit : affichage d'un titre
st.header('2. Risque client :')

# Appel à la fct de chargement du CSV nettoyé en localisant la réf client (présent ici sous forme d'index)
# .loc[client_id:client_id] localise exactement la réf client souhaitée (<=> de "réf client X" à "ref client X")
data_client = data_processed.loc[client_id:client_id]

# et donc la var 'prediction_client' calcule un pourcentage (100*) de la proba (predict_proba) du
# client concerné (data_client) d'être en capacité de rembourser son prêt.
# [0][1] car 'prediction_client' fourni un array avec 2 valeurs dont la proba du client (en %) en dernier. Ainsi,
# en utilisant [0][1], on récupère la deuxième valeur, ici correspondant à la proba du client (en %).
prediction_client = 100*LGB.predict_proba(data_client)[0][1]


# Or, on veut comparer cette proba avec les clients similaires à celui sélectionné (pour le lui montrer)
# C'est là qu'on utilise le KDTree en faisant une query.
idx_vois = tree.query([df_vois.loc[client_id].fillna(0)], k=10)[1][0]


# Ensuite, on selectionne les données via la fct de chargement du CSV nettoyé (data_processed) de ces
# voisins les plus proches (.iloc[idx_vois]) que l'on met dans une variable 'data_vois'
data_vois = data_processed.iloc[idx_vois]


# Et rebelote, on entre dans une varibale 'prediction_voisins' la proba d'estimation pour
# les voisins les plus proches (data_vois) du client concerné
prediction_voisins = 100*LGB.predict_proba(data_vois).mean(axis=0)[1]



# Streamlit : définir un indicateur en forme de jauge pour la capacité du client à ne pas rembourser un prêt

# Plotly : go.Figure(go.Indicator()) fourni une échelle de couleurs à définir de 0% (vert) à 100% (rouge)
gauge = go.Figure(go.Indicator(

    # On veut une jauge (gauge) + le chiffre du score (number) + le delta (delta) "score du client / ses voisins"
    mode = "gauge+delta+number",

    # La valeur que l'on souhaite afficher en score est la proba 'prediction_client'
    value = prediction_client,
    domain = {'x': [0, 1], 'y': [0, 1]},
    gauge = {'axis': {'range': [None, 100]},
            'steps' : [
                {'range': [0, 25], 'color': "yellowgreen"},
                {'range': [25, 50], 'color': "khaki"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"},
                ],
            'threshold': {
            'line': {'color': 'rgba(69, 205, 246, 0.61)', 'width': 4},
            'thickness': 0.8,
            'value': prediction_client},

            'bar': {'color': 'rgba(69, 205, 246, 0.61)', 'thickness' : 0.8},
            },

    # La valeur que l'on souhaite afficher en delta est la proba 'prediction_voisins'
    delta = {'reference': prediction_voisins,
    'increasing': {'color': 'red'},
    'decreasing' : {'color' : 'green'}}
    ))
# Plotly : permet de mettre à jour les couleurs et titres du graphique
gauge.update_layout(paper_bgcolor = 'rgba(247, 247, 247, 0.55)', font = {'color': "black"})


# Streamlit : on demande d'afficher un graphique potly (st.plotly_chart) qui a ci-dessus été appelé 'gauge'
st.plotly_chart(gauge)

# Streamlit : on demande d'afficher un texte markdown, ici le score du client en gras (**{0:.1f}%**)
st.markdown('Pour le client sélectionné : **{0:.1f}%**'.format(prediction_client))

# Streamlit : on demande d'afficher un texte markdown, ici le score des voisins en gras (**{0:.1f}%**)
st.markdown('Pour les clients similaires : **{0:.1f}%** (critères de similarité : âge, genre,\
     statut familial, éducation, profession, années d\'ancienneté)'.format(prediction_voisins))



## PARTIE INTERPRETATION DE L'ALGORITHME

# Streamlit : affichage d'un titre
st.header('3. Comparaison des critères du client :')


# Définition dans un dictionnaire 'feature_desc' des features impactant l'algorithme
feature_desc = {'EXT_SOURCE_1' : 'Score normalisé attribué par un organisme "1" indépendant',
		'EXT_SOURCE_2' : 'Score normalisé attribué par un organisme "2" indépendant',
                'EXT_SOURCE_3' : 'Score normalisé attribué par un organisme "3" indépendant',
                'CREDIT_TERM' : 'Durée de paiement en mois'}

# # Streamlit : affichage d'un filtre avec pour variable les noms de col que l'on retrouve dans le df 'moyennes'
feature = st.selectbox('Selectionnez la variable à comparer', moyennes.columns)


# Moyenne (.mean()) des features des plus proches voisins du client sélectionné (data_vois)

# On fait un df avec une colonne nommée 'voisins' qui calcule la moyenne (.mean()) de chaque colonne du
# df des voisins 'data_vois'. Ces colonnes passant en index, il faut transposer le résultat (.T)
mean_vois = pd.DataFrame(data_vois.mean(), columns=['voisins']).T


# On fusionne dans un df 'dfcomp', les df 'moyennes', 'mean_vois', et 'data_client' pour l'afficher sur streamlit
dfcomp = pd.concat([moyennes, mean_vois, data_client], join='inner').round(2)


# Streamlit : définir un indicateur en forme de barres horizontales pour comparer les features importantes du
# client avec les autres types de clients similaires ou non


# Plotly : go.Figure() fourni un graphique dont la donnée (data=) correspond à
# un graphique en forme de barres (go.Bar())

fig2 = go.Figure(data=[go.Bar(

    # En x, on affiche les taux des caractéristiques du df 'dfcomp' mis en place précédemment
    x=dfcomp[feature],

    # En y, on affiche les colonnes du df 'dfcomp'
    y=['Moyenne des clients en règle ',
      'Moyenne des clients en défaut ',
      'Moyenne des clients similaires ',
      'Client Sélectionné '],

    # On attribue une couleur à chacune des variables précédentes à afficher
    marker_color=['rgba(133, 38, 242, 0.41)', 'rgba(116, 246, 69, 0.61)', 'rgba(69, 205, 246, 0.61)', 'rgba(242, 133, 38, 0.41)'],

    # On choisi une orientation horizontale ('h')
    orientation ='h'
)])


# Plotly : On met à jour le graphique en affichant les définitions des features du dictionnaire défini précédemment et les couleurs
fig2.update_layout(title_text=feature_desc[feature], paper_bgcolor = 'rgba(247, 247, 247, 0.55)', template='ggplot2')

# Streamlit : on demande d'afficher le graphique potly (st.plotly_chart) qui a ci-dessus été appelé 'fig2'
st.plotly_chart(fig2)



## INDICATEUR SHAP

st.header('4. Interprétation de la décision :')

# Explication de l'indicateur
st.markdown('''La situation du client (`base value`) est représentée en fonction du seuil de décision (`f(x)`) attribuant au client une classe **Client solvable** ou **Client en défaut**.''')
st.markdown('''Lorsque la situation du client est *à gauche* du seuil de décision, le client est considéré comme en défaut. À l'inverse, une position *à droite* du seuil de décision indique un client solvable.''')
st.markdown('''Les motifs de décisions sont représentés sur l'échelle par les caractéristiques du client ainsi que leurs valeurs.''')





# Preprocessing
app_train_clean = pd.read_csv('https://raw.githubusercontent.com/MohamedBouzidGit/web-app-pret/master/app_train_clean.csv', nrows=n_rows, index_col=0)
X = app_train_clean.drop(columns = ['TARGET'])
Y = app_train_clean['TARGET']
xtrain, xeval, ytrain, yeval = train_test_split(X, Y, train_size=0.8, random_state=6, stratify=Y)



# Import de SHAP
shap.initjs()

# Fonction perso pour afficher SHAP sur Streamlit
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


# Calcul des valeurs Shapley et de l'interprétabilité
explainer = shap.TreeExplainer(LGB, 
                               model_output = 'probability',
                               data = xtrain.sample(200),
                               feature_dependence='independent')

shap_values = explainer.shap_values(xeval, yeval)
    

# lier l'index au client_id
c_id_idx = data_processed.index.get_loc(client_id) 

# Fonction représentant le diagramme SHAP Force plot dont la classe réelle du client VS les estimations
def shap_plot(i):
    print(LGB.predict_proba(xeval.iloc[c_id_idx:c_id_idx+1]).round(2))
    print('True class :', yeval.iloc[c_id_idx])
    return(shap.force_plot(explainer.expected_value, shap_values[c_id_idx], xeval.iloc[c_id_idx]))
    

# Affichage du force plot pour le client_id concerné
st_shap(shap_plot(c_id_idx))



## SIMULATEUR DE MODIFICATION DE CRITÈRES

# Simulateur (via case à cocher)
simulation = st.checkbox('''SIMULER DE MEILLEURS CRITÈRES POUR UN FINANCEMENT ?''')

# Si la case est cochée
if simulation:
    # On affiche des jauges pour modifier le montant des critères impactant la décision
    EXT_SOURCE_1 = st.slider('EXT_SOURCE_1', float(0),float(0),float(1),0.1) # syntaxe : st.slider('x', valeur minimale, valeur maximale, incrémentations)
    EXT_SOURCE_2 = st.slider('EXT_SOURCE_2', float(0),float(0),float(1),0.1) 
    EXT_SOURCE_3 = st.slider('EXT_SOURCE_3', float(0),float(0),float(1),0.1)
    CREDIT_TERM = st.slider('CREDIT_TERM', float(0),float(0),float(0.15),0.01)
    


    # On clone le subset concernant les données clients d'origine
    data_client_sim = data_client.copy()

    # Et on modifie les données clone par l'output des jauges
    data_client_sim['EXT_SOURCE_1'] = EXT_SOURCE_1
    data_client_sim['EXT_SOURCE_2'] = EXT_SOURCE_2
    data_client_sim['EXT_SOURCE_3'] = EXT_SOURCE_3
    data_client_sim['CREDIT_TERM'] = CREDIT_TERM

    # Puis on utilise l'algorithme sur les nouveaux critères
    estimation_client_sim = 100*LGB.predict_proba(data_client_sim)[0][1]
    # Enfin, on affiche le tout dans une jauge
    # Plotly : go.Figure(go.Indicator()) fourni une échelle de couleurs à définir de 0% (vert) à 100% (rouge)
    gauge_sim = go.Figure(go.Indicator(

        # On veut une jauge (gauge) + le chiffre du score (number) + le delta (delta) "score du client / ses voisins"
        mode = "gauge+number",

        # La valeur que l'on souhaite afficher en score est la proba 'prediction_client'
        value = estimation_client_sim,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {'axis': {'range': [None, 100]},
                'steps' : [
                    {'range': [0, 25], 'color': "yellowgreen"},
                    {'range': [25, 50], 'color': "khaki"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"},
                    ],
                'threshold': {
                'line': {'color': 'rgba(69, 205, 246, 0.61)', 'width': 4},
                'thickness': 0.8,
                'value': estimation_client_sim},

                'bar': {'color': 'rgba(69, 205, 246, 0.61)', 'thickness' : 0.8},
                }
        ))
    # Plotly : permet de mettre à jour les couleurs et titres du graphique
    gauge_sim.update_layout(paper_bgcolor = 'rgba(247, 247, 247, 0.55)', font = {'color': "black"}, )

    # Streamlit : on demande d'afficher un graphique potly (st.plotly_chart) qui a ci-dessus été appelé 'gauge'
    st.plotly_chart(gauge_sim)
