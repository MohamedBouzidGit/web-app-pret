#Fichier requis par Heroku, fait manuellement, composé comme ci-dessous :
#web = application web
#sh setup.sh = commande qui spécifie le type de setup (ici un fichier shell, i.e. un script)
#&& = passe à la commande suivante si la précédante est réussie
#streamlit run app.py = lance la commande pour lancer streamlit sur le serveur interne de Heroku
web: sh setup.sh && streamlit run app.py
