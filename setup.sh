#Ce fichier nécessaire à Heroku (fait manuellement et copié-collé selon la doc) permet de déployer l'app streamlit dans Heroku :
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
