# Client Satisfaction Project

## Context

This "Wise Analyzer" app was made by Marie Bonin and I, as part of the June 2022 Data Engineer formation in DataScientest. L'objectif de cette application est d'analyser le contenu des commentaires liés à l'entreprise financière "Wise" sur le site [TrustPilot](https://www.trustpilot.com/review/wise.com).

Plus précisemment, elle répond à plusieurs besoins :
- Obtentions de statistiques sur les commentaires.
- Résultat du modèle et analyse des résultats.
- Possibilité d'ajout par l'utilisateur de son propre commentaire.
- Actualisation des statistiques après ajout du commentaire.

## Extraction des données

L'extraction des données se fait par <B>web scraping</B>, via un programme écrit en <B>Python</B> et stocké dans un dossier GitHub <B>privé</B>, étant exécuté toutes les 8 heures en utilisant <B>GitHub Actions</B>. Une copie du code est accessible dans ce dossier, sous le nom de <B>Update_RDS.py</B>. Ce code permet de récupérer les données des commentaires de la première page de commentaires liés à Wise sur TrustPilot, et de les stocker dans une base de données <B>MySQL</B> hébergée chez <B>Amazon Web Services</B> dans son service <B>RDS</B>.

## Structure de l'application

L'application est codée en <B>Python</B>. Le framework utilisé est <B>Dash</B>. Elle utilise, entre autres, les bibliothèques <B>Pandas</B> pour le traitement des données, <B>nltk</B> pour la partie NLP et <B>plotly</B> pour les graphiques.

Les données sont entièrement chargées en RAM (< 50000 commentaires) pour un traitement plus rapide. L'application répond à plusieurs besoins :





