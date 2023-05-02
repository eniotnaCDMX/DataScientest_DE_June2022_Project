# DataScientest_DE_June2022_Project

## Context

This "Wise Analyzer" app was made by Marie Bonin and I, as part of the June 2022 Data Engineer formation in DataScientest. L'objectif de cette application est d'analyser le contenu des commentaires liés à l'entreprise financière "Wise" sur le site [TrustPilot](https://www.trustpilot.com/review/wise.com).

## Extraction des données

L'extraction des données se fait par <B>web scraping</B>, via un programme écrit en <B>Python</B> et stocké dans un dossier GitHub <B>privé</B>, étant exécuté toutes les 8 heures en utilisant <B>GitHub Actions</B>. Une copie du code est accessible dans ce dossier, sous le nom de <B>Update_RDS.py</B>. Ce code permet de récupérer les données des commentaires de la première page de commentaires liés à Wise sur TrustPilot, et de les stocker dans une base de données <B>MySQL</B> hébergé chez <B>Amazon Web Services</B> dans son service <B>RDS</B>.




