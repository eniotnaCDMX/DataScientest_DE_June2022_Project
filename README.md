# Client Satisfaction Project

## Context

This "Wise Analyzer" app was developed by Marie Bonin and me as part of the June 2022 Data Engineer training at DataScientest. The objective of this application is to analyze the content of comments related to the financial company "Wise" on the TrustPilot website. It is therefore based on the data contained on TrustPilot, offers the user the possibility to add a comment themselves, sets up a sentiment analysis algorithm to try to understand the links between ratings and sentiments, and what makes a sentiment positive or negative.

More specifically, it meets several needs:

- Obtain statistics on the comments in the database.
- Model output and analysis of results.
- Possibility for the user to add their own comment to the database and find out which category their comment was classified in.
- Update the content of the application after adding the comment.

## Extraction des données

Data extraction is done through <B>web scraping</B>, using a program written in <B>Python</B> and stored in a <B>private</B> GitHub folder, executed every 8 hours using <B>GitHub Actions</B>. A copy of the code is available in this folder under the name of <B>Update_RDS.py</B>. This code retrieves data from the comments of the first page of Wise-related comments on TrustPilot and stores them in a <B>MySQL</B> database hosted on <B>Amazon Web Services</B> in its <B>RDS</B> service.

## Structure de l'application

The application is coded in <B>Python</B>. The framework used is <B>Dash</B>. It uses, among others, the <B>Pandas</B> library for data processing, <B>nltk</B> for the NLP part, and <B>plotly</B> for graphics.

The data is fully loaded into RAM (< 50000 comments) for faster processing.





