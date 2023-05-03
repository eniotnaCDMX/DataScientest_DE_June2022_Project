# Client Satisfaction Project

## Context

This "Wise Analyzer" app was developed by Marie Bonin and me as part of the June 2022 Data Engineer training at DataScientest. The objective of this application is to analyze the content of comments related to the financial company "Wise" on the TrustPilot website. It is therefore based on the data contained on TrustPilot, offers the user the possibility to add a comment themselves, sets up a sentiment analysis algorithm to try to understand the links between ratings and sentiments, and what makes a sentiment positive or negative.

More specifically, it meets several needs:

- Obtain statistics on the comments in the database.
- Model output and analysis of results.
- Possibility for the user to add their own comment to the database and find out which category their comment was classified in.
- Update the content of the application after adding the comment.

## Data architecture

We can represent the architecture of our app as the folowing :

![GitHub Logo](/Architecture.png)
Format: ![Alt Text](url)

## Data Extraction

Data extraction is done through <B>web scraping</B>, using a program written in <B>Python</B> and stored in a <B>private</B> GitHub folder, executed every 8 hours using <B>GitHub Actions</B>. A copy of the code is available in this folder under the name of <B>Update_RDS.py</B>. This code retrieves data from the comments of the first page of Wise-related comments on TrustPilot and stores them in a <B>MySQL</B> database hosted on <B>Amazon Web Services</B> in its <B>RDS</B> service.

## Model Application

To make the sentiment classification, we chose to use the <B>Vader</B> model, available in <B>nltk</B>. After comparing various models available in nltk, we chose to use Vader because it seemed to achieve, generally, a better performance.

When lauching our app, data is fully loaded to RAM (kept in a Pandas DataFrame) from RDS for faster processing (data fits in RAM with no problem). We then applied to it the Vader model and add a "sentiment" column, which is the output of the Vader model.

## Analysis Application Structure

The application is coded in <B>Python</B>. The framework used to build the app is <B>Dash</B> and uses plotly for every plot. The output of the Vader model is analyzed in python and the result of the analysis is available on the first page and the third page of the Dash app.

The goal of the analysis is, as we said earlier, to understand why a client gives a bad/good rate and if a bad/good rate always is a bad/good thing (that's why we lead a sentiment analysis), what we can do to make the service better. 

## Add a comment

The second page of our Dash app allows the user to add a comment to our database. When sending his comment, we apply the Vader model to it and he will be able to know immediately how his comment was classified.

The user will then be able to clic on an "Update Database" botton on the first page of the app to be able to actualize the statistics on page one and see how it changed.














