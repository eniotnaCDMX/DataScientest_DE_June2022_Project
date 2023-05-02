#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# General libraries
import json
import re
import datetime

# Data analysis libraries
import pandas as pd
#disable the false positive warning 
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import pymysql
from textblob import TextBlob


# Web libraries
import requests
from bs4 import BeautifulSoup
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash import dash_table as dt
import plotly.express as px
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import geocoder


# Machine Learning libraries
import seaborn as sb
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


#Sentiment Analysis
import os
from langdetect import detect
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import string
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
nltk.downloader.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# AWS Database Credential

Instance_Identifier = ""
Port = 0
Password = ""
User = ""
Host = ""



class Work_Data():
    
    def Make_DF_And_Cleaning():
        def Get_AWS():
            Conn = pymysql.connect(host = Host, user = User, password = Password)
            Curs = Conn.cursor()
            SQL2 = '''USE Wise_Data'''
            Curs.execute(SQL2)
            DB = pd.read_sql_query('''SELECT * FROM Commentaires2''',Conn).sort_values("Date_of_Review",ascending = False)
            Curs.close()
            Conn.close()
            return DB.sort_values("Date_of_Review",ascending = False)
        df = Get_AWS()
        def clean(text):
            text = re.sub('https?://\S+|www\.\S+', '', text)
            text = re.sub(r'\s+', ' ', text, flags=re.I)
            text = re.sub('\[.*?\]', '', text)
            text = re.sub('\n', '', text)
            text = re.sub('\w*\d\w*', '', text)
            text = re.sub('<.*?>+', '', text)
            return text
        df['Content2'] = df['Content'].apply(lambda x:clean(x))
        analyser = SentimentIntensityAnalyzer()
        scores=[]
        for i in range(len(df['Content2'])):
            score = analyser.polarity_scores(df['Content2'][i])
            score=score['compound']
            scores.append(score)
        sentiment=[]
        for i in scores:
            if i>=0.44:
                sentiment.append('Positive')
            elif i<=0.43:
                sentiment.append('Negative')
            else:
                sentiment.append('Neutral')
        df['sentiment']=pd.Series(np.array(sentiment))
        df['score']=pd.Series(np.array(scores))

        def clean_text(text):        
            text = str(text).lower()
            text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
            return text
        
        df['Content2'] = df['Content2'].apply(lambda x:clean_text(x))

        def tokenization(text):
            text = re.split('\W+', text)
            return text

        df['Content2'] = df['Content2'].apply(lambda x: tokenization(x.lower()))
        stopword = nltk.corpus.stopwords.words('english')
        def remove_stopwords(text):
            text = [word for word in text if word not in stopword]
            return text
            
        df['Content2'] = df['Content2'].apply(lambda x: remove_stopwords(x))
        wn = nltk.WordNetLemmatizer()
        def lemmatizer(text):
            text = [wn.lemmatize(word) for word in text]
            return text
        
        df['Content2'] = df['Content2'].apply(lambda x: lemmatizer(x))
        return df
    
#######################################################################################################

# Functions that will be used in the app

class Page_1():
    
    def Real_Cust(DF):
        Real_DF = DF.copy()
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Customer"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "customer"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Unknown"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "unknown"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "David"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "John"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Paul"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Peter"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Michael"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Anonymous"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Chris"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Robert"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Daniel"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Martin"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Thomas"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Andrew"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "James"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Richard"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Alex"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Mark"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Kunde"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Steve"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Stephen"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Mike"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "George"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Susan"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Ian"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Anna"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Alan"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Jan"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Frank"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Dan"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Brian"]
        Real_DF = Real_DF.loc[Real_DF["Author"] != "Kevin"]
        return Real_DF

    def Proportion_1_Comment(DF):
        Real_DF = Page_1.Real_Cust(DF)
        A = Real_DF["Author"].value_counts()
        return [len(A.loc[A == 1]),len(A.loc[A == 1])/len(A)]
    
    def Make_Funnel(DF):
        

        # Define the data for the chart
        data = go.Funnel(
            y = ['More than 1', 'More than 2', 'More than 3', 'More than 4', "5"],
            x = [len(DF.loc[DF["Rate"] >= 1]), len(DF.loc[DF["Rate"] >= 2]), len(DF.loc[DF["Rate"] >= 3]), len(DF.loc[DF["Rate"] >= 4]), len(DF.loc[DF["Rate"] == 5])]
        )

        # Define the layout for the chart
        layout = go.Layout(title='Rate distribution')

        # Create the figure object
        fig = go.Figure(data=[data], layout=layout)
        return fig


    def Funnel2(DF):
        temp = DF.groupby('sentiment').count()['Content'].reset_index().sort_values(by='Content',ascending=False)
        layout = go.Layout(title='Sentiment distribution')
        plt.figure(figsize=(12,6))
        sb.countplot(x='sentiment',data=DF)
        fig = go.Figure(go.Funnelarea(
                        text =temp.sentiment,
                        values = temp.Content
    ), layout=layout)
        return fig

    def Tree_Map(DF):
        all_words=[]
        for i in range(len(DF['Content2'])):
            a=DF['Content2'][i]
            for i in a:
                all_words.append(i)
        all_words=pd.Series(np.array(all_words))
        common_words=all_words.value_counts()[:30].rename_axis('Common Words').reset_index(name='count')
        layout = go.Layout(title='30 Most Common Words In Comments') 
        fig = px.treemap(common_words, path=['Common Words'], values='count',title='30 Most Common Words In Comments')
        return fig

    def Tree_Map_Neg(DF):
        DF = DF.loc[DF["sentiment"] == "Negative"].reset_index(drop=True)
        all_words=[]
        for i in range(len(DF['Content2'])):
            a=DF['Content2'][i]
            for i in a:
                all_words.append(i)
        all_words=pd.Series(np.array(all_words))
        common_words=all_words.value_counts()[:30].rename_axis('Common Words').reset_index(name='count')
        layout = go.Layout(title='30 Most Common Words In Negative Comments') 
        fig = px.treemap(common_words, path=['Common Words'], values='count',title='30 Most Common Words In Negative Comments')
        return fig
    
    def Tree_Map_Pos(DF):
        DF = DF.loc[DF["sentiment"] == "Positive"].reset_index(drop=True)
        all_words=[]
        for i in range(len(DF['Content2'])):
            a=DF['Content2'][i]
            for i in a:
                all_words.append(i)
        all_words=pd.Series(np.array(all_words))
        common_words=all_words.value_counts()[:30].rename_axis('Common Words').reset_index(name='count')
        layout = go.Layout(title='30 Most Common Words In Positive Comments') 
        fig = px.treemap(common_words, path=['Common Words'], values='count',title='30 Most Common Words In Positive Comments')
        return fig


    def Scatter(DF):
        layout = go.Layout(title='Link between rate and analysed sentiment')
        plt.figure(figsize=(12,6))
        fig = go.Figure(px.scatter(DF, x="score", y="Rate",
                 color="Rate"), layout=layout)
        return fig
    
    def Make_Rate_Sentiment(DF):
        X = sorted(DF["Rate"].unique())
        traces = []
        for sent in DF["sentiment"].unique():
            Y = []
            for rate in X:
                df = DF[DF["Rate"] == rate]
                Prop = len(df[df["sentiment"] == sent]) / len(df)
                Y.append(Prop)
            trace = go.Scatter(x=X, y=Y, mode='lines', name=sent)
            traces.append(trace)

        layout = go.Layout(title='Proportion of Sentiments by Rate',
                           xaxis=dict(title='Rate'),
                           yaxis=dict(title='Proportion'))
        
        fig = go.Figure(data=traces, layout=layout)
        return fig
    
    
    
    
#######################################################################################################
# Starting the app
AWS_DF = Work_Data.Make_DF_And_Cleaning()

# Créer l'application Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    # Bande transversale pertie supérieure
    html.Div(style={'background-color':'#1C1E2E','height':'90px'},children=[
        html.Img(src='assets/Wise_Logo.png', style={'height':'100%', 'width':'7%','display':'inline-block','vertical-align':'middle'})]),
    # Bande transversale partie inférieure
    html.Div(style={'background-color':'blue','height':'10%'},children = [
        dcc.Tabs(
            id = 'page-tabs',
            value='page-1',
            children=[
                dcc.Tab(label='General statistics',value='page-1'),
                dcc.Tab(label='Add a comment',value='page-2'),
                dcc.Tab(label='Exploring Database',value='page-3'),
                dcc.Tab(label='About the app',value='page-4'),
                ]
            )
        ]),
    html.Div(id="page-content")])



page_1_layout = html.Div(
    [
        html.Div([html.H1("General statistics about the Database", style={'text-align': 'center'}),
        html.Div([html.Button('Update DataBase', id='db_update', n_clicks=0, style={'position': 'absolute', 'top': '50%', 'right': '5%', 'transform': 'translateY(-50%)'}),
        html.Div(id='message_update', style={'textAlign': 'right', 'margin-right': '20px'})])
                 ],style={'position':'relative'}),
        html.H3("Total number of comments"),
        html.P(id='total_comments'),
        html.H3("Number of customers"),
        html.P(id='num_customers'),
        html.H3("Rates"),
        html.P(f"We show here a funnel chart of the rates distribution"),
        dcc.Graph(id='Graph_1-1', figure= Page_1.Make_Funnel(AWS_DF)),
        html.H3("Sentiment analysis"),
        html.P(id='positive_comments'),
        html.P(id='neutral_comments'),
        html.P(id='negative_comments'),
        html.Br(),
        html.P(f"We show here a funnel chart of the sentiment distribution"),
        dcc.Graph(id='Graph_2-1', figure= Page_1.Funnel2(AWS_DF), style={'width': '50%', 'textAlign': 'left'}),
        html.H3("Keywords"),
        html.P(f"We show here a tree map of the most frequent words"),
        dcc.Graph(id='Graph_3-1', figure= Page_1.Tree_Map(AWS_DF)),
        html.P(children = [f"The most frequent words amongst ",html.B("negative "), "comments are :"]),
        dcc.Graph(id='Graph_4-1', figure= Page_1.Tree_Map_Neg(AWS_DF)),
        html.P(children = [f"The most frequent words amongst ",html.B("positive "), "comments are :"]),
        dcc.Graph(id='Graph_4-2', figure= Page_1.Tree_Map_Pos(AWS_DF)),
        html.H3("Model Performance"),
        html.P(f"This is a correlation matrix between the given rate and the sentiment analyzed by the algorithm"),
        html.Div([dt.DataTable(id='matrix1')],style={'width': '50%', 'textAlign': 'left'}),
        html.Br(),
        html.Div([dt.DataTable(id='matrix2')],style={'width': '50%', 'textAlign': 'left'}),
        html.Br(),
        html.P(f"""We can observe the evolution of the proportion of positive, neutral and negative contents down below :"""),
        dcc.Graph(id='Graph_5', figure= Page_1.Make_Rate_Sentiment(AWS_DF)),
        html.P(id = "tendency"),
        html.P(id = "remarks"),
        html.P(f"""Let's try to understand why high rate comments are classified as negative and low rate comments classified as positive."""),
        html.P(f"""Let's first display comments with rate = 1 but classified as positive comment :"""),
        html.Div([dt.DataTable(id='rate_1_positive')],style={'width': '50%', 'textAlign': 'left'}),
        html.Br(),
        html.P(children = [html.Span("Remarks :", style={"text-decoration": "underline"}), " We notice that the model does not really understand the context of the sentence, especially that a 'not' before a positive word make it something negative. Ex : 'card has NOT arrived', 'this should be easy ...', 'good fu*king job !', 'Wise has been good until ...'"]),
        html.P(f"""Let's then display comments with rate = 5 but classified as negative comment :"""),
        html.Div([dt.DataTable(id='rate_5_negative')],style={'width': '50%', 'textAlign': 'left'}),
        html.P(children = [html.Span("Remarks :", style={"text-decoration": "underline"}), " Again, the model doesn't really understand the context, especially that a bad word before another bad word make it something good. Ex : 'small charges', 'low fees', 'low charges', 'no problem', etc ..."])
])


        
        
page_2_layout = html.Div([
    html.H3("Have you ever tried Wise services ? Please don't hesitate to post your personnal comments or feedback and help us to enhance our database !",
            style={
                'textAlign': 'center',
                'margin': '50px auto'
            }),
    
    html.Div([
        html.Label('Pseudo'),
        dcc.Input(
            id='pseudo-input',
            type='text',
            placeholder='Enter your pseudo',
            style={
                'width': '30%',
                'margin': '10px'
            }
        ),

        html.Label('Rate (Between 1 and 5)'),
        dcc.Input(
            id='rating-input',
            type='number',
            min=1,
            max=5,
            placeholder='Enter your note',
            style={
                'width': '30%',
                'margin': '10px'
            }
        ),
        
    ], style={
        'display': 'flex',
        'flex-direction': 'row',
        'align-items': 'center',
        'justify-content': 'center',
        'margin-bottom': '50px'
    }),

    html.Div([
        dcc.Textarea(
            id='comment-input',
            placeholder='Write your comment about Wise',
            style={
                'width': '90%',
                'height': '100px',
                'resize': 'vertical',
                'textAlign': 'center',
                'fontSize': '18px',
                'borderRadius': '10px',
                'border': '2px solid #ccc',
                'padding': '10px',
                'box-sizing': 'border-box',
                'position': 'relative',
                'margin': 'auto'
            }
        ),

        html.Button('Send', id='comment-submit', n_clicks=0, style={'position': 'absolute', 'top': '50%', 'right': '5%', 'transform': 'translateY(-50%)'})
    ], style={
        'position': 'relative'
    }),

    html.Div(id='output-message', style={'textAlign': 'center', 'margin-top': '20px'})
])

@app.callback(Output('output-message', 'children'),
              Input('comment-submit', 'n_clicks'),
              State('pseudo-input', 'value'),
              State('rating-input', 'value'),
              State('comment-input', 'value'))
def update_output(n_clicks, pseudo, rating, comment):
    def clean(text):
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\s+', ' ', text, flags=re.I)
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub('<.*?>+', '', text)
        return text
    if n_clicks > 0:
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(clean(comment))
        compound_score = scores['compound']
        if compound_score >= 0.05:
            sentiment = 'positive'
        elif compound_score > -0.05 and compound_score < 0.05:
            sentiment = 'neutral'
        else:
            sentiment = 'negative'
        g = geocoder.ip('me')
        now = datetime.datetime.now()
        C1 = pseudo
        C2 = g.country
        C3 = rating 
        C4 = now.strftime("%Y-%m-%d %H:%M:%S")
        C5 = detect(comment)
        C6 = comment
        C7 = "Comment uploaded from Wise Analyzer (will not receive any response)"
        C8 = "Comment uploaded from Wise Analyzer (will not receive any response)"
        Line = [C1,C2,C3,C4,C5,C6,C7,C8]
        Conn = pymysql.connect(host = Host, user = User, password = Password)
        Curs = Conn.cursor()
        SQL2 = '''USE Wise_Data'''
        Curs.execute(SQL2)
        try:
            Curs.execute('''INSERT INTO Commentaires2 VALUES (%s,%s,%s,%s,%s,%s,%s,%s)''',Line)
        except:
            print("Already there") 
        Curs.connection.commit()
        Curs.close()
        Conn.close()
        return f"Your comment has been received, and it has been categorized {sentiment}. Pseudo : {pseudo}, Rate : {rating}, Comment : {comment}"
    else:
        return ""


page_3_layout = html.Div([
        html.Div([
        html.Label('Country'),
        dcc.Dropdown(AWS_DF['Country_Code'].unique().tolist(),
            id='D_country',
            value='CA',
            multi=True,
            style={
                'width': '30%',
                'margin': '10px'
            }
        ),

        html.Label('Rate'),
        dcc.Dropdown(AWS_DF['Rate'].unique().tolist(),
            id = 'D_rate',
            value= 1,
            multi=True,
            style={
                'width': '30%',
                'margin': '10px'
            }
        ),

       html.Label('Date of Review'),
       dcc.DatePickerRange(
                id='date_filter',
                start_date=pd.to_datetime(AWS_DF['Date_of_Review']).dt.date.min(),
                end_date=pd.to_datetime(AWS_DF['Date_of_Review']).dt.date.max(),

                min_date_allowed=pd.to_datetime(AWS_DF['Date_of_Review']).dt.date.min(),
                max_date_allowed=pd.to_datetime(AWS_DF['Date_of_Review']).dt.date.max(),
                style={
                'width': '50%',
                'margin': '20px'
            }
       )

    ], style={
        'display': 'flex',
        'flex-direction': 'row',
        'align-items': 'center',
        'justify-content': 'center',
        'margin': '50px'
    }),

html.Div([
            html.H2("Number of comments", style={'textAlign': 'left', 'color': 'mediumturquoise', 'margin': '20px'}),
            html.Div(id='nb_comment', style={'margin': '20px', 'margin-left': '5%'})
    ],style={
        'position': 'relative'}
   ),
html.Div([
            html.H2("Sentiment Distribution", style={'textAlign': 'left', 'color': 'mediumturquoise', 'margin': '20px'}),
            dt.DataTable(id='sen_dis',
             columns=[{"name": "sentiment", "id": "sentiment"},
                      {"name": "Count", "id": "Count"}])
             #data=AWS_DF.groupby('sentiment').count()['Content'].reset_index().sort_values(by='Content',ascending=False).to_dict("records"))
],style={'width': '20%'}),
html.Div([
            html.H2("Most common Keywords", style={'textAlign': 'left', 'color': 'mediumturquoise', 'margin': '20px'}),
            dt.DataTable(id='key_dis',
            columns=[{"name": "Keywords", "id": "Keywords"},
                      {"name": "Count", "id": "Count"}])
             #data=AWS_DF.groupby('sentiment').count()['Content'].reset_index().sort_values(by='Content',ascending=False).to_dict("records"))
],style={'width': '20%'})
])
@app.callback(Output(component_id='sen_dis', component_property='data'),
              Input(component_id='D_country', component_property='value'),
              Input(component_id='D_rate', component_property='value'),
              Input(component_id='date_filter', component_property='start_date'),
              Input(component_id='date_filter', component_property='end_date'))
def update_tab(D_country, D_rate,start_date, end_date):
    start_date = start_date + ' 00:00:00'
    end_date = end_date + ' 00:00:00'
    df=AWS_DF[(AWS_DF['Date_of_Review']>=start_date) & (AWS_DF['Date_of_Review']<=end_date)]
    if type(D_country)==str:
        df=df[df['Country_Code']==D_country]
    else:
        df=df[df['Country_Code'].isin(D_country)]
    if type(D_rate)==int:
        df=df[df['Rate']==D_rate]
    else:
        df=df[df['Rate'].isin(D_rate)]
    tab = df.groupby('sentiment').count()['Content'].reset_index(name='Count').sort_values(by='Count',ascending=False).to_dict("records")
    return tab

@app.callback(Output('nb_comment', 'children'),
              Input('D_country', 'value'),
              Input('D_rate', 'value'),
              Input('date_filter', 'start_date'),
              Input('date_filter', 'end_date'))
def update_nb(D_country, D_rate,start_date, end_date):
    start_date = start_date + ' 00:00:00'
    end_date = end_date + ' 00:00:00'
    df=AWS_DF[(AWS_DF['Date_of_Review']>=start_date) & (AWS_DF['Date_of_Review']<=end_date)]
    if type(D_country)==str:
        df=df[df['Country_Code']==D_country]
    else:
        df=df[df['Country_Code'].isin(D_country)]
    if type(D_rate)==int:
        df=df[df['Rate']==D_rate]
    else:
        df=df[df['Rate'].isin(D_rate)]
    return f"{len(df)} ({round((len(df)*100)/len(AWS_DF),2)}%)"  

@app.callback(Output(component_id='key_dis', component_property='data'),
              Input(component_id='D_country', component_property='value'),
              Input(component_id='D_rate', component_property='value'),
              Input(component_id='date_filter', component_property='start_date'),
              Input(component_id='date_filter', component_property='end_date'))
def update_tab2(D_country, D_rate,start_date, end_date):
    start_date = start_date + ' 00:00:00'
    end_date = end_date + ' 00:00:00'
    df=AWS_DF[(AWS_DF['Date_of_Review']>=start_date) & (AWS_DF['Date_of_Review']<=end_date)]
    if type(D_country)==str:
        df=df[df['Country_Code']==D_country]
    else:
        df=df[df['Country_Code'].isin(D_country)]
    if type(D_rate)==int:
        df=df[df['Rate']==D_rate]
    else:
        df=df[df['Rate'].isin(D_rate)]
    all_words=[]
    for i in range(len(df['Content'])):
        a=df['Content'].reset_index()['Content'][i]
        for i in a:
            all_words.append(i)
    all_words=pd.Series(np.array(all_words))
    tab = all_words.value_counts().rename_axis('Keywords').reset_index(name='Count').sort_values(by='Count',ascending=False).head().to_dict("records")
    return tab




page_4_layout = html.Div(
    [
        html.H1("About this app", style={"text-align": "center"}),
        html.H2("Context"),
        html.P("This application was created by Antoine Ballet and Marie Bonin in 2023."),
        html.P("If you want to know more about this app, please visit the GitHub page of the project :"),
        html.A("Github Page", href="https://github.com/eniotnaCDMX/DataScientest_DE_June2022_Project", target="_blank"),
    ]
)




# Callback made to switch pages

@app.callback(
    Output('page-content', 'children'),
    Input('page-tabs', 'value')
)
def Make_page_content(tab):
    if tab == 'page-1':
        return page_1_layout
    if tab == 'page-2':
        return page_2_layout
    if tab == 'page-3':
        return page_3_layout
    if tab == 'page-4':
        return page_4_layout


############################ Callbacks for first page

# Callback for the Update Database button in the first page
@app.callback(
    Output('message_update', 'children'),
    [Input('db_update', 'n_clicks')])
def refresh_data(n_clicks):
    if n_clicks > 0:
        global AWS_DF
        AWS_DF = Work_Data.Make_DF_And_Cleaning()
        return "Database updated, please press F5"
    else:
        return ""

# Callback for the first figure, Funnel Chart, in the first page
@app.callback(
    Output('Graph_1-1', 'figure'),
    Input('db_update', 'n_clicks'))
def update_graph_1(n_clicks):
    return Page_1.Make_Funnel(AWS_DF)

@app.callback(
    Output('Graph_2-1', 'figure'),
    Input('db_update', 'n_clicks'))
def update_graph_2(n_clicks):
    return Page_1.Funnel2(AWS_DF)

@app.callback(
    Output('Graph_3-1', 'figure'),
    Input('db_update', 'n_clicks'))
def update_graph_3(n_clicks):
    return Page_1.Tree_Map(AWS_DF)



# Callback to update Graph 4-1 in page 1
@app.callback(
    Output('Graph_4-1', 'figure'),
    Input('db_update', 'n_clicks'))
def update_graph_4_1(n_clicks):
    return Page_1.Tree_Map_Neg(AWS_DF)


# Callback to update Graph 4-1 in page 1
@app.callback(
    Output('Graph_4-2', 'figure'),
    Input('db_update', 'n_clicks'))
def update_graph_4_2(n_clicks):
    return Page_1.Tree_Map_Pos(AWS_DF)



# Callback to update Graph 5 in page 1
@app.callback(
    Output('Graph_5', 'figure'),
    Input('db_update', 'n_clicks'))
def update_graph_5(n_clicks):
    return Page_1.Make_Rate_Sentiment(AWS_DF)


# Callback for analyzing sentiment distribution for a given rate
@app.callback(
    Output('tendency', 'children'),
    Input('db_update', 'n_clicks'))
def update_tendency_analysis(n_clicks):
    return f"Remarks : We can observe that, as the rate increases, the proportion of 'positive' comments increases as well and the proportion of 'negative' decreases, which makes sense. We see that, if rate is 1 or 2, there are more negative comments than negative ones, and when rate is 4 or 5, there are more positive comments than negative ones. When the rate is equal to 3, the proportion of positive comments is pretty close to the proportion of negative comments, which makes sense, because it's actually unclear whether 3 is a good rate or not ..."


# Callback for making remarks on sentiment distribution for a given rate
@app.callback(
    Output('remarks', 'children'),
    Input('db_update', 'n_clicks'))
def update_tendency_remarks(n_clicks):
    Comm = "However, we might expect that, for rate = 1, the proportion of negative comments is closer to 100%, and for rate = 5 the proportion of positive comments closer to 100% as well. "
    P_neg_r5 = len(AWS_DF.loc[(AWS_DF["Rate"] == 5) & (AWS_DF["sentiment"] == "Negative")] )/len(AWS_DF.loc[AWS_DF["Rate"] == 5])
    P_pos_r1 = len(AWS_DF.loc[(AWS_DF["Rate"] == 1) & (AWS_DF["sentiment"] == "Positive")] )/len(AWS_DF.loc[AWS_DF["Rate"] == 1])
    if P_pos_r1 >= P_neg_r5:
        Comm = Comm + "The model actually seems to have a positive biais."
    else:
        Comm = Comm + "The model actually seems to have a negative biais."
    return Comm




@app.callback(
    Output('matrix1', 'data'),
    Input('db_update', 'n_clicks'))
def update_matrix1(n_clicks):
    df = pd.crosstab(AWS_DF['Rate'], AWS_DF['sentiment'])
    df['Rate'] = df.index
    df = df[['Rate', 'Negative', 'Neutral', 'Positive']]
    tab = df.to_dict('records')
    return tab

@app.callback(
    Output('matrix2', 'data'),
    Input('db_update', 'n_clicks'))
def update_matrix2(n_clicks):
    df = pd.crosstab(AWS_DF['Rate'], AWS_DF['sentiment'], normalize='index')
    df['Rate'] = df.index
    df = df[['Rate', 'Negative', 'Neutral', 'Positive']]
    df['Negative']=df['Negative'].apply(lambda x : str(round(x*100,1)) + '%')
    df['Neutral']=df['Neutral'].apply(lambda x : str(round(x*100,1)) + '%')
    df['Positive']=df['Positive'].apply(lambda x : str(round(x*100,1)) + '%')
    tab = df.to_dict('records')
    return tab

@app.callback(
    Output('rate_1_positive', 'data'),
    Input('db_update', 'n_clicks'))
def update_rate_1_positive(n_clicks):
    df = AWS_DF[(AWS_DF['Rate']== 1) & (AWS_DF['sentiment']=='Positive')]
    df['lenght']=df['Content'].apply(lambda x : len(x))
    df = df[df['lenght']<=120]
    df =  df[['Author', 'Country_Code', 'Rate', 'Date_of_Review', 'Language','Content', 'Date_of_Reply', 'sentiment', 'score']].head(20)
    tab = df.to_dict('records')
    return tab



@app.callback(
    Output('rate_5_negative', 'data'),
    Input('db_update', 'n_clicks'))
def update_rate_5_negative(n_clicks):
    df = AWS_DF[(AWS_DF['Rate']== 5) & (AWS_DF['sentiment']=='Negative')]
    df['lenght']=df['Content'].apply(lambda x : len(x))
    df = df[df['lenght']<=120]
    df =  df[['Author', 'Country_Code', 'Rate', 'Date_of_Review', 'Language','Content', 'Date_of_Reply', 'sentiment', 'score']].head(20)
    tab = df.to_dict('records')
    return tab




@app.callback(
    Output('positive_comments', 'children'),
    Input('db_update', 'n_clicks'))
def update_positive_comments(n_clicks):
    return f'There is for the moment {np.round(AWS_DF["sentiment"].value_counts(normalize=True)["Positive"]*100,2)}% positive comments'

@app.callback(
    Output('neutral_comments', 'children'),
    Input('db_update', 'n_clicks'))
def update_neutral_comments(n_clicks):
    return f'There is for the moment {np.round(AWS_DF["sentiment"].value_counts(normalize=True)["Neutral"]*100,2)}% neutral comments'

@app.callback(
    Output('negative_comments', 'children'),
    Input('db_update', 'n_clicks'))
def update_negative_comments(n_clicks):
    return f'There is for the moment {np.round(AWS_DF["sentiment"].value_counts(normalize=True)["Negative"]*100,2)}% negative comments'


# Callback used to generate the first comment of Page 1, in the "Total number of comments" section
@app.callback(
    Output('total_comments', 'children'),
    Input('db_update', 'n_clicks'))
def update_total_comments(n_clicks):
    return f"Our data base currently contains {len(AWS_DF)} comments. The older one was posted on {AWS_DF['Date_of_Review'].min()[:10]} and the newest one was posted on {AWS_DF['Date_of_Review'].max()[:10]}"


# Callback used to generate the second comment of Page 1, in the "Number of customers" section
@app.callback(
    Output('num_customers', 'children'),
    Input('db_update', 'n_clicks'))
def update_num_customers(n_clicks):
    num_customers = len(Page_1.Real_Cust(AWS_DF)['Author'].unique())
    proportion = Page_1.Proportion_1_Comment(AWS_DF)
    return f"A total of {num_customers} persons left a commentary. From this population, {proportion[0]} left only 1 comment, which is around {np.round(proportion[1], 2) * 100}%"


# Lancer l'application
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
