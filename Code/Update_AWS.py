# -*- coding: utf-8 -*-

###############################################################################

# Importing libraries
import pandas as pd
import pymysql
import requests
from bs4 import BeautifulSoup
import numpy as np
import json

###############################################################################

# AWS Database Credential

Instance_Identifier = ""
Port = 0
Password = ""
User = ""
Host = ""

###############################################################################

def New_Data():
    Number_Of_Pages = 4
    DF = pd.DataFrame(columns=['Author','Country_Code','Rate','Date_Of_Review','Language','Content','Reply','Date_of_Reply'])
    for i in range(1,Number_Of_Pages+1):
        url = f'https://www.trustpilot.com/review/wise.com?page={i}'
        reponse = requests.get(url)
        soup = BeautifulSoup(reponse.text, "html.parser")
        #return soup
        items = json.loads(soup.findAll('script')[-1].contents[0])["props"]['pageProps']['reviews']
        for i in range(len(items)):
            Author = items[i]['consumer']["displayName"]
            Country = items[i]['consumer']["countryCode"]
            Rate = items[i]["rating"]
            Date_Review = items[i]["dates"]["publishedDate"].replace('T',' ')[:-5]
            Lang = items[i]["language"]
            Cont = items[i]["text"].replace("\n"," ")
            if items[i]["reply"] == None:
                Rep = np.nan
                Date_Rep = np.nan
            else:
                Rep = items[i]["reply"]["message"].replace("\n"," ")
                Date_Rep = items[i]["reply"]['publishedDate'].replace('T',' ')[:-5]
            DF = pd.concat([DF,pd.DataFrame([[Author,Country,Rate,Date_Review,Lang,Cont,Rep,Date_Rep]],columns = DF.columns)],ignore_index = True)
    return DF


Conn = pymysql.connect(host = Host, user = User, password = Password)
Curs = Conn.cursor()
SQL2 = '''USE Wise_Data'''
Curs.execute(SQL2)
DB = pd.read_sql_query('''SELECT * FROM Commentaires2''',Conn).sort_values("Date_of_Review",ascending = False)
New_Lines = New_Data().drop_duplicates()
New_Lines[New_Lines.columns[0]] = New_Lines[New_Lines.columns[0]].fillna("")
New_Lines[New_Lines.columns[1]] = New_Lines[New_Lines.columns[1]].fillna("")
New_Lines[New_Lines.columns[2]] = New_Lines[New_Lines.columns[2]].fillna(0)
New_Lines[New_Lines.columns[3]] = New_Lines[New_Lines.columns[3]].fillna("")
New_Lines[New_Lines.columns[4]] = New_Lines[New_Lines.columns[4]].fillna("")
New_Lines[New_Lines.columns[5]] = New_Lines[New_Lines.columns[5]].fillna("")
New_Lines[New_Lines.columns[6]] = New_Lines[New_Lines.columns[6]].fillna("")
New_Lines[New_Lines.columns[7]] = New_Lines[New_Lines.columns[7]].fillna("")
for i in range(len(New_Lines)):
    print(i)
    try:
        Curs.execute('''INSERT INTO Commentaires2 VALUES (%s,%s,%s,%s,%s,%s,%s,%s)''',list(New_Lines.iloc[i]))
    except:
        print("Already there")
        
Curs.connection.commit()
Curs.close()
Conn.close()
