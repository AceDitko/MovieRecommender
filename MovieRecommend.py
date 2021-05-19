import json
import pandas as pd
import numpy as np
import requests
import datetime
import gspread
import spacy
import re
from io import StringIO
#import functions
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from oauth2client.service_account import ServiceAccountCredentials
pd.options.mode.chained_assignment = None  # default='warn'

class Movies:

    def __init__(self, start_year="2018"):

        self.file_names = []
        self.paths = []
        self.csv_df_list = []
        self.input_df_list = []
        self.json_data_list = []
        self.spreadsheet_json = []
        self.imdb_json = []
        self.spreadsheet_df = []
        self.imdb_df = []
        self.input_df = pd.DataFrame()
        self.year_dict = {"2018": 0, "2019": 1, "2020": 2, "2021": 3}

        self.rt_ratings = []
        self.j_ratings = []
        self.j_ratings_percent = []

        self.colours = ["cornflowerblue", "indianred", "limegreen"]

        self.url = 'https://docs.google.com/spreadsheets/d/1w-LyhsGUqNFiFqsF2QVgfA5MARTMRWFxMbG-FiWQf9c/edit#gid=621622277'

        self.load_dataset(start_year)

    def getdays(self, date):
        '''
        Creates an integer number representing the difference between
        a given date and today's date.
        '''

        test_date = date
        temp_test_dates = test_date.split('/')
        test_dates = [int(i) for i in temp_test_dates]

        today = datetime.date.today()
        today_date = today.strftime('%d/%m/%Y')
        temp_today_dates = today_date.split('/')
        today_dates = [int(i) for i in temp_today_dates]

        d0 = datetime.date(test_dates[2], test_dates[1], test_dates[0])
        d1 = datetime.date(today_dates[2], today_dates[1], today_dates[0])
        delta = d1 - d0
        return delta.days

    def get_df(self):

        return self.input_df

    def train_model(self):

        self.create_features()

    def create_features(self):

        print("In create features")
        
        print("Writer column cleaned and assigned")

        in_df = self.input_df
        
        print("Dataframe assigned for features")

        in_df['Writer'] = in_df['Writer'].apply(lambda x: re.sub("[\(\[].*?[\)\]]", "", str(x)))
        form_cols = ['Genre', 'Actors', 'Director', 'Production', 'Writer']
        
        def dropPunc(doc):
            tokens = []
            for token in doc:
                if not token.is_punct and not token.is_stop:
                    tokens.append(token.lemma_)
            return tokens
        
        nlp = spacy.load('en_core_web_sm')

        in_df['Plot Lemma'] = in_df['Plot'].apply(lambda x: dropPunc(nlp(x.lower())))

        print("Plot lemma column created")

        for col in form_cols:
            in_df[col] = in_df[col].str.replace(' ','')
            in_df[col] = in_df[col].str.replace('N/A','nan')
            in_df[col] = in_df[col].apply(lambda x: x.lower())
            in_df[col] = in_df[col].apply(lambda x: re.split(',|/',x))
        
        print("Form columns split")

        g_df = pd.DataFrame(in_df.Genre.values.tolist()).add_prefix('Genre_')
        a_df = pd.DataFrame(in_df.Actors.values.tolist()).add_prefix('Actors_')
        d_df = pd.DataFrame(in_df.Director.values.tolist()).add_prefix('Director_')
        p_df = pd.DataFrame(in_df.Production.values.tolist()).add_prefix('Production_')
        w_df = pd.DataFrame(in_df.Writer.values.tolist()).add_prefix('Writer_')
        
        print("Created new dataframes of split columns")

        join_dfs = [g_df, a_df, d_df, p_df, w_df]

        print("Joining dataframes")

        for df in join_dfs:
            in_df = in_df.merge(df, left_index=True, right_index=True)
            #in_df = pd.concat([in_df, df], axis=1, keys=[in_df.index])
            #in_df = pd.concat([in_df, df])

        print("Dataframes joined")

        #in_df['Bag of Words'] = (in_df['Plot Lemma'] + 
                                 #in_df['Actors_bow'] +
                                 #in_df['Director_bow'] +
                                 #in_df['Genre_bow'] +
                                 #in_df['Production_bow'] +
                                 #in_df['Writer_bow'])

        #in_df = in_df.drop(columns=bow_col_list)
        self.input_df = in_df



    def load_dataset(self, start_year):
        '''
        Calls the pull_from_sheet method for each year of recorded
        data (beginning in 2018) including the current year.
        '''
        
        today = datetime.date.today()
        today_dates = [int(i) for i in today.strftime('%d/%m/%Y').split('/')]
        current_year = today_dates[2]

        year_diff = current_year - int(start_year)

        for i in range (0, year_diff+1):
            year = int(start_year) + i
            self.pull_from_sheet(str(year))

        self.merge_sheets()

    def merge_sheets(self):
        '''
        Concatenates the dfs for all years and creates an input df
        with relevant data only.
        '''

        self.spreadsheet_df[-1] = self.spreadsheet_df[-1][self.spreadsheet_df[-1].Name != ""]

        c_imdb_df = pd.concat(self.imdb_df)
        c_sheet_df = pd.concat(self.spreadsheet_df)
        c_imdb_df['Format'] = c_sheet_df['Format'].values
        c_imdb_df['Days to View'] = c_sheet_df['Days to View'].values
        c_imdb_df['Days Since Release'] = c_sheet_df['Release Date'].apply(self.getdays)
        c_imdb_df['True Rating'] = c_sheet_df['True Rating'].apply(lambda x: int(x))
        c_imdb_df = c_imdb_df.drop(columns=['imdbID', 'imdbVotes', 'Response', 'Type', 'Website', 'Language', 'DVD', 'Country']).reset_index(drop=True)
        self.input_df = c_imdb_df

    def pull_from_sheet(self, year="2018"):
        '''
        Pulls the data (created by user input) from the google sheet 
        for a given year and stores it in a json object, appending 
        each json object to a dataframe in the class. Calls the
        pull_from_imdb method using the aformentioned json object 
        as input. 
        '''

        scope = [   'https://spreadsheets.google.com/feeds',
                    'https://www.googleapis.com/auth/spreadsheets',
                    'https://www.googleapis.com/auth/drive.file',
                    'https://www.googleapis.com/auth/drive']

        creds = ServiceAccountCredentials.from_json_keyfile_name('service_account.json', scope)
        client = gspread.authorize(creds)

        sheet_title = "Films " + year
        sheet = client.open_by_url(self.url).worksheet(sheet_title)
        json_sheet = sheet.get_all_records()
        json_object = json.dumps(json_sheet)

        self.spreadsheet_df.append(pd.read_json(json_object))
        self.pull_from_imdb(year)

    def pull_from_imdb(self, year="2018"):
        '''
        Scrapes imdb for movie data for each film using omdb. The year
        is used to differentiate films if duplicates of a film title are
        present in the data (e.g. in the case of film remakes). 
        '''

        in_df = self.spreadsheet_df[self.year_dict[year]]    
        in_df = in_df[in_df.Name != ""]

        file_name = year+"_imdb.json"

        api_key = "?apikey=72bc447a&t="
        url = "http://omdbapi.com/"

        search_df = in_df[['Name', 'Release Date']]
        search_df['search_title'] = search_df['Name'].apply(self.name_search_string)
        search_df['search_year'] = search_df['Release Date'].apply(self.year_search_string)

        search_df['counts'] = search_df['search_title'].map(search_df['search_title'].value_counts())
        search_df['counts'].apply(lambda x: int(x))
        
        search_df['search_key'] = np.where(search_df['counts'] == 1, api_key + search_df['search_title'], api_key+search_df['search_title'] + "&y=" + search_df['search_year'])

        out_list = []

        for key in search_df['search_key'].to_list():
            key_url = url + key
            r = requests.get(key_url)
            json_data = r.json()
            print("importing " + json_data['Title'] + "...")
            out_list.append(json_data)

        json_object = json.dumps(out_list)
        self.imdb_df.append(pd.read_json(StringIO(json_object)))

    def name_search_string(self, title):
        substrings = str(title).split(" ")
        search_string = ""
        for substring in substrings:
            search_string += substring + "+"
        return search_string

    def year_search_string(self, date):     
        dates = date.split("/")
        return str(dates[-1])

    def print_files(self):
        print(self.input_df)

    def preview_files(self):
        print(self.input_df.head())