"""Processes data and trains pipeline for MovieRecommend project."""
import json
import pandas as pd
import numpy as np
import requests
import datetime
import gspread
import spacy
import re
from tqdm import tqdm
from io import StringIO

# import functions
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from oauth2client.service_account import ServiceAccountCredentials

pd.options.mode.chained_assignment = None  # default='warn'

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer


class Movies:
    """Contains all information for MovieRecommend model."""

    def __init__(self, start_year="2018"):
        """Initialize empty structure and loads dataset."""
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
        self.year_dict = {
            "2018": 0,
            "2019": 1,
            "2020": 2,
            "2021": 3,
            "2022": 4,
            "2023": 5,
        }

        self.rt_ratings = []
        self.j_ratings = []
        self.j_ratings_percent = []

        self.colours = ["cornflowerblue", "indianred", "limegreen"]

        self.url = "https://docs.google.com/spreadsheets/d/1w-LyhsGUqNFiFqsF2QVgfA5MARTMRWFxMbG-FiWQf9c/edit#gid=621622277"

        self.load_dataset(start_year)

    def getdays(self, date):
        """Calculate difference in days between today and given date."""
        test_date = date
        temp_test_dates = test_date.split("/")
        test_dates = [int(i) for i in temp_test_dates]

        today = datetime.date.today()
        today_date = today.strftime("%d/%m/%Y")
        temp_today_dates = today_date.split("/")
        today_dates = [int(i) for i in temp_today_dates]

        d0 = datetime.date(test_dates[2], test_dates[1], test_dates[0])
        d1 = datetime.date(today_dates[2], today_dates[1], today_dates[0])
        delta = d1 - d0
        return delta.days

    def is_weekday(self, date):
        """Calculate wether film was watched on weekday or weekend."""
        dt_obj = datetime.datetime.strptime(date, "%d/%m/%Y")

        if dt_obj.weekday() < 5:
            return "Weekday"
        else:
            return "Weekend"

    def is_cinema(self, format):
        """Determine if film was streamed or watched in a cinema."""
        streams = ["Netflix", "Amazon", "NowTv", "Plex"]
        if format in streams:
            return "Stream"
        else:
            return "Cinema"

    def get_df(self):
        """Return movie dataframe."""
        return self.input_df

    def train_model(self):
        """Train the model."""
        self.clean_data()
        self.create_features()
        self.preprocess()

    def clean_data(self):
        """Clean the data."""
        in_df = self.input_df

        in_df["BoxOffice"] = in_df["BoxOffice"].str.replace(",", "")
        in_df["BoxOffice"] = in_df["BoxOffice"].str.replace("$", "")
        in_df["BoxOffice"] = in_df["BoxOffice"].replace("N/A", np.NaN)
        in_df["BoxOffice"] = in_df["BoxOffice"].apply(lambda x: float(x))

        in_df["Released"] = pd.to_datetime(in_df["Released"], errors="coerce")
        in_df["Released"] = in_df["Released"].dt.strftime("%d/%m/%y")

        in_df["Runtime"] = in_df["Runtime"].str.replace(" min", "")
        in_df["Runtime"] = in_df["Runtime"].replace("N/A", np.NaN)
        in_df["Runtime"] = in_df["Runtime"].apply(lambda x: float(x))

        in_df["True Rating"] = in_df["True Rating"].apply(lambda x: int(x))

        self.input_df = in_df

    def create_features(self):
        """Create useful features."""
        print("In create features")

        print("Writer column cleaned and assigned")

        in_df = self.input_df

        print("Dataframe assigned for features")

        in_df["Is_cinema"] = in_df["Format"].apply(lambda x: self.is_cinema(x))
        in_df["Is_weekday"] = in_df["Viewing Date"].apply(lambda x: self.is_weekday(x))

        in_df["Writer"] = in_df["Writer"].apply(
            lambda x: re.sub("[\(\[].*?[\)\]]", "", str(x))
        )
        form_cols = ["Actors", "Director", "Production", "Writer"]

        print(in_df["Director"])

        def dropPunc(doc):
            tokens = []
            for token in doc:
                if not token.is_punct and not token.is_stop:
                    tokens.append(token.lemma_)
            return tokens

        nlp = spacy.load("en_core_web_sm")

        in_df["Plot Lemma"] = in_df["Plot"].apply(lambda x: dropPunc(nlp(x.lower())))

        print("Plot lemma column created")

        for col in form_cols:
            in_df[col] = in_df[col].str.replace(" ", "")
            in_df[col] = in_df[col].str.replace("N/A", "nan")
            in_df[col] = in_df[col].apply(lambda x: x.lower())
            in_df[col] = in_df[col].apply(lambda x: re.split(",|/", x))

        print("Form columns split")

        temp_df = in_df[form_cols[0]]
        for col in form_cols[1:]:
            temp_df += in_df[col]

        temp_df2 = pd.DataFrame(temp_df.values.tolist()).add_prefix("Word_")
        for col in temp_df2.columns:
            temp_df2[col] = temp_df2[col].replace("nan", np.nan)
            temp_df2[col] = temp_df2[col].fillna(value=np.nan)

        self.rare_encode("Word", temp_df2)

        to_join_df = temp_df2[temp_df2.columns.tolist()].values.tolist()

        to_join_df = pd.DataFrame({"Bag_of_words": to_join_df})

        in_df = in_df.merge(to_join_df, left_index=True, right_index=True)

        self.input_df = in_df

    def rare_encode(self, col_str, in_df):
        """Encode infrequent variables."""
        col_str += "_"

        in_cols = [col for col in in_df.columns if col_str in col]
        temp_df = in_df[in_cols].apply(pd.Series.value_counts)
        temp_df.fillna(0, inplace=True)

        temp_df["Sum"] = 0
        print("Data frame pre count")
        print(temp_df)
        for col in in_cols:
            temp_df["Sum"] += temp_df[col]

        temp_df = temp_df[temp_df["Sum"] > 1]
        pop_vals = temp_df.index.to_list()
        for col in in_cols:
            in_df[col] = in_df[col].apply(
                lambda x: x if str(x) in pop_vals else "other"
            )

    def preprocess(self):
        """Preprocess the data."""
        in_df = self.input_df

        numerical_cols = [
            "BoxOffice",
            "Metascore",
            "Runtime",
            "Year",
            "imdbRating",
            "Days to View",
            "Days Since Release",
        ]
        categorical_cols = ["Rated", "Format", "Is_cinema", "Is_weekday"]
        form_cols = ["Bag_of_words"]

        features = numerical_cols + categorical_cols

        for i in features:
            in_df[i] = in_df[i].replace("N/A", np.NaN)
            in_df[i] = in_df[i].replace("", np.NaN)
            in_df[i] = in_df[i].replace(" ", np.NaN)
            in_df[i] = in_df[i].replace("missing_value", np.NaN)
            in_df[i] = in_df[i].replace("nan", np.NaN)

        numerical_transformer = SimpleImputer(missing_values=np.NaN, strategy="median")

        categorical_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(missing_values=np.NaN, strategy="most_frequent"),
                ),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_cols),
                ("cat", categorical_transformer, categorical_cols)
                # ('tfidf', tfidf_transformer, plots)
            ]
        )

    def load_dataset(self, start_year):
        """Load the dataset."""
        today = datetime.date.today()
        today_dates = [int(i) for i in today.strftime("%d/%m/%Y").split("/")]
        current_year = today_dates[2]

        year_diff = current_year - int(start_year)

        if Path("movie_db.json").is_file():
            print("Loading saved movie database...")
            self.input_df = pd.read_json("movie_db.json")
        else:
            for i in range(0, year_diff + 1):
                year = int(start_year) + i
                self.pull_from_sheet(str(year))

            self.merge_sheets()

            self.input_df.to_json("movie_db.json")

    def merge_sheets(self):
        """Merge sheets from all years together."""
        self.spreadsheet_df[-1] = self.spreadsheet_df[-1][
            self.spreadsheet_df[-1].Name != ""
        ]

        c_imdb_df = pd.concat(self.imdb_df)
        c_sheet_df = pd.concat(self.spreadsheet_df)
        c_imdb_df["Format"] = c_sheet_df["Format"].values
        c_imdb_df["Viewing Date"] = c_sheet_df["Viewing Date"].values
        c_imdb_df["Days to View"] = c_sheet_df["Days to View"].values
        c_imdb_df["Days Since Release"] = c_sheet_df["Release Date"].apply(self.getdays)
        c_imdb_df["True Rating"] = c_sheet_df["True Rating"].apply(lambda x: int(x))
        c_imdb_df = c_imdb_df.drop(
            columns=[
                "imdbID",
                "imdbVotes",
                "Response",
                "Type",
                "Website",
                "Language",
                "DVD",
                "Country",
            ]
        ).reset_index(drop=True)
        self.input_df = c_imdb_df

    def pull_from_sheet(self, year="2018"):
        """
        Scrapes google sheet.

        Pulls the data (created by user input) from the google sheet
        for a given year and stores it in a json object, appending
        each json object to a dataframe in the class. Calls the
        pull_from_imdb method using the aformentioned json object
        as input.
        """
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.file",
            "https://www.googleapis.com/auth/drive",
        ]

        creds = ServiceAccountCredentials.from_json_keyfile_name(
            "service_account.json", scope
        )
        client = gspread.authorize(creds)

        sheet_title = "Films " + year
        sheet = client.open_by_url(self.url).worksheet(sheet_title)
        json_sheet = sheet.get_all_records()
        json_object = json.dumps(json_sheet)

        self.spreadsheet_df.append(pd.read_json(json_object))
        self.pull_from_imdb(year)

    def pull_from_imdb(self, year="2018"):
        """
        Scrapes data from imdb.

        Scrapes imdb for movie data for each film using omdb. The year
        is used to differentiate films if duplicates of a film title are
        present in the data (e.g. in the case of film remakes).
        """
        in_df = self.spreadsheet_df[self.year_dict[year]]
        in_df = in_df[in_df.Name != ""]

        file_name = year + "_imdb.json"

        api_key = "?apikey=72bc447a&i="
        url = "http://omdbapi.com/"

        search_df = in_df[["Name", "Release Date", "IMDB ID"]]
        # search_df['search_title'] = search_df['Name'].apply(self.name_search_string)
        # search_df['search_year'] = search_df['Release Date'].apply(self.year_search_string)

        # search_df['counts'] = search_df['search_title'].map(search_df['search_title'].value_counts())
        # search_df['counts'].apply(lambda x: int(x))

        # search_df['search_key'] = np.where(search_df['counts'] == 1, api_key + search_df['search_title'], api_key+search_df['search_title'] + "&y=" + search_df['search_year'])

        search_df["search_key"] = api_key + search_df["IMDB ID"]

        out_list = []

        pbar = tqdm(search_df["search_key"].to_list())

        for key in pbar:
            key_url = url + key
            r = requests.get(key_url)
            json_data = r.json()
            pbar.set_description("Importing " + year)
            out_list.append(json_data)

        json_object = json.dumps(out_list)
        self.imdb_df.append(pd.read_json(StringIO(json_object)))

    def name_search_string(self, title):
        """Convert a string movie title into an omdb compatible search name key."""
        substrings = str(title).split(" ")
        search_string = ""
        for substring in substrings:
            search_string += substring + "+"
        return search_string

    def year_search_string(self, date):
        """Convert a date into an omdb compatible search date key."""
        dates = date.split("/")
        return str(dates[-1])

    def print_files(self):
        """Print the movie dataframe."""
        print(self.input_df)

    def preview_files(self):
        """Print the head of the movie dataframe."""
        print(self.input_df.head())
