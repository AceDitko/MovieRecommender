import datetime
from pathlib import Path
import pandas as pd
import json
import tqdm
from io import StringIO
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials

pd.options.mode.chained_assignment = None  # default='warn'

# repo: interrogate
#   id: interrogate
#   args: quiet, fail under 95
#
# repo: pydocstyle
#   id: pydocstyle
#
# repo: yamllint
#   id: yamllint
#   args: -s, -c=.yamllint (we don't know what these are but carl copied them)
# test

start_year = 2018
url = "https://docs.google.com/spreadsheets/d/1w-LyhsGUqNFiFqsF2QVgfA5MARTMRWFxMbG-FiWQf9c/edit#gid=621622277"


def getdays(date):
    """
    Creates an integer number representing the difference between
    a given date and today's date.
    """

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


def load_dataset() -> pd.DataFrame:
    """
    Loads movie dataset into input_df.

    If movie_db.json exists, reads the file and returns it as a df.
    If movie_db.json does not exist, creates the file using imdb info pulled
    using the omdb api and returns it as a df.
    """

    if Path("movie_db.json").is_file():
        return pd.read_json("movie_db.json")
    else:
        return create_moviedb()


def create_moviedb() -> pd.DataFrame:
    """Creates the movie dataset if it does not exist.

    pull_from_sheet() creates a single dataframe containing all user data
    pulled from the online google sheet.
    pull_from_imdb() queries imdb using this dataframe as input and returns
    a single dataframe containing all of the imdb data on the submitted movies
    joined with some of the user provided information contained in the google
    sheet.
    """

    return pull_from_imdb(pull_from_sheet)


def pull_from_sheet() -> pd.DataFrame:
    """
    Pulls the data (created by user input) from the google sheet
    and concatenates it into a single dataframe which is then returned.
    """

    # Authenticate with the google sheet
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

    # Obtain the current year and number of sheets that should exist
    today = datetime.date.today()
    current_year = [int(i) for i in today.strftime("%d/%m/%Y").split("/")][2]

    year_diff = current_year - start_year

    # Create the dataframes of each year's data and append to list
    spreadsheet_df_list = []

    for i in range(0, year_diff + 1):
        year = int(start_year) + 1

        sheet_title = "Films " + year
        sheet = client.open_by_url(url).worksheet(sheet_title)
        json_sheet = sheet.get_all_records()
        json_object = json.dumps(json_sheet)

        spreadsheet_df_list.append(pd.read_json(json_object))

    # Concatenate the list of dfs into a single dataframe
    return pd.concat(spreadsheet_df_list)


def pull_from_imdb(*, in_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scrapes imdb for movie data for each film using omdb.

    Uses the IMDB ID stored in the google sheet to query omdb and pull data
    from imdb, which is stored in a single output dataframe. Once this dataframe
    is created, relevant information from the google sheet data is added by
    calling update_imdb_df() and then this final combined dataframe is returned.
    """

    in_df = in_df[in_df.Name != ""]

    api_key = "?apikey=72bc447a&i="
    omdb_url = "http://omdbapi.com/"

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
        key_url = omdb_url + key
        r = requests.get(key_url)
        json_data = r.json()
        pbar.set_description("Importing imdb data...")
        out_list.append(json_data)

    json_object = json.dumps(out_list)
    out_df = pd.read_json(StringIO(json_object))
    return update_imdb_df(out_df, in_df)


def update_imdb_df(
    *, imdb_df: pd.DataFrame, spreadsheet_df: pd.DataFrame
) -> pd.DataFrame:
    """Adds additional fields from spreadsheet to df created from omdb data."""

    imdb_df["Format"] = spreadsheet_df["Format"].values
    imdb_df["Viewing Date"] = spreadsheet_df["Viewing Data"].values
    imdb_df["Days to View"] = spreadsheet_df["Days to View"].values
    imdb_df["Days Since Release"] = spreadsheet_df["Release Date"].apply(getdays())
    imdb_df["True Rating"] = spreadsheet_df["True Rating"].apply(lambda x: int(x))
    imdb_df = imdb_df.drop(
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
    return imdb_df


def name_search_string(self, title):
    """Converts a string movie title into an omdb compatible search name key."""
    substrings = str(title).split(" ")
    search_string = ""
    for substring in substrings:
        search_string += substring + "+"
    return search_string


def year_search_string(self, date):
    """Converts a date into an omdb compatible search date key."""
    dates = date.split("/")
    return str(dates[-1])
