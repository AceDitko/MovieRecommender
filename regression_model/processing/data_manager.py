"""Load/manipulate/process input data."""
import datetime
from pathlib import Path
import pandas as pd
import json
import tqdm
from io import StringIO
import requests
from sklearn.pipeline import Pipeline
import joblib
import typing as t
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient import discovery

# from regression_model import __version__ as _version
from regression_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

pd.options.mode.chained_assignment = None  # default='warn'


start_year = 2018
url = (
    "https://docs.google.com/spreadsheets/d/1w-"
    "LyhsGUqNFiFqsF2QVgfA5MARTMRWFxMbG-FiWQf9c/edit#gid=621622277"
)
SPREADSHEET_ID = "1w-LyhsGUqNFiFqsF2QVgfA5MARTMRWFxMbG-FiWQf9c"


def get_version() -> str:
    """Read the application version from VERSION file.

    This only exists because setuptools are being a nightmare. Will be removed
    after migration to poetry.

    Returns:
    version(str): The version number of the package
    """
    with open("../VERSION") as f:
        return f.read().strip()


def load_dataset(file_name: str) -> pd.DataFrame:
    """
    Load movie dataset into input_df.

    If movie_db.json exists, reads the file and returns it as a df.
    If movie_db.json does not exist, creates the file using imdb info pulled
    using the omdb api and returns it as a df.

    Parameters:
    file_name(str): File name of data to load

    Returns:
    movie_db(pd.DataFrame): If data is already saved as a .json file, this is
                            read and returned as a df. If the file does not
                            exist, the data will be created via create_moviedb
                            and subsequent functions.
                            If file_name matches the name of the test_data_file
                            in the config, only ten rows of data will be
                            returned in order to create the pytest data file.
    """
    path = Path("f{DATASET_DIR}/{file_name}")
    if path.is_file():
        return pd.read_json(path)
    else:
        if file_name == config.app_config.test_data_file:
            return create_moviedb(file_name, True)
        else:
            return create_moviedb(file_name)


def create_moviedb(file_name: str, test_file: bool = False) -> pd.DataFrame:
    """Create the movie dataset if it does not exist.

    pull_from_sheet() creates a single dataframe containing all user data
    pulled from the online google sheet.
    pull_from_imdb() queries imdb using this dataframe as input and returns
    a single dataframe containing all of the imdb data on the submitted movies
    joined with some of the user provided information contained in the google
    sheet.
    When create_moviedb() creates the dataset, it saves it to the config
    specified file name in the DATASET_DIR.

    Parameters:
    file_name(str): Name for final saved dataset
    test_file(bool): If true only 10 rows are returned to create pytest data

    Returns:
    moviedb(pd.DataFrame): Final output df containing combination of data from
                            omdb and google sheet
    """
    moviedb = pull_from_imdb(pull_from_sheet(test_file))
    moviedb.to_json(f"{DATASET_DIR}/{file_name}")
    return moviedb


def pull_from_sheet(test_file: bool = False) -> pd.DataFrame:
    """Pull input data and concatenate into single sheet.

    Parameters:
    test_file(bool): If true only 10 rows are returned to create pytest data

    Returns:
    final_df(pd.DataFrame): Dataframe containing raw data from google sheet
    """
    # Authenticate with google sheet
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive",
    ]

    creds = ServiceAccountCredentials.from_json_keyfile_name(
        "service_account.json", scope
    )

    try:
        service = discovery.build("sheets", "v4", credentials=creds)
        in_sheets = service.spreadsheets()
    except FileNotFoundError:
        raise FileNotFoundError("Service account key not found")

    # Obtain the current year and number of sheets that should exist
    today = datetime.date.today()
    current_year = [int(i) for i in today.strftime("%d/%m/%Y").split("/")][2]

    year_diff = current_year - start_year

    # Create the dataframes of each year's data and append to list
    spreadsheet_df_list = []

    for i in range(0, year_diff + 1):
        year = int(start_year) + i
        sheet_title = "Films " + str(year)

        # Get values of each sheet using sheet_title
        result = (
            in_sheets.values()
            .get(spreadsheetId=SPREADSHEET_ID, range=f"{sheet_title}!A1:N101")
            .execute()
        )

        # First list item is columns, rest is data
        spreadsheet_df_list.append(
            pd.DataFrame(result["values"][1:], columns=result["values"][0])
        )

    # Close the connection to the google sheet
    service.close()

    # Concatenate the list of dfs into a single dataframe
    final_df = pd.concat(spreadsheet_df_list).dropna()

    # If making test file, only return 10 random rows
    if test_file:
        return final_df.sample(10, random_state=42).reset_index(drop=True)
    else:
        return final_df


def pull_from_imdb(in_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scrape imdb for movie data for each film using omdb.

    Uses the IMDB ID stored in the google sheet to query omdb and pull data
    from imdb, which is stored in a single output dataframe. Once this dataframe
    is created, relevant information from the google sheet data is added by
    calling update_imdb_df() and then this final combined dataframe is returned.

    Parameters:
    in_df(pd.DataFrame): df scraped from google sheet containing API search key

    Returns:
    updated_df(pd.DataFrame): Updated version of out_df created by this function
                                including modifications from update_imdb_df()
    """
    in_df = in_df[in_df.Name != ""]

    api_key = "?apikey=72bc447a&i="
    omdb_url = "http://omdbapi.com/"

    search_df = in_df[["Name", "Release Date", "IMDB ID"]]

    search_df["search_key"] = api_key + search_df["IMDB ID"]

    out_list = []

    pbar = tqdm.tqdm(search_df["search_key"].to_list())

    for key in pbar:
        key_url = omdb_url + key
        r = requests.get(key_url)
        json_data = r.json()
        pbar.set_description("Importing imdb data...")
        out_list.append(json_data)

    json_object = json.dumps(out_list)
    out_df = pd.read_json(StringIO(json_object))
    return update_imdb_df(out_df, in_df)


def update_imdb_df(imdb_df: pd.DataFrame, spreadsheet_df: pd.DataFrame) -> pd.DataFrame:
    """Add additional fields from spreadsheet to df created from omdb data.

    Parameters:
    imdb_df(pd.DataFrame): df containing unprocessed data from omdb api
    spreadsheet_df(pd.DataFrame): df containing data from google sheet

    Returns:
    imdb_df(pd.DataFrame): Modified input imdb_df with the following changes:
                            - Format, Viewing Date, Days to View copied from
                            spreadsheet_df
                            - Days Since Release added from applying get_days()
                            to spreadsheet_df's release date column
                            - True rating created from spreadsheet_df
                            - imdbVotes, Response, Type, Website, Language, DVD,
                            and Country fields dropped
    """
    imdb_df["Format"] = spreadsheet_df["Format"].values
    imdb_df["Viewing Date"] = spreadsheet_df["Viewing Date"].values
    imdb_df["Days to View"] = spreadsheet_df["Days to View"].values
    imdb_df["Days Since Release"] = spreadsheet_df["Release Date"].apply(
        lambda x: get_days(x)
    )
    imdb_df["True Rating"] = spreadsheet_df["True Rating"].apply(lambda x: int(x))
    imdb_df = imdb_df.drop(
        columns=[
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


def get_days(date: str) -> int:
    """Calculate difference in days between today and given date.

    Parameters:
    date(int): A date string of format dd/mm/yyyy

    Returns:
    int: Number of whole days passed since input date
    """
    if "/" not in date:
        raise ValueError("Date not '/' delimited. Format should be dd/mm/yyyy")

    str_past_dates = date.split("/")
    if len(str_past_dates) > 3:
        raise ValueError("Date format incorrect. Should be dd/mm/yyyy")

    past_dates = [int(i) for i in str_past_dates]

    if past_dates[0] < 1 or past_dates[0] > 31:
        raise ValueError("Day value of date must be between 1 and 31 inclusive")
    if past_dates[1] < 1 or past_dates[1] > 12:
        raise ValueError("Month value of date must be between 1 and 12 inclusive")
    if past_dates[2] < 1888:
        raise ValueError("Year value of date earlier than first ever film in 1888")

    today = datetime.date.today()
    today_date = today.strftime("%d/%m/%Y")
    str_today_dates = today_date.split("/")
    today_dates = [int(i) for i in str_today_dates]

    d0 = datetime.date(past_dates[2], past_dates[1], past_dates[0])
    d1 = datetime.date(today_dates[2], today_dates[1], today_dates[0])
    delta = d1 - d0
    return delta.days


def save_pipeline(pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.

    Save the versioned model, and overwrite any previously saved models. This
    ensures that when the package is published, there is only one trained model
    that can be called.
    """
    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{get_version()}.pk1"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(file_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(file_name: str) -> Pipeline:
    """Load the persisted pipeline.

    Parameters:
    file_name(str): The name of the saved pipeline file

    Returns:
    trained_model(Pipeline): Returns the pipeline loaded from file_name
    """
    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filenmae=file_path)
    return trained_model


def remove_old_pipelines(files_to_keep: t.List[str]) -> None:
    """Remove old model pipelines.

    This is to ensure there is a simple one-to-one mapping between the package
    version and the model version to be imported and used by other applications.

    Parameters:
    files_to_keep(t.List[str]): List of string filenames to persist
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
