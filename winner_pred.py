import json
from statsbombpy import sb
import pandas as pd

# Fetch all competition data
competitions = sb.competitions()
matches = sb.matches(competition_id=2, season_id=27)

# # Convert the competitions data to a list of dictionaries (optional if it is a DataFrame)
# # In case it's a Pandas DataFrame, convert it to a dictionary first
# if isinstance(matches, pd.DataFrame):
#     competitions = matches.to_dict(orient='records')
#
# # Write the data to a JSON file
# with open('competitions_data.json', 'w') as json_file:
#     json.dump(competitions, json_file, indent=4)
#
# print("Competitions data has been written to 'competitions_data.json'.")
# pd.read_html('https://en.wikipedia.org/wiki/2022_FIFA_World_Cup')
