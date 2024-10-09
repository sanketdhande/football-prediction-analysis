import matplotlib.pyplot as plt
import pandas as pd
from statsbombpy import sb

# Fetch matches for competition_id=55 and season_id=43
matches = sb.matches(competition_id=55, season_id=43)

# Convert the fetched data into a pandas DataFrame
matches_df = pd.DataFrame(matches)

# Extract relevant columns for visualization
team_matches = pd.concat([matches_df['home_team'], matches_df['away_team']])

# Count the number of matches per team
match_count = team_matches.value_counts()

# Plot the number of matches per team
plt.figure(figsize=(10, 6))
match_count.plot(kind='bar', color='skyblue')
plt.title('Number of Matches Played by Each Team (Competition ID 55, Season ID 43)')
plt.xlabel('Team')
plt.ylabel('Number of Matches')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Extract goals scored by home and away teams
goals_scored = matches_df[['home_score', 'away_score']]

# Plot the distribution of goals scored
plt.figure(figsize=(10, 6))
plt.hist([goals_scored['home_score'], goals_scored['away_score']], bins=range(0, 10), label=['Home Goals', 'Away Goals'], color=['blue', 'orange'], alpha=0.7)
plt.title('Distribution of Goals Scored (Competition ID 55, Season ID 43)')
plt.xlabel('Goals Scored')
plt.ylabel('Number of Matches')
plt.legend()
plt.tight_layout()
plt.show()
