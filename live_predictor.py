import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURATION ---
API_KEY = "d3fc32609fd644b4a81bd82b35bf5366"
BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}

# --- 1. TRAIN THE MODELS (The Brains) ---
# To predict scores, we need training data that INCLUDES scores.
# I've added 'HomeGoals' and 'AwayGoals' columns here.
training_data = {
    'HomeTeam': ['Arsenal', 'Man City', 'Liverpool', 'Chelsea', 'Man Utd', 'Arsenal', 'Liverpool'],
    'AwayTeam': ['Man Utd', 'Chelsea', 'Arsenal', 'Liverpool', 'Man City', 'Chelsea', 'Everton'],
    'HomeRank': [2, 1, 3, 5, 4, 2, 3],
    'AwayRank': [4, 5, 2, 3, 1, 5, 8],
    'HomeGoals': [3, 1, 1, 0, 1, 3, 2],  # Past home goals scored
    'AwayGoals': [1, 0, 1, 0, 2, 1, 0],  # Past away goals scored
    'Result':    [1, 1, 0, 0, 2, 1, 1]   # 1=Home Win, 0=Draw, 2=Away Win
}

# Team Dictionary
team_codes = {
    'Arsenal': 1, 'Aston Villa': 2, 'Bournemouth': 3, 'Brentford': 4,
    'Brighton Hove': 5, 'Chelsea': 6, 'Crystal Palace': 7, 'Everton': 8,
    'Fulham': 9, 'Liverpool': 10, 'Man City': 11, 'Man Utd': 12,
    'Newcastle': 13, 'Nottingham': 14, 'Tottenham': 15, 'West Ham': 16,
    'Wolverhampton': 17, 'Burnley': 18, 'Sheffield Utd': 19, 'Luton Town': 20,
    'Ipswich Town': 21, 'Leicester City': 22, 'Southampton': 23
}

df = pd.DataFrame(training_data)
df['HomeCode'] = df['HomeTeam'].map(team_codes)
df['AwayCode'] = df['AwayTeam'].map(team_codes)
features = df[['HomeCode', 'AwayCode', 'HomeRank', 'AwayRank']]

# --- CREATING 3 SEPARATE MODELS ---
# Model 1: Predicts the Winner
rf_winner = RandomForestClassifier(n_estimators=50, random_state=42)
rf_winner.fit(features, df['Result'])

# Model 2: Predicts Home Goals
rf_home_goals = RandomForestClassifier(n_estimators=50, random_state=42)
rf_home_goals.fit(features, df['HomeGoals'])

# Model 3: Predicts Away Goals
rf_away_goals = RandomForestClassifier(n_estimators=50, random_state=42)
rf_away_goals.fit(features, df['AwayGoals'])

# --- 2. GET REAL DATA ---
def get_rankings():
    print("Fetching live League Table...")
    try:
        data = requests.get(f"{BASE_URL}/competitions/PL/standings", headers=HEADERS).json()
        standings = data['standings'][0]['table']
        ranks = {t['team']['shortName']: t['position'] for t in standings}
        return ranks
    except:
        print("Could not fetch rankings. Using defaults.")
        return {}

current_ranks = get_rankings()

# --- 3. PREDICT ---
print("\nFetching upcoming matches...")
matches = requests.get(f"{BASE_URL}/competitions/PL/matches?status=SCHEDULED", headers=HEADERS).json()['matches']

print(f"\n--- PREDICTED SCORE LINES ---\n")

for match in matches[:5]: # Predict next 5 games
    home = match['homeTeam']['shortName']
    away = match['awayTeam']['shortName']
    
    if home in team_codes and away in team_codes:
        # Prepare Data
        h_code = team_codes[home]
        a_code = team_codes[away]
        h_rank = current_ranks.get(home, 10)
        a_rank = current_ranks.get(away, 10)
        
        match_input = pd.DataFrame([[h_code, a_code, h_rank, a_rank]], 
                                  columns=['HomeCode', 'AwayCode', 'HomeRank', 'AwayRank'])

        # ASK ALL 3 MODELS
        pred_result = rf_winner.predict(match_input)[0]
        pred_h_goals = rf_home_goals.predict(match_input)[0]
        pred_a_goals = rf_away_goals.predict(match_input)[0]
        
        # Format the text
        winner_text = "Draw"
        if pred_result == 1: winner_text = home
        if pred_result == 2: winner_text = away
        
        print(f"{home} vs {away}")
        print(f"   Winner: {winner_text}")
        print(f"   Score:  {pred_h_goals} - {pred_a_goals}")
        print("-" * 30)