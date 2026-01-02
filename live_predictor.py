import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import glob  # Tool to find multiple files

# --- CONFIGURATION ---
API_KEY = "d3fc32609fd644b4a81bd82b35bf5366"
BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}

# --- 1. LOAD & MERGE ALL SEASONS ---
print("--- LOADING HISTORY ---")
# This grabs ANY file ending in .csv (E0.csv, E0 (1).csv, etc.)
all_files = glob.glob('*.csv') 

if not all_files:
    print("ERROR: No CSV files found! Please move your E0.csv files into this folder.")
    exit()

print(f"Found {len(all_files)} season files: {all_files}")

df_list = []
for filename in all_files:
    try:
        data = pd.read_csv(filename)
        # Keep only necessary columns
        data = data[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]
        df_list.append(data)
    except:
        print(f"Skipping {filename} (File might be empty or wrong format)")

# Combine all seasons into one big table
if df_list:
    df = pd.concat(df_list, ignore_index=True)
else:
    print("Error: Could not load any data.")
    exit()

# Map Results to Numbers
df['Result'] = df['FTR'].map({'H': 1, 'D': 0, 'A': 2})

# Learn every team name from the last 4 years
all_teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).dropna().unique())
team_codes = {team: i for i, team in enumerate(all_teams)}

# Create the "Average Team" code for generic backups
avg_team_code = len(team_codes) // 2

# Convert Names to Numbers
df['HomeCode'] = df['HomeTeam'].map(team_codes)
df['AwayCode'] = df['AwayTeam'].map(team_codes)

# Clean data
df = df.dropna()

print(f"Successfully loaded {len(df)} matches from {len(all_files)} seasons!")
print(f"AI Brain now knows {len(team_codes)} unique teams (including Leeds, Burnley, etc.)")


# --- 2. TRAIN MODELS ---
print("Training AI models...")
X = df[['HomeCode', 'AwayCode']]
rf_winner = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, df['Result'])
rf_home_goals = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, df['FTHG'])
rf_away_goals = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, df['FTAG'])

# --- 3. NAME TRANSLATOR ---
name_translator = {
    'Wolverhampton': 'Wolves',
    'Brighton Hove': 'Brighton',
    'Nottingham': "Nott'm Forest",
    'West Ham': 'West Ham',
    'Man United': 'Man United',
    'Newcastle': 'Newcastle',
    'Sheffield Utd': 'Sheffield United',
    'Luton Town': 'Luton',
    'Leeds United': 'Leeds'  # Now this will actually work!
}

def get_team_code(api_name):
    clean_name = name_translator.get(api_name, api_name)
    if clean_name in team_codes:
        return team_codes[clean_name], clean_name, False
    else:
        return avg_team_code, clean_name, True

# --- 4. PREDICT ---
print("\nFetching upcoming matches...")
try:
    matches = requests.get(f"{BASE_URL}/competitions/PL/matches?status=SCHEDULED", headers=HEADERS).json()['matches']
except:
    print("Error connecting to internet.")
    exit()

print(f"\n--- PREDICTIONS (Based on {len(df)} historical games) ---\n")

for match in matches[:10]: 
    h_api = match['homeTeam']['shortName']
    a_api = match['awayTeam']['shortName']
    
    h_code, h_name, h_missing = get_team_code(h_api)
    a_code, a_name, a_missing = get_team_code(a_api)
    
    match_data = pd.DataFrame([[h_code, a_code]], columns=['HomeCode', 'AwayCode'])

    pred_winner = rf_winner.predict(match_data)[0]
    pred_h = rf_home_goals.predict(match_data)[0]
    pred_a = rf_away_goals.predict(match_data)[0]
    
    winner_text = "DRAW"
    if pred_winner == 1: winner_text = f"{h_name} WINS"
    if pred_winner == 2: winner_text = f"{a_name} WINS"
    
    # Add warning only if we truly don't know the team
    warning = ""
    if h_missing or a_missing:
        warning = " (⚠️ Unknown Team)"
    
    print(f"{h_name} vs {a_name}")
    print(f"   Prediction: {winner_text}{warning}")
    print(f"   Score:      {pred_h} - {pred_a}")
    print("-" * 30)