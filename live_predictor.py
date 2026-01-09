import matplotlib.pyplot as plt
import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import glob

# --- CONFIGURATION ---
API_KEY = "d3fc32609fd644b4a81bd82b35bf5366"
BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}

def add_form(df, num_games=5):
    print("Calculating team form (last 5 games)...")
    team_results = {}
    df['HomeForm'] = 0
    df['AwayForm'] = 0
    
    for index, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        result = row['FTR']
        
        # Calculate current form from past 5 games
        h_score = sum([3 if r=='W' else 1 if r=='D' else 0 for r in team_results.get(home, [])])
        a_score = sum([3 if r=='W' else 1 if r=='D' else 0 for r in team_results.get(away, [])])
        
        df.at[index, 'HomeForm'] = h_score
        df.at[index, 'AwayForm'] = a_score
        
        # Update history
        if home not in team_results: team_results[home] = []
        if away not in team_results: team_results[away] = []
        
        if result == 'H':
            team_results[home].append('W'); team_results[away].append('L')
        elif result == 'A':
            team_results[home].append('L'); team_results[away].append('W')
        else:
            team_results[home].append('D'); team_results[away].append('D')
            
        team_results[home] = team_results[home][-num_games:]
        team_results[away] = team_results[away][-num_games:]
        
    return df, team_results

# --- 1. LOAD & MERGE HISTORY ---
print("--- LOADING HISTORY ---")
all_files = glob.glob('*.csv') 

if not all_files:
    print("ERROR: No CSV files found!")
    exit()

print(f"Found {len(all_files)} files: {all_files}")

df_list = []
for filename in all_files:
    try:
        data = pd.read_csv(filename)
        # FIX: We MUST include 'Date' here so we can sort later!
        data = data[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]
        df_list.append(data)
    except Exception as e:
        print(f"Skipping {filename}: {e}")

if df_list:
    df = pd.concat(df_list, ignore_index=True)
else:
    print("Error: No valid data loaded.")
    exit()

# --- NEW: SORT BY DATE ---
# This ensures we calculate "Form" in the correct chronological order
print("Sorting games by date...")
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.sort_values('Date')
df = df.dropna() # Remove any bad rows

# Map Results
df['Result'] = df['FTR'].map({'H': 1, 'D': 0, 'A': 2})

# Create Team Codes
all_teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
team_codes = {team: i for i, team in enumerate(all_teams)}
avg_team_code = len(team_codes) // 2

df['HomeCode'] = df['HomeTeam'].map(team_codes)
df['AwayCode'] = df['AwayTeam'].map(team_codes)

# --- ADD FORM STATS ---
df, final_form = add_form(df)

# --- 2. TRAIN MODELS ---
print(f"Training on {len(df)} matches...")
X = df[['HomeCode', 'AwayCode', 'HomeForm', 'AwayForm']]
rf_winner = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, df['Result'])
rf_home_goals = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, df['FTHG'])
rf_away_goals = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, df['FTAG'])

# ... (Keep all imports and functions at the top exactly the same) ...
# ... (Keep Step 1: LOAD HISTORY exactly the same) ...
# ... (Keep Step 2: TRAIN MODELS exactly the same) ...

# --- 3. PREDICT LIVE & SAVE REPORT ---
name_translator = {
    'Wolverhampton': 'Wolves', 'Brighton Hove': 'Brighton', 'Nottingham': "Nott'm Forest",
    'West Ham': 'West Ham', 'Man United': 'Man United', 'Newcastle': 'Newcastle',
    'Sheffield Utd': 'Sheffield United', 'Luton Town': 'Luton', 'Leeds United': 'Leeds'
}

def get_team_info(api_name):
    clean_name = name_translator.get(api_name, api_name)
    if clean_name in team_codes:
        code = team_codes[clean_name]
        form = sum([3 if r=='W' else 1 if r=='D' else 0 for r in final_form.get(clean_name, [])])
        return code, form, clean_name, False
    else:
        return avg_team_code, 7, clean_name, True

print("\nFetching upcoming matches...")
try:
    matches = requests.get(f"{BASE_URL}/competitions/PL/matches?status=SCHEDULED", headers=HEADERS).json()['matches']
except:
    print("Connection Error.")
    exit()

print(f"\n--- GENERATING REPORT ---\n")

# We will store all data here to save to Excel later
report_data = []

for match in matches[:5]: # Predict next 5 matches
    h_api, a_api = match['homeTeam']['shortName'], match['awayTeam']['shortName']
    
    h_code, h_form, h_name, h_miss = get_team_info(h_api)
    a_code, a_form, a_name, a_miss = get_team_info(a_api)
    
    match_data = pd.DataFrame([[h_code, a_code, h_form, a_form]], 
                              columns=['HomeCode', 'AwayCode', 'HomeForm', 'AwayForm'])

    # Predict
    winner_code = rf_winner.predict(match_data)[0]
    h_goals = rf_home_goals.predict(match_data)[0]
    a_goals = rf_away_goals.predict(match_data)[0]
    probs = rf_winner.predict_proba(match_data)[0] # Get confidence %
    
    winner = "DRAW"
    if winner_code == 1: winner = h_name
    if winner_code == 2: winner = a_name
    
    # 1. Print to screen (for you to see now)
    print(f"{h_name} vs {a_name} -> Pred: {winner} ({h_goals}-{a_goals})")
    
    # 2. Add to Report List
    report_data.append({
        'Date': match['utcDate'][:10],
        'Home Team': h_name,
        'Away Team': a_name,
        'Predicted Winner': winner,
        'Predicted Score': f"{h_goals}-{a_goals}",
        'Confidence (Home)': f"{probs[1]*100:.1f}%",
        'Confidence (Draw)': f"{probs[0]*100:.1f}%",
        'Confidence (Away)': f"{probs[2]*100:.1f}%"
    })

    # 3. SAVE THE CHART (Don't show it pop-up)
    plt.figure(figsize=(5, 3))
    plt.bar([h_name, 'Draw', a_name], [probs[1], probs[0], probs[2]], color=['green', 'gray', 'red'])
    plt.title(f"Pred: {winner}")
    plt.ylim(0, 1)
    
    # Save file like "Arsenal_vs_Liverpool.png"
    filename = f"{h_name}_vs_{a_name}.png"
    plt.savefig(filename) 
    plt.close() # Close memory
    print(f"   -> Chart saved as {filename}")

# --- 4. SAVE EXCEL REPORT ---
report_df = pd.DataFrame(report_data)
report_df.to_csv("Weekly_Predictions.csv", index=False)
print("\nSUCCESS: Saved 'Weekly_Predictions.csv' and chart images!")