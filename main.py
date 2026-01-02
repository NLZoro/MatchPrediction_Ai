import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --- STEP 1: CREATE THE DATA ---
# We represent: 1=Home Win, 0=Draw, 2=Away Win
# Rankings: Lower number is better (e.g., Rank 1 is best)
data = {
    'HomeTeam': ['Arsenal', 'Chelsea', 'Liverpool', 'Arsenal', 'ManUtd', 'Chelsea'],
    'AwayTeam': ['Chelsea', 'ManUtd', 'ManUtd', 'Liverpool', 'Arsenal', 'Liverpool'],
    'HomeTeamRank': [1, 5, 2, 1, 4, 5],
    'AwayTeamRank': [5, 4, 4, 2, 1, 2],
    'Result': [1, 0, 1, 0, 2, 2] 
}

# Turn this dictionary into a clear table
df = pd.DataFrame(data)

# --- STEP 2: CONVERT TEAMS TO NUMBERS ---
# Computers can't read words like "Arsenal", so we give each team a specific ID number.
team_codes = {'Arsenal': 1, 'Chelsea': 2, 'Liverpool': 3, 'ManUtd': 4}

df['HomeCode'] = df['HomeTeam'].map(team_codes)
df['AwayCode'] = df['AwayTeam'].map(team_codes)

# --- STEP 3: TRAIN THE MODEL ---
# X = The input data (Team ID numbers + Their Rankings)
X = df[['HomeCode', 'AwayCode', 'HomeTeamRank', 'AwayTeamRank']]

# y = The target (The actual match result)
y = df['Result']

# Create the Random Forest model
rf = RandomForestClassifier(n_estimators=50, random_state=42)

# Train the model (This is where the AI "learns")
rf.fit(X, y)

# --- STEP 4: MAKE A PREDICTION ---
print("---------------------------------")
print("   FOOTBALL MATCH PREDICTOR 3000")
print("---------------------------------")

# Let's predict a match: Arsenal (Home) vs ManUtd (Away)
h_team = 'Arsenal'
a_team = 'ManUtd'

# We must manually provide the current rankings for this specific prediction
h_rank = 1 
a_rank = 4

# Convert names to the hidden numbers
h_code = team_codes[h_team]
a_code = team_codes[a_team]

# Predict!
prediction = rf.predict([[h_code, a_code, h_rank, a_rank]])

# Decode the result
print(f"Match: {h_team} vs {a_team}")
if prediction[0] == 1:
    print("AI Predicts: HOME WIN (Arsenal)")
elif prediction[0] == 2:
    print("AI Predicts: AWAY WIN (ManUtd)")
else:
    print("AI Predicts: DRAW")
print("---------------------------------")