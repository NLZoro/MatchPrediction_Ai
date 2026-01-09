import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import glob

# --- PAGE CONFIGURATION (Make it look like a real site) ---
st.set_page_config(page_title="Football AI", page_icon="⚽", layout="centered")

st.title("⚽ Premier League AI Predictor")
st.write("This AI learns from 4 years of history + recent form to predict upcoming matches.")

# --- CONFIGURATION ---
API_KEY = "d3fc32609fd644b4a81bd82b35bf5366" # Your Key
BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}

# --- CACHING (SPEED BOOST) ---
# We use @st.cache_data so the AI doesn't re-train every time you click a button.
@st.cache_data
def load_and_train_model():
    # 1. LOAD DATA
    all_files = glob.glob('*.csv') 
    if not all_files:
        return None, None, None, None

    df_list = []
    for filename in all_files:
        try:
            data = pd.read_csv(filename)
            data = data[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]
            df_list.append(data)
        except:
            pass

    if not df_list: return None, None, None, None

    df = pd.concat(df_list, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values('Date').dropna()
    df['Result'] = df['FTR'].map({'H': 1, 'D': 0, 'A': 2})

    # 2. CALCULATE FORM
    team_results = {}
    df['HomeForm'] = 0; df['AwayForm'] = 0
    
    for index, row in df.iterrows():
        home, away, result = row['HomeTeam'], row['AwayTeam'], row['FTR']
        h_score = sum([3 if r=='W' else 1 if r=='D' else 0 for r in team_results.get(home, [])])
        a_score = sum([3 if r=='W' else 1 if r=='D' else 0 for r in team_results.get(away, [])])
        df.at[index, 'HomeForm'] = h_score
        df.at[index, 'AwayForm'] = a_score
        
        if home not in team_results: team_results[home] = []
        if away not in team_results: team_results[away] = []
        
        if result == 'H': team_results[home].append('W'); team_results[away].append('L')
        elif result == 'A': team_results[home].append('L'); team_results[away].append('W')
        else: team_results[home].append('D'); team_results[away].append('D')
            
        team_results[home] = team_results[home][-5:]
        team_results[away] = team_results[away][-5:]

    # 3. TRAIN
    all_teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    team_codes = {team: i for i, team in enumerate(all_teams)}
    
    df['HomeCode'] = df['HomeTeam'].map(team_codes)
    df['AwayCode'] = df['AwayTeam'].map(team_codes)
    
    X = df[['HomeCode', 'AwayCode', 'HomeForm', 'AwayForm']]
    rf_winner = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, df['Result'])
    rf_home_goals = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, df['FTHG'])
    rf_away_goals = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, df['FTAG'])
    
    return rf_winner, rf_home_goals, rf_away_goals, team_codes, team_results

# Load the AI
with st.spinner("Training AI on historical data..."):
    rf_winner, rf_h_goals, rf_a_goals, team_codes, final_form = load_and_train_model()

if rf_winner is None:
    st.error("Error: No CSV files found! Please move your E0.csv files to this folder.")
    st.stop()

# --- PREDICTION INTERFACE ---
name_translator = {
    'Wolverhampton': 'Wolves', 'Brighton Hove': 'Brighton', 'Nottingham': "Nott'm Forest",
    'West Ham': 'West Ham', 'Man United': 'Man United', 'Newcastle': 'Newcastle',
    'Sheffield Utd': 'Sheffield United', 'Luton Town': 'Luton', 'Leeds United': 'Leeds'
}

def get_team_data(name):
    clean = name_translator.get(name, name)
    if clean in team_codes:
        code = team_codes[clean]
        form = sum([3 if r=='W' else 1 if r=='D' else 0 for r in final_form.get(clean, [])])
        return code, form, clean
    else:
        return len(team_codes)//2, 7, clean # Average defaults

if st.button("🔮 Predict Next 5 Live Matches"):
    try:
        matches = requests.get(f"{BASE_URL}/competitions/PL/matches?status=SCHEDULED", headers=HEADERS).json()['matches']
        
        for match in matches[:5]:
            h_api = match['homeTeam']['shortName']
            a_api = match['awayTeam']['shortName']
            
            h_code, h_form, h_name = get_team_data(h_api)
            a_code, a_form, a_name = get_team_data(a_api)
            
            # Predict
            input_data = pd.DataFrame([[h_code, a_code, h_form, a_form]], 
                                      columns=['HomeCode', 'AwayCode', 'HomeForm', 'AwayForm'])
            
            winner_code = rf_winner.predict(input_data)[0]
            h_g = rf_h_goals.predict(input_data)[0]
            a_g = rf_a_goals.predict(input_data)[0]
            probs = rf_winner.predict_proba(input_data)[0]
            
            winner = "DRAW"
            if winner_code == 1: winner = f"{h_name} Wins"
            if winner_code == 2: winner = f"{a_name} Wins"
            
            # --- DISPLAY CARD ---
            with st.container():
                st.subheader(f"{h_name} vs {a_name}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Home Form", f"{h_form}/15")
                col2.metric("Prediction", winner, f"{h_g} - {a_g}")
                col3.metric("Away Form", f"{a_form}/15")
                
                # Chart
                fig, ax = plt.subplots(figsize=(5,2))
                ax.bar([h_name, 'Draw', a_name], [probs[1], probs[0], probs[2]], color=['#2ecc71', '#95a5a6', '#e74c3c'])
                ax.set_ylim(0,1)
                st.pyplot(fig)
                st.divider()
                
    except Exception as e:
        st.error(f"Error fetching live matches: {e}")

# --- MANUAL PREDICTOR (Bonus) ---
st.sidebar.header("Manual Predictor")
h_team = st.sidebar.selectbox("Home Team", list(team_codes.keys()))
a_team = st.sidebar.selectbox("Away Team", list(team_codes.keys()), index=1)

if st.sidebar.button("Predict Custom Match"):
    h_code = team_codes[h_team]
    a_code = team_codes[a_team]
    # Use current form from memory
    h_form = sum([3 if r=='W' else 1 if r=='D' else 0 for r in final_form.get(h_team, [])])
    a_form = sum([3 if r=='W' else 1 if r=='D' else 0 for r in final_form.get(a_team, [])])
    
    input_data = pd.DataFrame([[h_code, a_code, h_form, a_form]], columns=['HomeCode', 'AwayCode', 'HomeForm', 'AwayForm'])
    pred = rf_winner.predict(input_data)[0]
    
    res = "Draw"
    if pred == 1: res = f"{h_team} Wins"
    if pred == 2: res = f"{a_team} Wins"
    
    st.sidebar.success(f"Result: {res}")