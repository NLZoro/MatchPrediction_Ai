import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import glob

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Football AI", page_icon="⚽", layout="centered")

st.title("⚽ Premier League AI Predictor")
st.write("This AI learns from history + recent form + attack/defense ratings.")

# --- CONFIGURATION ---
try:
    API_KEY = st.secrets["FOOTBALL_API_KEY"]
except FileNotFoundError:
    st.error("Secrets file not found. Make sure you have a .streamlit/secrets.toml file!")
    st.stop()

BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}

# --- 1. FUNCTION: LOAD & TRAIN AI ---
@st.cache_data
def load_and_train_model():
    # Load all CSV files
    all_files = glob.glob('*.csv') 
    if not all_files:
        return None, None, None, None, None, None

    df_list = []
    for filename in all_files:
        try:
            data = pd.read_csv(filename)
            data = data[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]
            df_list.append(data)
        except:
            pass

    if not df_list: return None, None, None, None, None, None

    # Merge and Clean
    df = pd.concat(df_list, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values('Date').dropna()
    df['Result'] = df['FTR'].map({'H': 1, 'D': 0, 'A': 2})

    # --- NEW: CALCULATE ATTACK & DEFENSE ---
    team_stats = {} # Stores last 5 games: {'Arsenal': [{'gf': 2, 'ga': 0}, {'gf': 1, 'ga': 1}...]}
    
    df['HomeForm'] = 0; df['AwayForm'] = 0
    df['HomeAtt'] = 0.0; df['AwayAtt'] = 0.0
    df['HomeDef'] = 0.0; df['AwayDef'] = 0.0
    
    for index, row in df.iterrows():
        home, away, result = row['HomeTeam'], row['AwayTeam'], row['FTR']
        h_g, a_g = row['FTHG'], row['FTAG']
        
        # 1. Calculate Stats BEFORE this match (based on history)
        # Default to 0 if no history
        h_recent = team_stats.get(home, [])
        a_recent = team_stats.get(away, [])
        
        # Form (Points)
        h_pts = sum([3 if g['res']=='W' else 1 if g['res']=='D' else 0 for g in h_recent])
        a_pts = sum([3 if g['res']=='W' else 1 if g['res']=='D' else 0 for g in a_recent])
        
        # Attack (Avg Goals Scored)
        h_att = sum([g['gf'] for g in h_recent]) / 5 if len(h_recent) > 0 else 1.0
        a_att = sum([g['gf'] for g in a_recent]) / 5 if len(a_recent) > 0 else 1.0
        
        # Defense (Avg Goals Conceded)
        h_def = sum([g['ga'] for g in h_recent]) / 5 if len(h_recent) > 0 else 1.0
        a_def = sum([g['ga'] for g in a_recent]) / 5 if len(a_recent) > 0 else 1.0
        
        # Save to DF
        df.at[index, 'HomeForm'] = h_pts
        df.at[index, 'AwayForm'] = a_pts
        df.at[index, 'HomeAtt'] = h_att
        df.at[index, 'AwayAtt'] = a_att
        df.at[index, 'HomeDef'] = h_def
        df.at[index, 'AwayDef'] = a_def
        
        # 2. Update History
        if home not in team_stats: team_stats[home] = []
        if away not in team_stats: team_stats[away] = []
        
        # Result for Home
        res_h = 'W' if result == 'H' else 'L' if result == 'A' else 'D'
        team_stats[home].append({'res': res_h, 'gf': h_g, 'ga': a_g})
        
        # Result for Away
        res_a = 'W' if result == 'A' else 'L' if result == 'H' else 'D'
        team_stats[away].append({'res': res_a, 'gf': a_g, 'ga': h_g})
        
        # Keep only last 5
        team_stats[home] = team_stats[home][-5:]
        team_stats[away] = team_stats[away][-5:]

    # Train Model
    all_teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    team_codes = {team: i for i, team in enumerate(all_teams)}
    
    df['HomeCode'] = df['HomeTeam'].map(team_codes)
    df['AwayCode'] = df['AwayTeam'].map(team_codes)
    
    # Train on EVERYTHING
    features = ['HomeCode', 'AwayCode', 'HomeForm', 'AwayForm', 'HomeAtt', 'AwayAtt', 'HomeDef', 'AwayDef']
    X = df[features]
    
    rf_winner = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, df['Result'])
    rf_home_goals = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, df['FTHG'])
    rf_away_goals = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, df['FTAG'])
    
    return rf_winner, rf_home_goals, rf_away_goals, team_codes, team_stats, df

# --- 2. FUNCTION: FETCH LEAGUE TABLE ---
@st.cache_data
def get_league_table():
    url = f"{BASE_URL}/competitions/PL/standings"
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        table = data['standings'][0]['table']
        simplified_table = []
        for team in table:
            simplified_table.append({
                'Pos': team['position'], 'Team': team['team']['shortName'],
                'MP': team['playedGames'], 'Pts': team['points'],
                'GF': team['goalsFor'], 'GA': team['goalsAgainst'], 'GD': team['goalDifference']
            })
        return pd.DataFrame(simplified_table)
    except: return None

# --- LOAD AI ---
with st.spinner("Training AI on Attack & Defense data..."):
    rf_winner, rf_h_goals, rf_a_goals, team_codes, final_stats, df_history = load_and_train_model()

if rf_winner is None:
    st.error("Error: No CSV files found!"); st.stop()

# --- HELPER: GET DATA ---
name_translator = {
    'Wolverhampton': 'Wolves', 'Brighton Hove': 'Brighton', 'Nottingham': "Nott'm Forest",
    'West Ham': 'West Ham', 'Man United': 'Man United', 'Newcastle': 'Newcastle',
    'Sheffield Utd': 'Sheffield United', 'Luton Town': 'Luton', 'Leeds United': 'Leeds'
}

def get_team_data(name):
    clean = name_translator.get(name, name)
    if clean in team_codes:
        code = team_codes[clean]
        recent = final_stats.get(clean, [])
        
        # Calculate live stats
        form = sum([3 if g['res']=='W' else 1 if g['res']=='D' else 0 for g in recent])
        att = sum([g['gf'] for g in recent]) / 5 if len(recent) > 0 else 1.0
        defense = sum([g['ga'] for g in recent]) / 5 if len(recent) > 0 else 1.0
        
        return code, form, att, defense, clean
    else:
        return len(team_codes)//2, 7, 1.0, 1.0, clean

def get_h2h_stats(h_team, a_team):
    if df_history is None: return 0, 0, 0, []
    mask = ((df_history['HomeTeam'] == h_team) & (df_history['AwayTeam'] == a_team)) | \
           ((df_history['HomeTeam'] == a_team) & (df_history['AwayTeam'] == h_team))
    h2h_games = df_history[mask].sort_values('Date', ascending=False).head(5)
    
    h_wins, a_wins, draws = 0, 0, 0
    recent_scores = []
    
    for _, row in h2h_games.iterrows():
        date_str = row['Date'].strftime("%Y-%m-%d")
        recent_scores.append(f"{date_str}: {row['HomeTeam']} {int(row['FTHG'])} - {int(row['FTAG'])} {row['AwayTeam']}")
        if row['Result'] == 1: 
            if row['HomeTeam'] == h_team: h_wins += 1
            else: a_wins += 1
        elif row['Result'] == 2:
            if row['AwayTeam'] == a_team: a_wins += 1
            else: h_wins += 1
        else: draws += 1
    return h_wins, draws, a_wins, recent_scores

# --- MAIN INTERFACE ---
tab1, tab2 = st.tabs(["🔮 Predictions", "🏆 League Table"])

with tab1:
    st.header("Upcoming Matches")
    if st.button("Predict Next 5 Matches"):
        with st.spinner("Analyzing Attack/Defense ratings..."):
            try:
                response = requests.get(f"{BASE_URL}/competitions/PL/matches?status=SCHEDULED", headers=HEADERS)
                response.raise_for_status()
                matches = response.json()['matches']
                
                if not matches: st.warning("No matches found.")
                
                for match in matches[:5]:
                    h_api, a_api = match['homeTeam']['shortName'], match['awayTeam']['shortName']
                    
                    # GET ALL 4 STATS
                    h_code, h_form, h_att, h_def, h_name = get_team_data(h_api)
                    a_code, a_form, a_att, a_def, a_name = get_team_data(a_api)
                    
                    # INPUT: 8 Features now!
                    input_data = pd.DataFrame([[h_code, a_code, h_form, a_form, h_att, a_att, h_def, a_def]], 
                                            columns=['HomeCode', 'AwayCode', 'HomeForm', 'AwayForm', 'HomeAtt', 'AwayAtt', 'HomeDef', 'AwayDef'])
                    
                    winner_code = rf_winner.predict(input_data)[0]
                    h_g = rf_h_goals.predict(input_data)[0]
                    a_g = rf_a_goals.predict(input_data)[0]
                    probs = rf_winner.predict_proba(input_data)[0]
                    
                    # LOGIC FIX
                    if winner_code == 1:
                        winner = f"{h_name} Wins"
                        if h_g <= a_g: h_g = a_g + 1
                    elif winner_code == 2:
                        winner = f"{a_name} Wins"
                        if a_g <= h_g: a_g = h_g + 1
                    else:
                        winner = "Draw"
                        if h_g != a_g: h_g = a_g

                    with st.container():
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col1:
                            try: st.image(match['homeTeam'].get('crest', ''), width=50)
                            except: pass
                            st.write(f"**{h_name}**")
                            st.caption(f"Att: {h_att:.1f} | Def: {h_def:.1f}") # Show Ratings
                        
                        with col2:
                            st.metric("Prediction", winner, f"{h_g} - {a_g}")
                            st.progress(probs[1] if winner_code==1 else probs[2] if winner_code==2 else probs[0])
                            st.caption(f"Conf: {max(probs)*100:.0f}%")
                        
                        with col3:
                            try: st.image(match['awayTeam'].get('crest', ''), width=50)
                            except: pass
                            st.write(f"**{a_name}**")
                            st.caption(f"Att: {a_att:.1f} | Def: {a_def:.1f}") # Show Ratings

                        h_wins, draws, a_wins, past_games = get_h2h_stats(h_name, a_name)
                        with st.expander(f"📜 History & Stats"):
                            st.write(f"**H2H:** {h_name} ({h_wins}) - Draw ({draws}) - {a_name} ({a_wins})")
                            for game in past_games: st.caption(game)
                        st.divider()

            except Exception as e: st.error(f"Error: {e}")

with tab2:
    st.header("Standings")
    df_table = get_league_table()
    if df_table is not None: st.dataframe(df_table.set_index('Pos'), height=600, use_container_width=True)
    
# --- SIDEBAR ---
st.sidebar.header("Manual Predictor")
h_team = st.sidebar.selectbox("Home", list(team_codes.keys()))
a_team = st.sidebar.selectbox("Away", list(team_codes.keys()), index=1)
if st.sidebar.button("Predict"):
    h_c, h_f, h_a, h_d, _ = get_team_data(h_team)
    a_c, a_f, a_a, a_d, _ = get_team_data(a_team)
    input_data = pd.DataFrame([[h_c, a_c, h_f, a_f, h_a, a_a, h_d, a_d]], 
                              columns=['HomeCode', 'AwayCode', 'HomeForm', 'AwayForm', 'HomeAtt', 'AwayAtt', 'HomeDef', 'AwayDef'])
    pred = rf_winner.predict(input_data)[0]
    probs = rf_winner.predict_proba(input_data)[0]
    res = f"{h_team} Wins" if pred == 1 else f"{a_team} Wins" if pred == 2 else "Draw"
    st.sidebar.success(f"{res} ({max(probs)*100:.0f}%)")
    st.sidebar.write(f"**{h_team}**: Att {h_a:.1f}, Def {h_d:.1f}")
    st.sidebar.write(f"**{a_team}**: Att {a_a:.1f}, Def {a_d:.1f}")