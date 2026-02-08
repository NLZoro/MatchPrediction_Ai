import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import glob
import numpy as np 

# --- 0. HELPER: RADAR CHART FUNCTION ---
def create_radar_chart(team_name, stats, color):
    labels = ['Form', 'Attack', 'Defense']
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] 
    stats += stats[:1]   
    
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    fig.patch.set_alpha(0) 
    ax.patch.set_alpha(0)  
    ax.fill(angles, stats, color=color, alpha=0.4) 
    ax.plot(angles, stats, color=color, linewidth=2)
    ax.set_ylim(0, 10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color='white', size=10)
    ax.set_yticklabels([])
    ax.grid(color='gray', alpha=0.3) 
    ax.spines['polar'].set_visible(False)
    ax.set_title(team_name, size=14, color='white', weight='bold', y=1.1)
    return fig

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Football AI", page_icon="⚽", layout="centered")

# --- 1. TITLE & TEXT (RENDER THIS FIRST) ---
st.title("⚽ Premier League AI")
st.caption("Powered by Machine Learning & Historical Data")

# --- 2. BACKGROUND ANIMATION & CSS (INJECT SECOND) ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: radial-gradient(ellipse at bottom, #0f172a 0%, #020617 100%);
    }
    
    /* Force Title to be visible */
    h1 {
        z-index: 99 !important;
        position: relative;
        color: white !important;
    }
    p {
        position: relative;
        z-index: 99;
    }
    
    /* Container */
    .floating-container {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        pointer-events: none; z-index: 0; overflow: hidden;
    }

    /* Icon Styling */
    .floating-icon {
        position: absolute; width: 60px; height: 60px;
        opacity: 0; transform: rotate(-15deg);
    }
    .floating-icon svg {
        width: 100%; height: 100%;
        fill: none; 
        stroke-width: 1.5;
        filter: drop-shadow(0 0 8px currentColor);
    }

    /* Animation */
    @keyframes shoot {
        0% { transform: translate(0, 0) rotate(-15deg); opacity: 0; }
        10% { opacity: 0.8; }
        90% { opacity: 0.8; }
        100% { transform: translate(100vw, -100vh) rotate(15deg); opacity: 0; }
    }

    /* Icon Variations */
    .icon-1 { top: 85%; left: 5%; color: #e2e8f0; animation: shoot 12s linear infinite; }
    .icon-2 { top: 60%; left: -5%; color: #22d3ee; animation: shoot 15s linear infinite 2s; width: 70px; }
    .icon-3 { top: 40%; left: -10%; color: #facc15; animation: shoot 18s linear infinite 5s; width: 80px; }
    .icon-4 { top: 20%; left: -5%; color: #f87171; animation: shoot 14s linear infinite 8s; width: 70px; }

    /* UI Styling */
    div[data-testid="stContainer"] {
        background-color: rgba(15, 23, 42, 0.6); 
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08); 
        border-radius: 16px;
        padding: 24px; 
        position: relative; z-index: 2;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    div[data-testid="stContainer"]:hover { border-color: #34d399; }
    
    div.stButton > button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6); 
        border: none; color: white; padding: 12px 24px; 
        border-radius: 12px; font-weight: 700; 
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.6);
    }
    
    h2, h3, label { color: white !important; }
    [data-testid="stMetricValue"] { color: #38bdf8 !important; }
    .stProgress > div > div > div > div { background-color: #34d399; }
</style>

<div class="floating-container">
<div class="floating-icon icon-1"><svg viewBox="0 0 24 24" stroke="currentColor"><circle cx="12" cy="12" r="10" stroke-width="2"/><path d="M12 2.5l2.5 4.5h-5l2.5 -4.5z"/><path d="M12 21.5l-2.5 -4.5h5l-2.5 4.5z"/><path d="M2.5 12l4.5 -2.5v5l-4.5 -2.5z"/><path d="M21.5 12l-4.5 2.5v-5l4.5 2.5z"/><path d="M7 9.5l5 -3l5 3v5l-5 3l-5 -3z"/></svg></div>
<div class="floating-icon icon-2"><svg viewBox="0 0 24 24" stroke="currentColor"><path d="M4 16c0-2 1-4 3-5c2-1 5-1 7 0c2 1 4 1 6 0c0 3-2 6-5 7h-8c-2 0-3-1-3-2z" stroke-width="2"/><path d="M5 18v3" stroke-width="2"/><path d="M9 18v3" stroke-width="2"/><path d="M16 18v3" stroke-width="2"/><path d="M19 16l2-2" stroke-width="1.5"/></svg></div>
<div class="floating-icon icon-3"><svg viewBox="0 0 24 24" stroke="currentColor"><path d="M6 4h12v2c0 4-2 7-6 7s-6-3-6-7v-2z" stroke-width="2"/><path d="M12 13v6" stroke-width="2"/><path d="M8 19h8" stroke-width="2"/><path d="M6 5c-3 0-4 2-4 5s1 5 4 0" stroke-width="1.5"/><path d="M18 5c3 0 4 2 4 5s-1 5-4 0" stroke-width="1.5"/></svg></div>
<div class="floating-icon icon-4"><svg viewBox="0 0 24 24" stroke="currentColor"><path d="M16 3h-8l-4 4v12h16v-12l-4-4z" stroke-width="2"/><path d="M8 3v4" stroke-width="1"/><path d="M16 3v4" stroke-width="1"/><path d="M10 10h4" stroke-width="1.5"/><path d="M18 7l2 2" stroke-width="1"/><path d="M6 7l-2 2" stroke-width="1"/></svg></div>
</div>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
try:
    API_KEY = st.secrets["FOOTBALL_API_KEY"]
except FileNotFoundError:
    st.error("Secrets file not found."); st.stop()

BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}

# --- 1. FUNCTION: LOAD & TRAIN AI ---
@st.cache_data
def load_and_train_model():
    all_files = glob.glob('*.csv') 
    if not all_files: return None, None, None, None, None, None

    df_list = []
    for filename in all_files:
        try:
            data = pd.read_csv(filename)
            data = data[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]
            df_list.append(data)
        except: pass

    if not df_list: return None, None, None, None, None, None

    df = pd.concat(df_list, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values('Date').dropna()
    df['Result'] = df['FTR'].map({'H': 1, 'D': 0, 'A': 2})

    team_stats = {} 
    df['HomeForm'] = 0; df['AwayForm'] = 0
    df['HomeAtt'] = 0.0; df['AwayAtt'] = 0.0
    df['HomeDef'] = 0.0; df['AwayDef'] = 0.0
    
    for index, row in df.iterrows():
        home, away, result = row['HomeTeam'], row['AwayTeam'], row['FTR']
        h_g, a_g = row['FTHG'], row['FTAG']
        
        h_recent = team_stats.get(home, [])
        a_recent = team_stats.get(away, [])
        
        h_pts = sum([3 if g['res']=='W' else 1 if g['res']=='D' else 0 for g in h_recent])
        a_pts = sum([3 if g['res']=='W' else 1 if g['res']=='D' else 0 for g in a_recent])
        h_att = sum([g['gf'] for g in h_recent]) / 5 if len(h_recent) > 0 else 1.0
        a_att = sum([g['gf'] for g in a_recent]) / 5 if len(a_recent) > 0 else 1.0
        h_def = sum([g['ga'] for g in h_recent]) / 5 if len(h_recent) > 0 else 1.0
        a_def = sum([g['ga'] for g in a_recent]) / 5 if len(a_recent) > 0 else 1.0
        
        df.at[index, 'HomeForm'] = h_pts; df.at[index, 'AwayForm'] = a_pts
        df.at[index, 'HomeAtt'] = h_att; df.at[index, 'AwayAtt'] = a_att
        df.at[index, 'HomeDef'] = h_def; df.at[index, 'AwayDef'] = a_def
        
        if home not in team_stats: team_stats[home] = []
        if away not in team_stats: team_stats[away] = []
        
        res_h = 'W' if result == 'H' else 'L' if result == 'A' else 'D'
        res_a = 'W' if result == 'A' else 'L' if result == 'H' else 'D'
        team_stats[home].append({'res': res_h, 'gf': h_g, 'ga': a_g})
        team_stats[away].append({'res': res_a, 'gf': a_g, 'ga': h_g})
        team_stats[home] = team_stats[home][-5:]; team_stats[away] = team_stats[away][-5:]

    all_teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    team_codes = {team: i for i, team in enumerate(all_teams)}
    df['HomeCode'] = df['HomeTeam'].map(team_codes)
    df['AwayCode'] = df['AwayTeam'].map(team_codes)
    
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
with st.spinner("Analyzing Season Data..."):
    rf_winner, rf_h_goals, rf_a_goals, team_codes, final_stats, df_history = load_and_train_model()

if rf_winner is None: st.error("No Data Found"); st.stop()

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
        form = sum([3 if g['res']=='W' else 1 if g['res']=='D' else 0 for g in recent])
        att = sum([g['gf'] for g in recent]) / 5 if len(recent) > 0 else 1.0
        defense = sum([g['ga'] for g in recent]) / 5 if len(recent) > 0 else 1.0
        return code, form, att, defense, clean
    else: return len(team_codes)//2, 7, 1.0, 1.0, clean

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
tab1, tab2 = st.tabs(["🔮 Live Predictions", "🏆 Standings"])

with tab1:
    if st.button("🚀 PREDICT NEXT MATCHES"):
        with st.spinner("Analyzing Match Data..."):
            try:
                response = requests.get(f"{BASE_URL}/competitions/PL/matches?status=SCHEDULED", headers=HEADERS)
                response.raise_for_status()
                matches = response.json()['matches']
                if not matches: st.warning("No matches found.")
                
                for match in matches[:5]:
                    h_api, a_api = match['homeTeam']['shortName'], match['awayTeam']['shortName']
                    h_code, h_form, h_att, h_def, h_name = get_team_data(h_api)
                    a_code, a_form, a_att, a_def, a_name = get_team_data(a_api)
                    
                    input_data = pd.DataFrame([[h_code, a_code, h_form, a_form, h_att, a_att, h_def, a_def]], 
                                            columns=['HomeCode', 'AwayCode', 'HomeForm', 'AwayForm', 'HomeAtt', 'AwayAtt', 'HomeDef', 'AwayDef'])
                    
                    winner_code = rf_winner.predict(input_data)[0]
                    h_g = rf_h_goals.predict(input_data)[0]
                    a_g = rf_a_goals.predict(input_data)[0]
                    probs = rf_winner.predict_proba(input_data)[0]
                    confidence = max(probs)

                    if winner_code == 1:
                        winner = f"{h_name} Wins"
                        h_color, a_color = '#48bb78', '#f56565' # Green / Red
                        if h_g <= a_g: h_g = a_g + 1
                    elif winner_code == 2:
                        winner = f"{a_name} Wins"
                        h_color, a_color = '#f56565', '#48bb78'
                        if a_g <= h_g: a_g = h_g + 1
                    else:
                        winner = "Draw"
                        h_color, a_color = '#4299e1', '#4299e1' # Blue
                        if h_g != a_g: h_g = a_g

                    # --- CARD UI ---
                    with st.container():
                        col1, col2, col3 = st.columns([1, 2, 1])
                        
                        with col1:
                            try: st.image(match['homeTeam'].get('crest', ''), width=50)
                            except: pass
                            h_stats = [h_form/1.5, h_att*3, h_def*3]
                            st.pyplot(create_radar_chart(h_name, h_stats, h_color), use_container_width=True)
                        
                        with col2:
                            st.markdown(f"<h1 style='text-align: center; color: white; margin-bottom: 0;'>{h_g} - {a_g}</h1>", unsafe_allow_html=True)
                            st.markdown(f"<p style='text-align: center; color: #a0aec0;'>PREDICTION: <b>{winner}</b></p>", unsafe_allow_html=True)
                            st.progress(probs[1] if winner_code==1 else probs[2] if winner_code==2 else probs[0])
                            st.caption(f"Confidence: {confidence*100:.0f}%")
                            
                        with col3:
                            try: st.image(match['awayTeam'].get('crest', ''), width=50)
                            except: pass
                            a_stats = [a_form/1.5, a_att*3, a_def*3]
                            st.pyplot(create_radar_chart(a_name, a_stats, a_color), use_container_width=True)

                        h_wins, draws, a_wins, past_games = get_h2h_stats(h_name, a_name)
                        with st.expander(f"📜 View History ({len(past_games)} games)"):
                            st.write(f"**H2H:** {h_name} ({h_wins}) - Draw ({draws}) - {a_name} ({a_wins})")
                            for game in past_games: st.caption(game)

            except Exception as e: st.error(f"Error: {e}")

with tab2:
    st.header("Standings")
    df_table = get_league_table()
    if df_table is not None: st.dataframe(df_table.set_index('Pos'), height=600, use_container_width=True)

# --- SIDEBAR ---
st.sidebar.header("Manual Simulator")
h_team = st.sidebar.selectbox("Home", list(team_codes.keys()))
a_team = st.sidebar.selectbox("Away", list(team_codes.keys()), index=1)
if st.sidebar.button("Simulate Match"):
    h_c, h_f, h_a, h_d, _ = get_team_data(h_team)
    a_c, a_f, a_a, a_d, _ = get_team_data(a_team)
    input_data = pd.DataFrame([[h_c, a_c, h_f, a_f, h_a, a_a, h_d, a_d]], columns=['HomeCode', 'AwayCode', 'HomeForm', 'AwayForm', 'HomeAtt', 'AwayAtt', 'HomeDef', 'AwayDef'])
    pred = rf_winner.predict(input_data)[0]
    probs = rf_winner.predict_proba(input_data)[0]
    
    res = f"{h_team} Wins" if pred == 1 else f"{a_team} Wins" if pred == 2 else "Draw"
    st.sidebar.success(f"{res} ({max(probs)*100:.0f}%)")
    
    h_col = '#48bb78' if pred == 1 else '#f56565' if pred == 2 else '#4299e1'
    a_col = '#48bb78' if pred == 2 else '#f56565' if pred == 1 else '#4299e1'

    st.sidebar.write("---")
    st.sidebar.pyplot(create_radar_chart(h_team, [h_f/1.5, h_a*3, h_d*3], h_col), use_container_width=True)
    st.sidebar.pyplot(create_radar_chart(a_team, [a_f/1.5, a_a*3, a_d*3], a_col), use_container_width=True)