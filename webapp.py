import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import glob
import numpy as np 

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Football AI", page_icon="⚽", layout="centered")

# --- 1. TITLE & HEADER ---
st.title("⚽ Premier League AI")
st.caption("Live Intelligence • Season 2025/26")

# --- 2. CSS STYLING ---
st.markdown("""
<style>
    .stApp { background: radial-gradient(ellipse at bottom, #0f172a 0%, #020617 100%); }
    h1, p, label { z-index: 99 !important; position: relative; color: white !important; }
    
    /* Background Animation */
    .floating-container { position: fixed; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 0; overflow: hidden; }
    .floating-icon { position: absolute; width: 60px; height: 60px; opacity: 0; transform: rotate(-15deg); }
    .floating-icon svg { width: 100%; height: 100%; fill: none; stroke-width: 1.5; filter: drop-shadow(0 0 5px currentColor); }
    @keyframes shoot { 0% { transform: translate(0, 0) rotate(-15deg); opacity: 0; } 10% { opacity: 0.6; } 90% { opacity: 0.6; } 100% { transform: translate(100vw, -100vh) rotate(15deg); opacity: 0; } }
    .icon-1 { top: 85%; left: 5%; color: #e2e8f0; animation: shoot 12s linear infinite; }
    .icon-2 { top: 60%; left: -5%; color: #22d3ee; animation: shoot 15s linear infinite 2s; width: 70px; }
    .icon-3 { top: 40%; left: -10%; color: #facc15; animation: shoot 18s linear infinite 5s; width: 80px; }
    .icon-4 { top: 20%; left: -5%; color: #f87171; animation: shoot 14s linear infinite 8s; width: 70px; }

    /* UI Components */
    div[data-testid="stContainer"] { background-color: rgba(15, 23, 42, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 24px; position: relative; z-index: 2; box-shadow: 0 4px 20px rgba(0,0,0,0.4); }
    div[data-testid="stContainer"]:hover { border-color: #34d399; }
    div.stButton > button { background: linear-gradient(135deg, #3b82f6, #8b5cf6); border: none; color: white; padding: 12px 24px; border-radius: 12px; font-weight: 700; box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4); transition: all 0.3s ease; }
    div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(59, 130, 246, 0.6); }
    
    /* Form Badges */
    .form-badge { display: inline-block; width: 28px; height: 28px; border-radius: 6px; margin-right: 4px; text-align: center; font-size: 12px; font-weight: 800; line-height: 28px; color: #1e293b; box-shadow: 0 2px 5px rgba(0,0,0,0.2); cursor: help; }
    .win { background-color: #4ade80; } .draw { background-color: #94a3b8; } .loss { background-color: #f87171; }
    [data-testid="stMetricValue"] { color: #38bdf8 !important; }
</style>
<div class="floating-container">
<div class="floating-icon icon-1"><svg viewBox="0 0 24 24" stroke="currentColor"><circle cx="12" cy="12" r="10" stroke-width="2"/><path d="M12 2.5l2.5 4.5h-5l2.5 -4.5z"/><path d="M12 21.5l-2.5 -4.5h5l-2.5 4.5z"/><path d="M2.5 12l4.5 -2.5v5l-4.5 -2.5z"/><path d="M21.5 12l-4.5 2.5v-5l4.5 2.5z"/><path d="M7 9.5l5 -3l5 3v5l-5 3l-5 -3z"/></svg></div>
<div class="floating-icon icon-2"><svg viewBox="0 0 24 24" stroke="currentColor"><path d="M4 16c0-2 1-4 3-5c2-1 5-1 7 0c2 1 4 1 6 0c0 3-2 6-5 7h-8c-2 0-3-1-3-2z" stroke-width="2"/><path d="M5 18v3" stroke-width="2"/><path d="M9 18v3" stroke-width="2"/><path d="M16 18v3" stroke-width="2"/><path d="M19 16l2-2" stroke-width="1.5"/></svg></div>
<div class="floating-icon icon-3"><svg viewBox="0 0 24 24" stroke="currentColor"><path d="M6 4h12v2c0 4-2 7-6 7s-6-3-6-7v-2z" stroke-width="2"/><path d="M12 13v6" stroke-width="2"/><path d="M8 19h8" stroke-width="2"/><path d="M6 5c-3 0-4 2-4 5s1 5 4 0" stroke-width="1.5"/><path d="M18 5c3 0 4 2 4 5s-1 5-4 0" stroke-width="1.5"/></svg></div>
<div class="floating-icon icon-4"><svg viewBox="0 0 24 24" stroke="currentColor"><path d="M16 3h-8l-4 4v12h16v-12l-4-4z" stroke-width="2"/><path d="M8 3v4" stroke-width="1"/><path d="M16 3v4" stroke-width="1"/><path d="M10 10h4" stroke-width="1.5"/><path d="M18 7l2 2" stroke-width="1"/><path d="M6 7l-2 2" stroke-width="1"/></svg></div>
</div>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
try: API_KEY = st.secrets["FOOTBALL_API_KEY"]
except: st.error("Secrets file not found."); st.stop()
BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}

# --- 3. HELPER FUNCTIONS ---
def get_safe_stats(recent_games):
    """Safely calculate stats from live games, handling empty lists."""
    if not recent_games: return 5.0, 5.0, 5.0 # Default if no data
    
    # Form: Points (W=3, D=1, L=0) normalized to 0-10
    pts = sum([3 if x['res']=='W' else 1 if x['res']=='D' else 0 for x in recent_games])
    form_score = (pts / (len(recent_games)*3)) * 10 if len(recent_games) > 0 else 5
    
    # Attack: Avg Goals Scored * 3 (Max 10)
    att = min(np.mean([x['gf'] for x in recent_games]) * 3.5, 10)
    
    # Defense: (3 - Avg Goals Conceded) * 3 (Max 10). Low GA = High Def Score.
    ga_avg = np.mean([x['ga'] for x in recent_games])
    defe = max(min((2.5 - ga_avg) * 4, 10), 1)
    
    return form_score, att, defe

def create_interactive_radar(team_name, stats, color):
    fig = go.Figure(go.Scatterpolar(r=stats, theta=['Form', 'Attack', 'Defense'], fill='toself', name=team_name, line=dict(color=color, width=3), fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.3,)}"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10], color='white', gridcolor='rgba(255,255,255,0.1)'), angularaxis=dict(color='white', gridcolor='rgba(255,255,255,0.1)'), bgcolor='rgba(0,0,0,0)'), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=20, r=20, t=20, b=20), showlegend=False, height=200)
    return fig

def create_trend_chart(recent_games, color):
    if not recent_games: return go.Figure()
    gf = [g['gf'] for g in recent_games]
    ga = [g['ga'] for g in recent_games]
    x = [f"vs {g['opp']}" for g in recent_games]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=gf, mode='lines+markers', name='GF', line=dict(color=color, width=3)))
    fig.add_trace(go.Scatter(x=x, y=ga, mode='lines', name='GA', line=dict(color='rgba(255,255,255,0.5)', width=1, dash='dot')))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title="Goals", titlefont=dict(size=10, color='white')), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=10, b=10), showlegend=False, height=120)
    return fig

def render_form_guide(recent_games):
    html = "<div style='display: flex; justify-content: center; margin-bottom: 10px; gap:4px;'>"
    for game in recent_games: # Already sorted Old -> New
        res = game['res']
        tooltip = f"vs {game['opp']} ({game['gf']}-{game['ga']})"
        cls = 'win' if res == 'W' else 'loss' if res == 'L' else 'draw'
        html += f"<div class='form-badge {cls}' title='{tooltip}'>{res}</div>"
    html += "</div>"
    return html

def generate_scout_report(h_name, a_name, h_stats, a_stats, prediction, conf):
    report = ""
    if conf > 0.65: report += f"**🚀 High Confidence:** The model strongly favors {prediction.split(' ')[0]}.\n\n"
    elif conf < 0.45: report += f"**⚠️ Tight Affair:** This match is too close to call.\n\n"
    if h_stats[1] > a_stats[2] + 2: report += f"⚔️ **Mismatch:** {h_name}'s attack is overwhelming compared to {a_name}'s defense.\n"
    elif a_stats[1] > h_stats[2] + 2: report += f"⚔️ **Mismatch:** {a_name}'s attack poses a massive threat to {h_name}.\n"
    return report

# --- 4. DATA ENGINE (MODELS + LIVE FORM) ---
@st.cache_data(ttl=3600) # Update every hour
def get_live_form_data():
    try:
        # Fetch 2025/26 season (or current)
        url = f"{BASE_URL}/competitions/PL/matches?status=FINISHED"
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        
        # 1. Sort matches by date to ensure accuracy
        matches = sorted(data['matches'], key=lambda x: x['utcDate'])
        
        form_db = {}
        for m in matches:
            h_team = m['homeTeam']['shortName']
            a_team = m['awayTeam']['shortName']
            h_score = m['score']['fullTime']['home']
            a_score = m['score']['fullTime']['away']
            
            if h_team not in form_db: form_db[h_team] = []
            if a_team not in form_db: form_db[a_team] = []
            
            # Save Opponent Name for Tooltips
            res_h = 'W' if h_score > a_score else 'L' if h_score < a_score else 'D'
            form_db[h_team].append({'res': res_h, 'gf': h_score, 'ga': a_score, 'opp': a_team})
            
            res_a = 'W' if a_score > h_score else 'L' if a_score < h_score else 'D'
            form_db[a_team].append({'res': res_a, 'gf': a_score, 'ga': h_score, 'opp': h_team})
            
        # Keep only last 5
        for t in form_db: form_db[t] = form_db[t][-5:]
        return form_db
    except Exception as e:
        return {}

@st.cache_data
def load_and_train_model():
    all_files = glob.glob('*.csv')
    if not all_files: return None, None, None
    df_list = []
    for f in all_files:
        try: df_list.append(pd.read_csv(f))
        except: pass
    if not df_list: return None, None, None
    
    df = pd.concat(df_list, ignore_index=True)
    df['Result'] = df['FTR'].map({'H': 1, 'D': 0, 'A': 2})
    
    # --- CRITICAL FIX: CLEAN DATA BEFORE SORTING ---
    # Ensure HomeTeam and AwayTeam are strings and remove NaN rows
    df = df.dropna(subset=['HomeTeam', 'AwayTeam'])
    df['HomeTeam'] = df['HomeTeam'].astype(str)
    df['AwayTeam'] = df['AwayTeam'].astype(str)
    
    all_teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    team_codes = {team: i for i, team in enumerate(all_teams)}
    df['HomeCode'] = df['HomeTeam'].map(team_codes).fillna(0)
    df['AwayCode'] = df['AwayTeam'].map(team_codes).fillna(0)
    
    # Simple features available in CSV
    features = ['HomeCode', 'AwayCode'] 
    X = df[features].fillna(0)
    y = df['Result']
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
    return rf, team_codes, team_codes 

# --- LOAD DATA ---
with st.spinner("Connecting to Premier League Live Data..."):
    rf_model, team_map, _ = load_and_train_model()
    live_form_db = get_live_form_data() 

if not live_form_db:
    st.error("⚠️ API Connection Error. Showing basic mode.")
    # Fallback to empty if API fails
    live_form_db = {}

# --- MAIN ---
tab1, tab2 = st.tabs(["🔮 Match Center", "🏆 Standings"])

with tab1:
    if st.button("🚀 PREDICT NEXT MATCHES"):
        with st.spinner("Analyzing Live Form..."):
            try:
                res = requests.get(f"{BASE_URL}/competitions/PL/matches?status=SCHEDULED", headers=HEADERS)
                matches = res.json()['matches']
                
                for m in matches[:5]:
                    h_name = m['homeTeam']['shortName']
                    a_name = m['awayTeam']['shortName']
                    
                    # 1. Get Live Stats
                    h_recent = live_form_db.get(h_name, [])
                    a_recent = live_form_db.get(a_name, [])
                    
                    h_stats = get_safe_stats(h_recent)
                    a_stats = get_safe_stats(a_recent)
                    
                    # 2. Predict (Map API Name -> CSV Code)
                    # Fuzzy match API name to CSV name
                    h_csv_code = next((v for k,v in team_map.items() if h_name in k or k in h_name), 0)
                    a_csv_code = next((v for k,v in team_map.items() if a_name in k or k in a_name), 0)
                    
                    input_data = pd.DataFrame([[h_csv_code, a_csv_code]], columns=['HomeCode', 'AwayCode'])
                    pred_code = rf_model.predict(input_data)[0]
                    probs = rf_model.predict_proba(input_data)[0]
                    conf = max(probs)
                    
                    if pred_code == 1: winner = f"{h_name} Wins"; h_col, a_col = '#4ade80', '#f87171'
                    elif pred_code == 2: winner = f"{a_name} Wins"; h_col, a_col = '#f87171', '#4ade80'
                    else: winner = "Draw"; h_col, a_col = '#22d3ee', '#22d3ee'

                    # UI
                    with st.container():
                        col1, col2, col3 = st.columns([1.2, 1.6, 1.2])
                        with col1:
                            try: st.image(m['homeTeam'].get('crest', ''), width=50)
                            except: pass
                            st.markdown(render_form_guide(h_recent), unsafe_allow_html=True)
                            st.plotly_chart(create_interactive_radar(h_name, h_stats, h_col), use_container_width=True, config={'displayModeBar': False})
                        with col2:
                            st.markdown(f"<h2 style='text-align: center; color:white;'>VS</h2>", unsafe_allow_html=True)
                            st.markdown(f"<p style='text-align: center; color: #94a3b8; font-weight: 600;'>{winner}</p>", unsafe_allow_html=True)
                            scout = generate_scout_report(h_name, a_name, h_stats, a_stats, winner, conf)
                            st.info(scout, icon="🤖")
                            st.progress(conf)
                        with col3:
                            try: st.image(m['awayTeam'].get('crest', ''), width=50)
                            except: pass
                            st.markdown(render_form_guide(a_recent), unsafe_allow_html=True)
                            st.plotly_chart(create_interactive_radar(a_name, a_stats, a_col), use_container_width=True, config={'displayModeBar': False})

            except Exception as e: st.error(f"Error: {e}")

with tab2:
    @st.cache_data
    def get_table():
        try:
            r = requests.get(f"{BASE_URL}/competitions/PL/standings", headers=HEADERS)
            t = r.json()['standings'][0]['table']
            return pd.DataFrame([{'Pos': x['position'], 'Team': x['team']['shortName'], 'P': x['playedGames'], 'Pts': x['points'], 'GD': x['goalDifference']} for x in t])
        except: return None
    st.header("Standings")
    t = get_table()
    if t is not None: st.dataframe(t.set_index('Pos'), use_container_width=True)

# --- SIDEBAR (Now using LIVE DATA) ---
st.sidebar.header("Manual Simulator")
# Sort teams alphabetically
teams_list = sorted(list(live_form_db.keys())) if live_form_db else ["Arsenal", "Aston Villa"]
h_team = st.sidebar.selectbox("Home", teams_list)
a_team = st.sidebar.selectbox("Away", teams_list, index=1)

if st.sidebar.button("Simulate"):
    h_r = live_form_db.get(h_team, [])
    a_r = live_form_db.get(a_team, [])
    h_stats = get_safe_stats(h_r)
    a_stats = get_safe_stats(a_r)
    
    # Fuzzy match API name to CSV code
    h_csv_code = next((v for k,v in team_map.items() if h_team in k or k in h_team), 0)
    a_csv_code = next((v for k,v in team_map.items() if a_team in k or k in a_team), 0)
    
    input_data = pd.DataFrame([[h_csv_code, a_csv_code]], columns=['HomeCode', 'AwayCode'])
    pred = rf_model.predict(input_data)[0]
    probs = rf_model.predict_proba(input_data)[0]
    
    res = f"{h_team} Wins" if pred == 1 else f"{a_team} Wins" if pred == 2 else "Draw"
    st.sidebar.success(f"{res} ({max(probs)*100:.0f}%)")
    
    # Dynamic Colors
    h_col = '#4ade80' if pred == 1 else '#f87171' if pred == 2 else '#22d3ee'
    a_col = '#4ade80' if pred == 2 else '#f87171' if pred == 1 else '#22d3ee'

    st.sidebar.write("---")
    st.sidebar.markdown(f"**{h_team} Form**")
    st.sidebar.markdown(render_form_guide(h_r), unsafe_allow_html=True)
    st.sidebar.plotly_chart(create_trend_chart(h_r, h_col), use_container_width=True, config={'displayModeBar': False})
    
    st.sidebar.write("---")
    st.sidebar.markdown(f"**{a_team} Form**")
    st.sidebar.markdown(render_form_guide(a_r), unsafe_allow_html=True)
    st.sidebar.plotly_chart(create_trend_chart(a_r, a_col), use_container_width=True, config={'displayModeBar': False})