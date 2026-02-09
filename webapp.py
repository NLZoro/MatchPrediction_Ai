import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import glob
import numpy as np 
import random
import google.generativeai as genai

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Football AI", page_icon="⚽", layout="wide")

# --- 1. CONFIGURATION & SECRETS ---
try: 
    API_KEY = st.secrets["FOOTBALL_API_KEY"]
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", None)
except: 
    st.error("Secrets file not found. Please setup .streamlit/secrets.toml"); st.stop()

BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

# --- 2. LEAGUE SYSTEM ---
LEAGUES = {
    "Premier League": {"code": "PL", "div": "E0", "theme": {"bg": "#37003c", "accent": "#00ff85", "grad": "radial-gradient(circle at top left, #37003c 0%, #000000 100%)"}, "icon": "🦁"},
    "La Liga": {"code": "PD", "div": "SP1", "theme": {"bg": "#ee8707", "accent": "#ffcc00", "grad": "radial-gradient(circle at top left, #991f1f 0%, #000000 100%)"}, "icon": "🇪🇸"},
    "Bundesliga": {"code": "BL1", "div": "D1", "theme": {"bg": "#d20515", "accent": "#ffffff", "grad": "radial-gradient(circle at top left, #d20515 0%, #000000 100%)"}, "icon": "🇩🇪"},
    "Serie A": {"code": "SA", "div": "I1", "theme": {"bg": "#008fd7", "accent": "#00ff85", "grad": "radial-gradient(circle at top left, #004687 0%, #000000 100%)"}, "icon": "🇮🇹"},
    "Ligue 1": {"code": "FL1", "div": "F1", "theme": {"bg": "#dae025", "accent": "#dae025", "grad": "radial-gradient(circle at top left, #091c3e 0%, #000000 100%)"}, "icon": "🇫🇷"}
}

# --- SIDEBAR ---
st.sidebar.title("🌍 League Select")
selected_league_name = st.sidebar.selectbox("Choose Competition", list(LEAGUES.keys()))
CURRENT_LEAGUE = LEAGUES[selected_league_name]
theme = CURRENT_LEAGUE["theme"]

# --- 3. CSS STYLING ---
st.markdown(f"""
<style>
    .stApp {{ background: {theme['grad']}; }}
    h1, h2, h3, p, label, .stMarkdown {{ color: white !important; }}
    
    .form-badge {{ display: inline-block; width: 28px; height: 28px; border-radius: 6px; margin-right: 4px; text-align: center; font-size: 12px; font-weight: 800; line-height: 28px; color: #1e293b; cursor: help; }}
    .win {{ background-color: #4ade80; }} .draw {{ background-color: #94a3b8; }} .loss {{ background-color: #f87171; }}
    
    div[data-testid="stContainer"] {{ background-color: rgba(20, 20, 20, 0.6); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 24px; position: relative; z-index: 2; }}
    div.stButton > button {{ background: linear-gradient(135deg, {theme['bg']}, #444); border: 1px solid {theme['accent']}; color: white; border-radius: 12px; font-weight: 700; }}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title(f"{CURRENT_LEAGUE['icon']} {selected_league_name} AI")
    st.caption(f"Live Intelligence • Season 2025/26")
with col_h2:
    try: st.image(f"https://crests.football-data.org/{CURRENT_LEAGUE['code']}.png", width=60)
    except: pass

# --- 4. CORE FUNCTIONS ---
def get_safe_stats(recent_games):
    if not recent_games: return 5.0, 5.0, 5.0 
    pts = sum([3 if x['res']=='W' else 1 if x['res']=='D' else 0 for x in recent_games])
    form = (pts / (len(recent_games)*3)) * 10 if len(recent_games) > 0 else 5
    att = min(np.mean([x['gf'] for x in recent_games]) * 3.5, 10)
    ga_avg = np.mean([x['ga'] for x in recent_games])
    defe = max(min((2.5 - ga_avg) * 4, 10), 1)
    return form, att, defe

@st.cache_data(show_spinner=False)
def get_ai_commentary(h_team, a_team, h_stats, a_stats, winner, league):
    """Uses Gemini to generate punditry."""
    if not GEMINI_KEY: return "⚠️ API Key Missing."
    
    prompt = (f"Act as a witty, slightly sarcastic football pundit (like Roy Keane). "
              f"Preview {league}: {h_team} vs {a_team}. "
              f"Stats - {h_team}: Att {h_stats[1]:.1f}, Def {h_stats[2]:.1f}. "
              f"{a_team}: Att {a_stats[1]:.1f}, Def {a_stats[2]:.1f}. "
              f"Predicted Winner: {winner}. Keep it short (2 sentences).")
    
    try:
        # Reverted to gemini-pro for stability
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: 
        return f"⚠️ Pundit Error: {str(e)}" 

def create_interactive_radar(team_name, stats, color):
    fig = go.Figure(go.Scatterpolar(r=stats, theta=['Form', 'Attack', 'Defense'], fill='toself', name=team_name, line=dict(color=color, width=3), fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.3,)}"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10], color='white', gridcolor='rgba(255,255,255,0.1)'), angularaxis=dict(color='white', gridcolor='rgba(255,255,255,0.1)'), bgcolor='rgba(0,0,0,0)'), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=20, r=20, t=20, b=20), showlegend=False, height=200)
    return fig

def create_momentum_pulse(h_name, a_name, h_goals, a_goals, h_col, a_col):
    intervals = np.arange(0, 95, 5) 
    momentum = []
    current_val = 0
    for _ in intervals:
        current_val += np.random.randint(-15, 15)
        current_val = max(min(current_val, 80), -80)
        momentum.append(current_val)
    
    events_x = []; events_y = []; events_text = []; events_color = []
    for _ in range(int(h_goals)):
        t = random.randint(10, 85); idx = int(t / 5); momentum[idx] = 90; events_x.append(t); events_y.append(95); events_text.append(f"⚽ {h_name}"); events_color.append(h_col)
    for _ in range(int(a_goals)):
        t = random.randint(10, 85); idx = int(t / 5); momentum[idx] = -90; events_x.append(t); events_y.append(-95); events_text.append(f"⚽ {a_name}"); events_color.append(a_col)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=intervals, y=momentum, mode='lines', line_shape='spline', line=dict(color='white', width=3), fill='tozeroy', name='Momentum', hoverinfo='skip'))
    if events_x: fig.add_trace(go.Scatter(x=events_x, y=events_y, mode='markers+text', text=events_text, textposition=["top center" if y>0 else "bottom center" for y in events_y], marker=dict(size=12, color=events_color, symbol='diamond', line=dict(width=2, color='white')), textfont=dict(color='white', size=10), showlegend=False))
    fig.update_layout(title=dict(text="Match Flow Simulation", font=dict(color='gray', size=12), x=0.5), xaxis=dict(visible=False, range=[0, 90]), yaxis=dict(visible=False, range=[-110, 110]), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=30, b=10), height=150, showlegend=False)
    return fig

def render_form_guide(recent_games):
    html = "<div style='display: flex; justify-content: center; margin-bottom: 10px; gap:4px;'>"
    for game in recent_games:
        cls = 'win' if game['res'] == 'W' else 'loss' if game['res'] == 'L' else 'draw'
        html += f"<div class='form-badge {cls}' title='vs {game['opp']}'>{game['res']}</div>"
    html += "</div>"
    return html

def adjust_scoreline(winner_code, h_g, a_g):
    if winner_code == 1: 
        if h_g <= a_g: h_g = a_g + 1 
    elif winner_code == 2: 
        if a_g <= h_g: a_g = h_g + 1 
    else: 
        if h_g != a_g: h_g = int((h_g + a_g) / 2); a_g = h_g
    return h_g, a_g

def get_csv_name(api_name, team_map):
    # Try exact match
    if api_name in team_map: return api_name
    # Try fuzzy match
    for k in team_map.keys():
        if api_name in k or k in api_name: return k
    return None

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

# --- 5. DATA LOADING ---
@st.cache_data(ttl=3600)
def get_live_data(code):
    try:
        r = requests.get(f"{BASE_URL}/competitions/{code}/matches?status=FINISHED", headers=HEADERS)
        data = r.json(); matches = sorted(data['matches'], key=lambda x: x['utcDate'])
        form_db = {}
        for m in matches:
            h, a = m['homeTeam']['shortName'], m['awayTeam']['shortName']
            hs, as_ = m['score']['fullTime']['home'], m['score']['fullTime']['away']
            if h not in form_db: form_db[h] = []
            if a not in form_db: form_db[a] = []
            res_h = 'W' if hs > as_ else 'L' if hs < as_ else 'D'
            res_a = 'W' if as_ > hs else 'L' if as_ < hs else 'D'
            form_db[h].append({'res': res_h, 'gf': hs, 'ga': as_, 'opp': a})
            form_db[a].append({'res': res_a, 'gf': as_, 'ga': hs, 'opp': h})
        for t in form_db: form_db[t] = form_db[t][-5:]
        return form_db, matches 
    except: return {}, []

@st.cache_data
def train_model(div):
    files = glob.glob('*.csv') + glob.glob('data/**/*.csv', recursive=True)
    df_list = []
    for f in files:
        try:
            d = pd.read_csv(f, encoding='unicode_escape') 
            if 'HomeTeam' in d.columns and 'Div' in d.columns and d['Div'].iloc[0] == div:
                df_list.append(d)
        except: pass
        
    if not df_list: return None, None, None, None, None
    
    # Safe Concat
    df = pd.concat(df_list, ignore_index=True)
    if 'HomeTeam' not in df.columns or 'AwayTeam' not in df.columns:
         return None, None, None, None, None
         
    df = df.dropna(subset=['HomeTeam','AwayTeam'])
    df['Result'] = df['FTR'].map({'H':1, 'D':0, 'A':2})
    teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    codes = {t: i for i, t in enumerate(teams)}
    
    df['HC'] = df['HomeTeam'].map(codes).fillna(-1)
    df['AC'] = df['AwayTeam'].map(codes).fillna(-1)
    df = df[(df['HC'] != -1) & (df['AC'] != -1)] 
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(df[['HC','AC']], df['Result'])
    rf_hg = RandomForestClassifier(n_estimators=100, random_state=42).fit(df[['HC','AC']], df['FTHG'])
    rf_ag = RandomForestClassifier(n_estimators=100, random_state=42).fit(df[['HC','AC']], df['FTAG'])
    return rf, rf_hg, rf_ag, codes, df 

# --- LOAD ---
with st.spinner(f"Initializing {selected_league_name}..."):
    live_form, all_matches = get_live_data(CURRENT_LEAGUE['code'])
    rf, rf_hg, rf_ag, team_map, hist_df = train_model(CURRENT_LEAGUE['div'])

if not live_form: st.error("⚠️ Data Error. Check API Key."); st.stop()

# --- MAIN ---
tab1, tab2 = st.tabs(["🔮 Match Center", "🏆 Standings"])

with tab1:
    if st.button(f"🚀 PREDICT {CURRENT_LEAGUE['code']} GAMES"):
        try:
            r = requests.get(f"{BASE_URL}/competitions/{CURRENT_LEAGUE['code']}/matches?status=SCHEDULED", headers=HEADERS)
            games = r.json()['matches'][:5]
            
            for i, m in enumerate(games):
                h, a = m['homeTeam']['shortName'], m['awayTeam']['shortName']
                h_rec, a_rec = live_form.get(h, []), live_form.get(a, [])
                h_stats, a_stats = get_safe_stats(h_rec), get_safe_stats(a_rec)
                
                h_csv = get_csv_name(h, team_map)
                a_csv = get_csv_name(a, team_map)
                
                if rf and h_csv and a_csv:
                    h_c = team_map[h_csv]
                    a_c = team_map[a_csv]
                    pred = rf.predict([[h_c, a_c]])[0]
                    hg = int(rf_hg.predict([[h_c, a_c]])[0]); ag = int(rf_ag.predict([[h_c, a_c]])[0])
                    hg, ag = adjust_scoreline(pred, hg, ag)
                else:
                    pred = 1; hg=1; ag=0 
                
                winner = f"{h} Wins" if pred==1 else f"{a} Wins" if pred==2 else "Draw"
                
                with st.container():
                    c1, c2, c3 = st.columns([1.2, 1.5, 1.2])
                    with c1:
                        try: st.image(m['homeTeam']['crest'], width=50)
                        except: pass
                        st.markdown(render_form_guide(h_rec), unsafe_allow_html=True)
                        st.plotly_chart(create_interactive_radar(h, h_stats, '#4ade80'), use_container_width=True, key=f"r_h_{i}", config={'displayModeBar':False})
                    with c2:
                        st.markdown(f"<h1 style='text-align:center;margin:0;'>{hg} - {ag}</h1>", unsafe_allow_html=True)
                        st.markdown(f"<p style='text-align:center;color:#94a3b8;'>{winner}</p>", unsafe_allow_html=True)
                        
                        punditry = get_ai_commentary(h, a, h_stats, a_stats, winner, selected_league_name)
                        st.info(f"🎙️ **The Pundit:** {punditry}")
                        
                        st.plotly_chart(create_momentum_pulse(h, a, hg, ag, '#4ade80', '#f87171'), use_container_width=True, key=f"p_{i}", config={'displayModeBar':False})
                        
                        with st.expander("⚔️ Head-to-Head History"):
                            if hist_df is not None and h_csv and a_csv:
                                h2h = hist_df[((hist_df['HomeTeam'] == h_csv) & (hist_df['AwayTeam'] == a_csv)) | 
                                              ((hist_df['HomeTeam'] == a_csv) & (hist_df['AwayTeam'] == h_csv))].tail(5)
                                if not h2h.empty:
                                    for _, row in h2h.iterrows():
                                        st.caption(f"{row['Date']}: {row['HomeTeam']} {int(row['FTHG'])}-{int(row['FTAG'])} {row['AwayTeam']}")
                                else: st.caption(f"No match history found between {h_csv} and {a_csv}.")
                            else: st.caption("Could not map teams to history database.")

                    with c3:
                        try: st.image(m['awayTeam']['crest'], width=50)
                        except: pass
                        st.markdown(render_form_guide(a_rec), unsafe_allow_html=True)
                        st.plotly_chart(create_interactive_radar(a, a_stats, '#f87171'), use_container_width=True, key=f"r_a_{i}", config={'displayModeBar':False})

        except Exception as e: st.error(f"Error: {e}")

with tab2:
    @st.cache_data
    def get_table(code):
        try:
            r = requests.get(f"{BASE_URL}/competitions/{code}/standings", headers=HEADERS)
            t = r.json()['standings'][0]['table']
            return pd.DataFrame([{'Pos': x['position'], 'Logo': x['team']['crest'], 'Team': x['team']['shortName'], 'P': x['playedGames'], 'Pts': x['points']} for x in t])
        except: return None
    st.header(f"{selected_league_name} Table")
    t = get_table(CURRENT_LEAGUE['code'])
    if t is not None: st.dataframe(t.set_index('Pos'), use_container_width=True, column_config={"Logo": st.column_config.ImageColumn("Logo", width="small")})

# --- SIDEBAR (Restored) ---
st.sidebar.header("Manual Simulator")
if live_form:
    teams = sorted(live_form.keys())
    h = st.sidebar.selectbox("Home", teams, index=0)
    a = st.sidebar.selectbox("Away", teams, index=1)
    
    if st.sidebar.button("Simulate Match"):
        h_rec, a_rec = live_form.get(h, []), live_form.get(a, [])
        h_stats, a_stats = get_safe_stats(h_rec), get_safe_stats(a_rec)
        
        h_csv = get_csv_name(h, team_map)
        a_csv = get_csv_name(a, team_map)
        
        if rf and h_csv and a_csv:
            h_c = team_map[h_csv]
            a_c = team_map[a_csv]
            pred = rf.predict([[h_c, a_c]])[0]
            hg = int(rf_hg.predict([[h_c, a_c]])[0])
            ag = int(rf_ag.predict([[h_c, a_c]])[0])
            hg, ag = adjust_scoreline(pred, hg, ag)
        else:
             pred = 1; hg=1; ag=0

        winner = f"{h} Wins" if pred==1 else f"{a} Wins" if pred==2 else "Draw"
        st.sidebar.success(f"{h} {hg} - {ag} {a}")
        st.sidebar.info(get_ai_commentary(h, a, h_stats, a_stats, winner, selected_league_name))
        
        # KEY FIX: Unique keys for sidebar charts
        st.sidebar.plotly_chart(create_momentum_pulse(h, a, hg, ag, '#4ade80', '#f87171'), use_container_width=True, key="sb_pulse", config={'displayModeBar':False})
        
        c1, c2 = st.sidebar.columns(2)
        c1.plotly_chart(create_trend_chart(h_rec, '#4ade80'), use_container_width=True, key="sb_trend_h", config={'displayModeBar':False})
        c2.plotly_chart(create_trend_chart(a_rec, '#f87171'), use_container_width=True, key="sb_trend_a", config={'displayModeBar':False})