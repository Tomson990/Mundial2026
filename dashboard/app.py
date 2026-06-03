"""
Dashboard Streamlit — WC 2026 Predictor
Correr con: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import pickle
import os
import sys
from itertools import combinations

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC = os.path.join(BASE_DIR, 'data', 'processed')
sys.path.insert(0, BASE_DIR)

st.set_page_config(page_title="WC 2026 Predictor", page_icon="⚽", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0a0e1a; color: #e8eaf0; }
    h1, h2, h3 { color: #c8d8ff !important; }
    .stTabs [data-baseweb="tab"] { color: #8899cc; font-size: 0.95rem; }
    .stTabs [aria-selected="true"] { color: #7eb3ff !important; border-bottom-color: #7eb3ff !important; }
    .group-header { background: #1a1f35; border-left: 3px solid #7eb3ff; padding: 8px 14px; border-radius: 4px; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    models   = {}
    dc_path  = os.path.join(DATA_PROC, 'dixon_coles.pkl')
    xgb_path = os.path.join(DATA_PROC, 'xgboost_model.pkl')
    try:
        if os.path.exists(dc_path):
            with open(dc_path, 'rb') as f:
                models['dc'] = pickle.load(f)
    except Exception:
        pass
    try:
        if os.path.exists(xgb_path):
            with open(xgb_path, 'rb') as f:
                models['gb'] = pickle.load(f)
    except Exception:
        pass
    return models


@st.cache_data
def load_ratings():
    path = os.path.join(DATA_PROC, 'elo_ratings.csv')
    if not os.path.exists(path):
        return {}
    df       = pd.read_csv(path)
    cols     = df.columns.tolist()
    team_col = next((c for c in cols if 'team'   in c.lower()), cols[0])
    rat_col  = next((c for c in cols if 'elo'    in c.lower() or 'rating' in c.lower()), cols[-1])
    return dict(zip(df[team_col], df[rat_col]))


@st.cache_data
def load_groups():
    path = os.path.join(BASE_DIR, 'data', 'wc2026_groups.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return data['groups'] if 'groups' in data else data


@st.cache_data
def load_mc_results():
    path = os.path.join(DATA_PROC, 'mc_results.csv')
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_data
def load_prode():
    path = os.path.join(DATA_PROC, 'prode_grupos.csv')
    return pd.read_csv(path) if os.path.exists(path) else None


def make_predictor(models, ratings):
    def predictor(home, away, neutral=False):
        preds, weights = [], []
        elo_h  = ratings.get(home, 1500)
        elo_a  = ratings.get(away, 1500)
        p_h    = 1 / (1 + 10 ** (-(elo_h - elo_a) / 400))
        draw_p = 0.22
        preds.append({'home_win': p_h*(1-draw_p), 'draw': draw_p, 'away_win': (1-p_h)*(1-draw_p)})
        weights.append(1)
        if 'dc' in models:
            try:
                p = models['dc'].predict_result(home, away)
                if p: preds.append(p); weights.append(2)
            except Exception:
                pass
        if 'gb' in models:
            try:
                p = models['gb'].predict_from_ratings(home, away, ratings, dc_model=models.get('dc'), neutral=neutral)
                if p: preds.append(p); weights.append(2)
            except Exception:
                pass
        w_total = sum(weights)
        return {
            'home_win': sum(p['home_win']*w for p,w in zip(preds,weights)) / w_total,
            'draw':     sum(p['draw']    *w for p,w in zip(preds,weights)) / w_total,
            'away_win': sum(p['away_win']*w for p,w in zip(preds,weights)) / w_total,
        }
    return predictor


def main():
    st.title("⚽ WC 2026 Predictor")
    st.markdown("*Ensemble Elo + Dixon-Coles + XGBoost · Monte Carlo 5,000 simulaciones*")

    models    = load_models()
    ratings   = load_ratings()
    groups    = load_groups()
    all_teams = sorted(set(list(ratings.keys()) + [t for grp in groups.values() for t in grp]))

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Campeón",
        "🗂️ Fase de Grupos",
        "⚡ Simulador de Partido",
        "📊 Rankings",
        "🎲 Monte Carlo",
    ])

    # ── TAB 1: Campeón ────────────────────────────────────────────
    with tab1:
        st.header("Probabilidades de Campeón")
        mc = load_mc_results()
        if mc is not None:
            top10 = mc.dropna(subset=['team']).head(10)
            fig = px.bar(
                top10, x='champion', y='team', orientation='h',
                color='champion', color_continuous_scale='Blues',
                labels={'champion': '%', 'team': ''},
                title='Top 10 — Probabilidad de ganar el Mundial'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#c8d8ff', yaxis={'categoryorder': 'total ascending'},
                showlegend=False, coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            show_cols = ['team']
            rename    = {'team': 'Equipo'}
            if 'qualified' in mc.columns:
                show_cols.append('qualified'); rename['qualified'] = 'Clasifica %'
            show_cols.append('champion'); rename['champion'] = 'Campeón %'
            st.dataframe(mc[show_cols].rename(columns=rename), use_container_width=True, hide_index=True)
        else:
            st.info("Andá a la pestaña **Monte Carlo** para correr la simulación primero.")

    # ── TAB 2: Fase de Grupos ─────────────────────────────────────
    with tab2:
        st.header("Fase de Grupos — Prode")
        prode_df = load_prode()

        if prode_df is not None:
            group_options = ['Todos'] + sorted(prode_df['group'].unique())
            selected = st.selectbox("Filtrar por grupo", group_options)
            df_show  = prode_df if selected == 'Todos' else prode_df[prode_df['group'] == selected]

            for grp in sorted(df_show['group'].unique()):
                grp_teams = groups.get(grp, [])
                st.markdown(f"<div class='group-header'><b>Grupo {grp}</b> &nbsp;·&nbsp; {' &nbsp;·&nbsp; '.join(grp_teams)}</div>", unsafe_allow_html=True)
                grp_df = df_show[df_show['group'] == grp].copy()
                grp_df['Partido']  = grp_df['home'] + ' vs ' + grp_df['away']
                grp_df['Local %']  = grp_df['home_win'].apply(lambda x: f"{x:.1f}%")
                grp_df['Empate %'] = grp_df['draw'].apply(lambda x: f"{x:.1f}%")
                grp_df['Visit. %'] = grp_df['away_win'].apply(lambda x: f"{x:.1f}%")
                grp_df['Favorito'] = grp_df['fav']
                st.dataframe(
                    grp_df[['Partido', 'Local %', 'Empate %', 'Visit. %', 'Favorito']],
                    use_container_width=True, hide_index=True
                )
                st.markdown("")
        else:
            st.info("Generá el prode desde la terminal: `python pipeline.py --prode`")
            if st.button("🗂️ Generar Prode ahora", type="primary"):
                if not groups:
                    st.error("No se encontró wc2026_groups.json")
                else:
                    predictor = make_predictor(models, ratings)
                    rows      = []
                    progress  = st.progress(0)
                    group_list = sorted(groups.items())
                    for i, (grp, teams) in enumerate(group_list):
                        for home, away in combinations(teams, 2):
                            pred = predictor(home, away, neutral=True)
                            hw, dw, aw = pred['home_win']*100, pred['draw']*100, pred['away_win']*100
                            fav = home if hw > aw else (away if aw > hw else 'Empate')
                            rows.append({'group': grp, 'home': home, 'away': away,
                                         'home_win': round(hw,1), 'draw': round(dw,1),
                                         'away_win': round(aw,1), 'fav': fav})
                        progress.progress((i+1) / len(group_list))
                    result_df = pd.DataFrame(rows)
                    result_df.to_csv(os.path.join(DATA_PROC, 'prode_grupos.csv'), index=False)
                    st.success("✓ Listo. Recargá la página.")
                    st.cache_data.clear()

    # ── TAB 3: Simulador de Partido ───────────────────────────────
    with tab3:
        st.header("Simulador de Partido")
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            home_team = st.selectbox("🏠 Local", all_teams,
                index=all_teams.index('Argentina') if 'Argentina' in all_teams else 0)
        with col2:
            st.markdown("<br><br><div style='text-align:center;font-size:1.5rem'>VS</div>", unsafe_allow_html=True)
        with col3:
            away_opts = [t for t in all_teams if t != home_team]
            away_team = st.selectbox("✈️ Visitante", away_opts,
                index=away_opts.index('France') if 'France' in away_opts else 0)

        neutral = st.checkbox("Campo Neutral", value=True)

        if st.button("⚡ Predecir", type="primary", use_container_width=True):
            predictor = make_predictor(models, ratings)
            pred = predictor(home_team, away_team, neutral)
            c1, c2, c3 = st.columns(3)
            with c1: st.metric(f"🏠 {home_team}", f"{pred['home_win']*100:.1f}%")
            with c2: st.metric("🤝 Empate",        f"{pred['draw']*100:.1f}%")
            with c3: st.metric(f"✈️ {away_team}",  f"{pred['away_win']*100:.1f}%")

            if 'dc' in models:
                try:
                    matrix = models['dc'].predict_scoreline(home_team, away_team)
                    if matrix is not None:
                        max_g = 5
                        mat   = matrix[:max_g+1, :max_g+1]
                        mat   = mat / mat.sum() * 100
                        fig   = go.Figure(data=go.Heatmap(
                            z=mat,
                            x=[str(i) for i in range(max_g+1)],
                            y=[str(i) for i in range(max_g+1)],
                            colorscale='Blues',
                            text=[[f"{mat[i,j]:.1f}%" for j in range(max_g+1)] for i in range(max_g+1)],
                            texttemplate="%{text}", showscale=False,
                        ))
                        fig.update_layout(
                            title=f"Marcadores — {home_team} (filas) vs {away_team} (columnas)",
                            xaxis_title=f"Goles {away_team}",
                            yaxis_title=f"Goles {home_team}",
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            font_color='#c8d8ff',
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass

    # ── TAB 4: Rankings ───────────────────────────────────────────
    with tab4:
        st.header("Ranking de Selecciones")
        col_elo, col_dc = st.columns(2)
        with col_elo:
            st.subheader("🔵 Elo")
            if ratings:
                elo_df = pd.DataFrame(list(ratings.items()), columns=['Equipo', 'Elo'])
                elo_df = elo_df.sort_values('Elo', ascending=False).reset_index(drop=True)
                elo_df.index += 1
                st.dataframe(elo_df.head(30), use_container_width=True)
        with col_dc:
            st.subheader("🟠 Dixon-Coles")
            if 'dc' in models:
                try:
                    dc_df = models['dc'].get_team_params()
                    if not dc_df.empty:
                        dc_df.columns = ['Equipo', 'Ataque', 'Defensa', 'Overall']
                        st.dataframe(dc_df.head(30).round(3), use_container_width=True)
                    else:
                        st.info("Sin datos suficientes.")
                except Exception:
                    st.info("Dixon-Coles no disponible en este entorno.")
            else:
                st.info("Dixon-Coles no disponible.")

    # ── TAB 5: Monte Carlo ────────────────────────────────────────
    with tab5:
        st.header("Simulación Monte Carlo")
        n_sims = st.slider("Simulaciones", 1000, 10000, 5000, step=1000)

        if st.button("🎲 Correr Simulación", type="primary", use_container_width=True):
            if not groups:
                st.error("No se encontró wc2026_groups.json")
            else:
                predictor = make_predictor(models, ratings)
                from simulation.monte_carlo import MonteCarloSimulator
                sim = MonteCarloSimulator(groups, predictor, n_simulations=n_sims)
                with st.spinner(f"Corriendo {n_sims:,} simulaciones..."):
                    results_df = sim.run()
                results_df.to_csv(os.path.join(DATA_PROC, 'mc_results.csv'), index=False)
                st.success("✓ Listo. Recargá la pestaña Campeón.")
                st.dataframe(results_df.head(20), use_container_width=True, hide_index=True)
                st.cache_data.clear()

        mc = load_mc_results()
        if mc is not None:
            st.subheader("Última simulación")
            st.dataframe(mc, use_container_width=True, hide_index=True)


if __name__ == '__main__':
    main()
