"""
WC 2026 Predictor — Pipeline Principal
Uso:
    python pipeline.py              # Fase 1: datos + Elo
    python pipeline.py --fase 2     # Fase 2: Dixon-Coles + XGBoost
    python pipeline.py --fase 3     # Fase 3: Monte Carlo
    python pipeline.py --fase 3 --sims 10000
    python pipeline.py --prode      # Prode fase de grupos
    python pipeline.py --force      # Fuerza re-entrenamiento
    streamlit run dashboard/app.py  # Fase 4: Dashboard
"""

import argparse
import os
import sys
import json
import pickle
from itertools import combinations
import pandas as pd

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_RAW  = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROC = os.path.join(BASE_DIR, 'data', 'processed')
os.makedirs(DATA_RAW,  exist_ok=True)
os.makedirs(DATA_PROC, exist_ok=True)

sys.path.insert(0, BASE_DIR)

from data_loader import download_data, prepare_data
from models.elo import EloSystem


def load_groups():
    path = os.path.join(BASE_DIR, 'data', 'wc2026_groups.json')
    with open(path) as f:
        data = json.load(f)
    return data['groups'] if 'groups' in data else data


def get_elo_ratings(elo):
    df   = elo.get_ratings_df()
    cols = df.columns.tolist()
    team_col = next((c for c in cols if 'team'   in c.lower()), cols[0])
    rat_col  = next((c for c in cols if 'elo'    in c.lower() or 'rating' in c.lower()), cols[-1])
    return dict(zip(df[team_col], df[rat_col]))


def predict_elo(elo, home, away):
    try:
        return elo.predict_proba(home, away)
    except Exception:
        return None


def make_predictor(models, ratings):
    def predictor(home, away, neutral=False):
        preds, weights = [], []
        elo_h  = ratings.get(home, 1500)
        elo_a  = ratings.get(away, 1500)
        diff   = elo_h - elo_a
        p_h    = 1 / (1 + 10 ** (-diff / 400))
        draw_p = 0.22
        preds.append({'home_win': p_h*(1-draw_p), 'draw': draw_p, 'away_win': (1-p_h)*(1-draw_p)})
        weights.append(1)
        if 'dc' in models:
            p = models['dc'].predict_result(home, away)
            if p: preds.append(p); weights.append(2)
        if 'gb' in models:
            p = models['gb'].predict_from_ratings(home, away, ratings, dc_model=models.get('dc'), neutral=neutral)
            if p: preds.append(p); weights.append(2)
        w_total = sum(weights)
        return {
            'home_win': sum(p['home_win']*w for p,w in zip(preds,weights)) / w_total,
            'draw':     sum(p['draw']    *w for p,w in zip(preds,weights)) / w_total,
            'away_win': sum(p['away_win']*w for p,w in zip(preds,weights)) / w_total,
        }
    return predictor


def load_models_and_ratings():
    raw_df  = download_data()
    matches = prepare_data(raw_df)
    elo     = EloSystem()
    elo.fit(matches)
    ratings = get_elo_ratings(elo)

    models = {}
    dc_path  = os.path.join(DATA_PROC, 'dixon_coles.pkl')
    xgb_path = os.path.join(DATA_PROC, 'xgboost_model.pkl')
    if os.path.exists(dc_path):
        with open(dc_path, 'rb') as f:
            models['dc'] = pickle.load(f)
    if os.path.exists(xgb_path):
        with open(xgb_path, 'rb') as f:
            models['gb'] = pickle.load(f)

    return models, ratings, matches, elo


# ── FASE 1 ─────────────────────────────────────────────────────────────────────
def run_fase1(force=False):
    print("=" * 55)
    print("  WC 2026 PREDICTOR — FASE 1: DATOS + ELO")
    print("=" * 55)

    raw_df  = download_data(force=force)
    matches = prepare_data(raw_df)
    elo     = EloSystem()
    elo.fit(matches)

    ratings_df = elo.get_ratings_df()
    ratings_df.to_csv(os.path.join(DATA_PROC, 'elo_ratings.csv'), index=False)

    print(f"\n--- Top 30 Selecciones (Elo) ---")
    print(ratings_df.head(30).to_string())

    matchups = [
        ('Argentina', 'France'), ('Brazil', 'Germany'),
        ('Spain', 'England'), ('United States', 'Mexico'),
        ('Argentina', 'Brazil'), ('France', 'Germany'),
    ]
    print(f"\n--- Predicciones clave ---")
    print(f"  {'Partido':<40} {'Local':>8}   {'Empate':>8}   {'Visit.':>8}")
    print(f"  {'-'*63}")
    for h, a in matchups:
        pred = predict_elo(elo, h, a)
        if pred:
            hw = pred.get('home_win', pred.get('home', 0))
            dw = pred.get('draw', 0)
            aw = pred.get('away_win', pred.get('away', 0))
            print(f"  {h+' vs '+a:<40} {hw*100:>7.1f}%   {dw*100:>7.1f}%   {aw*100:>7.1f}%")

    print(f"\n✓ Fase 1 completada. Próximo: python pipeline.py --fase 2")
    return matches, elo


# ── FASE 2 ─────────────────────────────────────────────────────────────────────
def run_fase2(force=False):
    from models.dixon_coles import DixonColesModel
    from models.gradient_boosting import GBMatchPredictor, build_features

    print("=" * 55)
    print("  WC 2026 PREDICTOR — FASE 2: DIXON-COLES + XGBOOST")
    print("=" * 55)

    raw_df  = download_data()
    matches = prepare_data(raw_df)
    elo     = EloSystem()
    elo.fit(matches)
    ratings = get_elo_ratings(elo)

    if not os.path.exists(os.path.join(DATA_PROC, 'elo_ratings.csv')):
        elo.get_ratings_df().to_csv(os.path.join(DATA_PROC, 'elo_ratings.csv'), index=False)

    # Dixon-Coles
    dc_path = os.path.join(DATA_PROC, 'dixon_coles.pkl')
    if not force and os.path.exists(dc_path):
        print("\n✓ Dixon-Coles cargando desde disco...")
        with open(dc_path, 'rb') as f:
            dc = pickle.load(f)
    else:
        print("\n🔧 Entrenando Dixon-Coles...")
        groups   = load_groups()
        wc_teams = [t for grp in groups.values() for t in grp]
        print(f"   Equipos del mundial: {len(wc_teams)}")
        recent = matches[
            (matches['date'] >= '2010-01-01') &
            (matches['home_team'].isin(wc_teams) | matches['away_team'].isin(wc_teams))
        ].dropna(subset=['home_score', 'away_score']).copy()
        print(f"   Partidos disponibles: {len(recent)}")
        dc = DixonColesModel(xi=0.002, min_matches=3)
        dc.fit(recent)
        with open(dc_path, 'wb') as f:
            pickle.dump(dc, f)

    top_dc = dc.top_teams(20)
    if not top_dc.empty:
        print("\n--- Top 20 Dixon-Coles ---")
        print(top_dc.to_string(index=False))

    # XGBoost
    xgb_path = os.path.join(DATA_PROC, 'xgboost_model.pkl')
    if not force and os.path.exists(xgb_path):
        print("\n✓ XGBoost cargando desde disco...")
        with open(xgb_path, 'rb') as f:
            gb = pickle.load(f)
    else:
        print("\n🚀 Entrenando XGBoost...")
        recent_xgb  = matches[matches['date'] >= '2010-01-01'].copy()
        features_df = build_features(recent_xgb, elo_ratings=ratings, dc_model=dc)
        gb = GBMatchPredictor()
        gb.fit(features_df, validate=True)
        with open(xgb_path, 'wb') as f:
            pickle.dump(gb, f)

    models    = {'dc': dc, 'gb': gb}
    predictor = make_predictor(models, ratings)

    matchups = [
        ('Argentina', 'France', False), ('Brazil', 'Germany', False),
        ('Spain', 'England', False), ('United States', 'Mexico', True),
        ('Argentina', 'Brazil', False), ('France', 'Germany', False),
        ('Morocco', 'Portugal', False), ('Japan', 'Colombia', False),
    ]
    print(f"\n--- Predicciones ENSEMBLE ---")
    print(f"  {'Partido':<35} {'Local':>8}  {'Empate':>8}  {'Visit.':>8}")
    print(f"  {'-'*65}")
    results = []
    for h, a, neutral in matchups:
        pred  = predictor(h, a, neutral)
        label = f"{h} vs {a}"
        print(f"  {label:<35} {pred['home_win']*100:>7.1f}%  {pred['draw']*100:>7.1f}%  {pred['away_win']*100:>7.1f}%")
        results.append({'match': label, **pred})
    pd.DataFrame(results).to_csv(os.path.join(DATA_PROC, 'ensemble_predictions.csv'), index=False)

    print(f"\n✓ Fase 2 completada. Próximo: python pipeline.py --fase 3")
    return dc, gb, ratings


# ── FASE 3 ─────────────────────────────────────────────────────────────────────
def run_fase3(n_sims=5000):
    from simulation.monte_carlo import MonteCarloSimulator

    print("=" * 55)
    print("  WC 2026 PREDICTOR — FASE 3: MONTE CARLO")
    print("=" * 55)

    if not os.path.exists(os.path.join(DATA_PROC, 'xgboost_model.pkl')):
        print("⚠ Corré --fase 2 primero.")
        return

    models, ratings, _, _ = load_models_and_ratings()
    groups    = load_groups()
    predictor = make_predictor(models, ratings)

    sim     = MonteCarloSimulator(groups, predictor, n_simulations=n_sims)
    results = sim.run()
    sim.print_summary(20)

    results.to_csv(os.path.join(DATA_PROC, 'mc_results.csv'), index=False)
    print(f"\n✓ Fase 3 completada.")
    print(f"  Próximo: streamlit run dashboard/app.py")
    return results


# ── PRODE ──────────────────────────────────────────────────────────────────────
def run_prode():
    print("=" * 65)
    print("  WC 2026 — PRODE FASE DE GRUPOS")
    print("=" * 65)

    models, ratings, _, _ = load_models_and_ratings()
    groups    = load_groups()
    predictor = make_predictor(models, ratings)

    all_rows = []

    for group_name, teams in sorted(groups.items()):
        print(f"\n{'─'*65}")
        print(f"  GRUPO {group_name} — {' · '.join(teams)}")
        print(f"{'─'*65}")
        print(f"  {'Partido':<35} {'Local':>7}  {'Empate':>7}  {'Visit.':>7}  Fav.")
        print(f"  {'·'*60}")

        for home, away in combinations(teams, 2):
            pred  = predictor(home, away, neutral=True)
            hw    = pred['home_win'] * 100
            dw    = pred['draw']     * 100
            aw    = pred['away_win'] * 100
            fav   = home if hw > aw else (away if aw > hw else 'Empate')
            label = f"{home} vs {away}"
            print(f"  {label:<35} {hw:>6.1f}%  {dw:>6.1f}%  {aw:>6.1f}%  {fav}")
            all_rows.append({
                'group': group_name, 'home': home, 'away': away,
                'home_win': round(hw, 1), 'draw': round(dw, 1), 'away_win': round(aw, 1), 'fav': fav
            })

    print(f"\n{'='*65}")
    print(f"  Ensemble: Elo + Dixon-Coles + XGBoost (campo neutral)")
    print(f"{'='*65}")

    pd.DataFrame(all_rows).to_csv(os.path.join(DATA_PROC, 'prode_grupos.csv'), index=False)
    print(f"\n  Guardado en data/processed/prode_grupos.csv")


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WC 2026 Predictor')
    parser.add_argument('--fase',  type=int, default=1)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--sims',  type=int, default=5000)
    parser.add_argument('--prode', action='store_true')
    args = parser.parse_args()

    if args.prode:
        run_prode()
    elif args.fase == 1:
        run_fase1(force=args.force)
    elif args.fase == 2:
        run_fase2(force=args.force)
    elif args.fase == 3:
        run_fase3(n_sims=args.sims)
    else:
        print(f"❌ Fase {args.fase} no reconocida.")
