"""
Gradient Boosting Model — WC 2026 Predictor
XGBoost con features de Elo, Dixon-Coles, forma reciente y contexto del torneo.
Target: resultado del partido (0=local gana, 1=empate, 2=visitante gana)
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score
import warnings
warnings.filterwarnings('ignore')


def compute_recent_form(matches, team, date, n=10):
    """
    Calcula estadísticas de forma de los últimos N partidos de un equipo.
    Retorna: win_rate, draw_rate, goals_scored_avg, goals_conceded_avg
    """
    team_matches = matches[
        ((matches['home_team'] == team) | (matches['away_team'] == team)) &
        (matches['date'] < date)
    ].sort_values('date').tail(n)
    
    if len(team_matches) == 0:
        return {'wins': 0.0, 'draws': 0.0, 'gf': 1.0, 'ga': 1.0, 'gd': 0.0, 'n_matches': 0}
    
    wins, draws, gf_total, ga_total = 0, 0, 0, 0
    
    for _, row in team_matches.iterrows():
        if row['home_team'] == team:
            gf, ga = row['home_score'], row['away_score']
        else:
            gf, ga = row['away_score'], row['home_score']
        
        gf_total += gf
        ga_total += ga
        if gf > ga: wins += 1
        elif gf == ga: draws += 1
    
    n_m = len(team_matches)
    return {
        'wins':  wins / n_m,
        'draws': draws / n_m,
        'gf':    gf_total / n_m,
        'ga':    ga_total / n_m,
        'gd':    (gf_total - ga_total) / n_m,
        'n_matches': n_m
    }


def build_features(matches, elo_ratings=None, dc_model=None):
    """
    Construye el DataFrame de features para XGBoost.
    
    Features generadas:
    - Elo difference (home - away)
    - Elo absolutos de cada equipo
    - Forma reciente (últimos 10 partidos): win rate, goles, GD
    - xG esperado de Dixon-Coles (si disponible)
    - Neutralidad del campo
    - Confederación de cada equipo
    """
    print("🔨 Construyendo features para XGBoost...")
    
    matches = matches.copy()
    matches['date'] = pd.to_datetime(matches['date'])
    matches = matches.sort_values('date').reset_index(drop=True)
    
    # Resultado como target
    def get_result(row):
        if row['home_score'] > row['away_score']:   return 0  # local
        elif row['home_score'] == row['away_score']: return 1  # empate
        else:                                         return 2  # visitante
    
    matches['result'] = matches.apply(get_result, axis=1)
    
    feature_rows = []
    
    for idx, row in matches.iterrows():
        h, a = row['home_team'], row['away_team']
        date = row['date']
        
        # --- Elo ---
        elo_h = elo_ratings.get(h, 1500) if elo_ratings else 1500
        elo_a = elo_ratings.get(a, 1500) if elo_ratings else 1500
        elo_diff = elo_h - elo_a
        
        # --- Forma reciente ---
        form_h = compute_recent_form(matches.iloc[:idx], h, date, n=10)
        form_a = compute_recent_form(matches.iloc[:idx], a, date, n=10)
        
        # --- xG Dixon-Coles ---
        xg_h, xg_a = np.nan, np.nan
        dc_home_win, dc_draw, dc_away_win = np.nan, np.nan, np.nan
        if dc_model is not None and dc_model.fitted:
            pred = dc_model.predict_result(h, a)
            if pred:
                xg_h = pred['xG_home']
                xg_a = pred['xG_away']
                dc_home_win = pred['home_win']
                dc_draw     = pred['draw']
                dc_away_win = pred['away_win']
        
        feature_rows.append({
            # Elo
            'elo_diff':      elo_diff,
            'elo_home':      elo_h,
            'elo_away':      elo_a,
            'elo_ratio':     elo_h / max(elo_a, 1),
            
            # Forma home
            'form_h_wins':   form_h['wins'],
            'form_h_draws':  form_h['draws'],
            'form_h_gf':     form_h['gf'],
            'form_h_ga':     form_h['ga'],
            'form_h_gd':     form_h['gd'],
            
            # Forma away
            'form_a_wins':   form_a['wins'],
            'form_a_draws':  form_a['draws'],
            'form_a_gf':     form_a['gf'],
            'form_a_ga':     form_a['ga'],
            'form_a_gd':     form_a['gd'],
            
            # Diferencias de forma
            'form_wins_diff': form_h['wins'] - form_a['wins'],
            'form_gd_diff':   form_h['gd']   - form_a['gd'],
            
            # xG / Dixon-Coles
            'xg_home':       xg_h,
            'xg_away':       xg_a,
            'xg_diff':       (xg_h - xg_a) if not np.isnan(xg_h) else np.nan,
            'dc_home_win':   dc_home_win,
            'dc_draw':       dc_draw,
            'dc_away_win':   dc_away_win,
            
            # Contexto
            'neutral':       int(row.get('neutral', False)),
            
            # Target
            'result':        row['result'],
            'date':          date,
            'home_team':     h,
            'away_team':     a,
        })
    
    df = pd.DataFrame(feature_rows)
    print(f"   Features construidas: {len(df)} partidos, {len(df.columns)} columnas")
    return df


class GBMatchPredictor:
    """
    XGBoost para predicción de resultados de partidos de fútbol.
    Incluye validación temporal walk-forward.
    """
    
    FEATURE_COLS = [
        'elo_diff', 'elo_home', 'elo_away', 'elo_ratio',
        'form_h_wins', 'form_h_draws', 'form_h_gf', 'form_h_ga', 'form_h_gd',
        'form_a_wins', 'form_a_draws', 'form_a_gf', 'form_a_ga', 'form_a_gd',
        'form_wins_diff', 'form_gd_diff',
        'xg_home', 'xg_away', 'xg_diff',
        'dc_home_win', 'dc_draw', 'dc_away_win',
        'neutral',
    ]
    
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1,
        )
        self.fitted = False
        self.feature_importance_ = None
    
    def fit(self, feature_df, validate=True):
        """Entrena el modelo. feature_df debe tener columnas FEATURE_COLS + 'result'."""
        print("🚀 Entrenando XGBoost...")
        
        # Usar solo partidos con suficientes datos de forma
        df = feature_df.dropna(subset=['elo_diff', 'form_h_wins', 'form_a_wins']).copy()
        
        X = df[self.FEATURE_COLS].fillna(df[self.FEATURE_COLS].median())
        y = df['result']
        
        if validate:
            self._walk_forward_validate(X, y, df['date'])
        
        # Entrenamiento final sobre todos los datos
        self.model.fit(X, y)
        self.fitted = True
        
        # Feature importance
        self.feature_importance_ = pd.Series(
            self.model.feature_importances_,
            index=self.FEATURE_COLS
        ).sort_values(ascending=False)
        
        print("\n📊 Top 10 features más importantes:")
        print(self.feature_importance_.head(10).to_string())
        
        return self
    
    def _walk_forward_validate(self, X, y, dates, n_splits=5):
        """Validación walk-forward respetando el orden temporal."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        logloss_scores, acc_scores = [], []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            m = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric='mlogloss',
                random_state=42, n_jobs=-1
            )
            m.fit(X_tr, y_tr)
            probs = m.predict_proba(X_val)
            
            ll  = log_loss(y_val, probs)
            acc = accuracy_score(y_val, probs.argmax(axis=1))
            logloss_scores.append(ll)
            acc_scores.append(acc)
        
        print(f"   Walk-forward CV ({n_splits} folds):")
        print(f"   Log-loss: {np.mean(logloss_scores):.4f} ± {np.std(logloss_scores):.4f}")
        print(f"   Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    
    def predict_proba(self, home_team, away_team, feature_row):
        """
        Predice probabilidades [home_win, draw, away_win] para un partido.
        feature_row: dict con los valores de las features.
        """
        if not self.fitted:
            raise RuntimeError("Modelo no entrenado.")
        
        row = pd.DataFrame([feature_row])[self.FEATURE_COLS]
        row = row.fillna(0)
        probs = self.model.predict_proba(row)[0]
        
        return {
            'home_win': round(probs[0], 4),
            'draw':     round(probs[1], 4),
            'away_win': round(probs[2], 4),
        }
    
    def predict_from_ratings(self, home_team, away_team, elo_ratings, dc_model=None, neutral=False):
        """
        Predicción directa dado el nombre de los equipos y los ratings.
        Usa forma reciente = media global (para predicciones futuras).
        """
        elo_h = elo_ratings.get(home_team, 1500)
        elo_a = elo_ratings.get(away_team, 1500)
        
        xg_h, xg_a = 1.3, 1.1
        dc_hw, dc_d, dc_aw = 0.40, 0.25, 0.35
        
        if dc_model and dc_model.fitted:
            pred = dc_model.predict_result(home_team, away_team)
            if pred:
                xg_h, xg_a = pred['xG_home'], pred['xG_away']
                dc_hw, dc_d, dc_aw = pred['home_win'], pred['draw'], pred['away_win']
        
        feature_row = {
            'elo_diff':       elo_h - elo_a,
            'elo_home':       elo_h,
            'elo_away':       elo_a,
            'elo_ratio':      elo_h / max(elo_a, 1),
            'form_h_wins':    0.5,
            'form_h_draws':   0.25,
            'form_h_gf':      1.4,
            'form_h_ga':      1.0,
            'form_h_gd':      0.4,
            'form_a_wins':    0.5,
            'form_a_draws':   0.25,
            'form_a_gf':      1.4,
            'form_a_ga':      1.0,
            'form_a_gd':      0.4,
            'form_wins_diff': 0.0,
            'form_gd_diff':   0.0,
            'xg_home':        xg_h,
            'xg_away':        xg_a,
            'xg_diff':        xg_h - xg_a,
            'dc_home_win':    dc_hw,
            'dc_draw':        dc_d,
            'dc_away_win':    dc_aw,
            'neutral':        int(neutral),
        }
        
        return self.predict_proba(home_team, away_team, feature_row)
