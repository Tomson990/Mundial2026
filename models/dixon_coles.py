"""
Dixon-Coles Model — WC 2026 Predictor
Estima parámetros de ataque/defensa por equipo via máxima verosimilitud.
Modela goles como Poisson bivariado con corrección Dixon-Coles para scores bajos.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')


def rho_correction(goals_h, goals_a, lambda_h, lambda_a, rho):
    """Corrección Dixon-Coles para dependencia entre goles bajos."""
    if goals_h == 0 and goals_a == 0:
        return 1 - lambda_h * lambda_a * rho
    elif goals_h == 0 and goals_a == 1:
        return 1 + lambda_h * rho
    elif goals_h == 1 and goals_a == 0:
        return 1 + lambda_a * rho
    elif goals_h == 1 and goals_a == 1:
        return 1 - rho
    else:
        return 1.0


def dixon_coles_log_likelihood(params, teams, matches, weight_func=None):
    """
    Log-verosimilitud del modelo Dixon-Coles.
    
    params: [ataque_t1, ..., ataque_tN, defensa_t1, ..., defensa_tN, home_adv, rho]
    """
    n_teams = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}
    
    attack  = params[:n_teams]
    defense = params[n_teams:2*n_teams]
    home_adv = params[2*n_teams]
    rho      = params[2*n_teams + 1]
    
    log_lik = 0.0
    
    matches = matches.dropna(subset=["home_score", "away_score"])
    matches = matches.dropna(subset=["home_score", "away_score"])
    matches = matches.dropna(subset=['home_score', 'away_score']).copy()
    for _, row in matches.iterrows():
        h = team_idx.get(row['home_team'])
        a = team_idx.get(row['away_team'])
        if h is None or a is None:
            continue
        
        lambda_h = np.exp(attack[h] - defense[a] + home_adv)
        lambda_a = np.exp(attack[a] - defense[h])
        
        gh = int(row['home_score'])
        ga = int(row['away_score'])
        
        # Probabilidad Poisson × corrección DC
        prob = (poisson.pmf(gh, lambda_h) * 
                poisson.pmf(ga, lambda_a) * 
                rho_correction(gh, ga, lambda_h, lambda_a, rho))
        
        if prob <= 0:
            prob = 1e-10
            
        w = row.get('weight', 1.0) if weight_func else 1.0
        log_lik += w * np.log(prob)
    
    return -log_lik  # negativo porque scipy minimiza


def time_weight(dates, xi=0.002):
    """Peso exponencial temporal — partidos recientes pesan más."""
    max_date = pd.to_datetime(dates).max()
    days_ago = (max_date - pd.to_datetime(dates)).dt.days
    return np.exp(-xi * days_ago)


class DixonColesModel:
    """
    Modelo Dixon-Coles completo para predicción de partidos de fútbol.
    """
    
    def __init__(self, xi=0.002, min_matches=5):
        """
        xi: parámetro de decay temporal (0.002 ≈ peso mitad cada ~1 año)
        min_matches: mínimo de partidos para incluir un equipo
        """
        self.xi = xi
        self.min_matches = min_matches
        self.params = None
        self.teams = None
        self.team_idx = None
        self.fitted = False
    
    def _filter_teams(self, matches):
        """Retiene solo equipos con suficientes partidos."""
        counts = pd.concat([
            matches['home_team'].value_counts(),
            matches['away_team'].value_counts()
        ]).groupby(level=0).sum()
        return counts[counts >= self.min_matches].index.tolist()
    
    def fit(self, matches):
        """
        Entrena el modelo sobre el DataFrame de partidos.
        Columnas requeridas: home_team, away_team, home_score, away_score, date
        """
        print("🔧 Entrenando Dixon-Coles...")
        
        matches = matches.copy()
        matches['date'] = pd.to_datetime(matches['date'])
        matches['weight'] = time_weight(matches['date'], self.xi)
        
        # Filtrar equipos con pocos datos
        valid_teams = self._filter_teams(matches)
        matches = matches[
            matches['home_team'].isin(valid_teams) & 
            matches['away_team'].isin(valid_teams)
        ].copy()
        
        self.teams = sorted(valid_teams)
        self.team_idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        
        print(f"   Equipos en el modelo: {n}")
        print(f"   Partidos de entrenamiento: {len(matches)}")
        
        # Parámetros iniciales
        x0 = np.zeros(2 * n + 2)
        x0[2*n] = 0.3    # home advantage
        x0[2*n+1] = -0.1  # rho (correlación)
        
        # Restricción: suma de ataques = 0 (identificabilidad)
        constraints = [{
            'type': 'eq',
            'fun': lambda x: np.sum(x[:n])
        }]
        
        result = minimize(
            dixon_coles_log_likelihood,
            x0,
            args=(self.teams, matches),
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-8}
        )
        
        self.params = result.x
        self.fitted = True
        
        print(f"   Convergencia: {'✓' if result.success else '⚠ parcial'}")
        print(f"   Log-verosimilitud: {-result.fun:.1f}")
        
        return self
    
    def _get_lambdas(self, home_team, away_team):
        """Calcula lambdas esperados para un partido."""
        n = len(self.teams)
        h = self.team_idx.get(home_team)
        a = self.team_idx.get(away_team)
        
        if h is None or a is None:
            return None, None
        
        attack  = self.params[:n]
        defense = self.params[n:2*n]
        home_adv = self.params[2*n]
        
        lambda_h = np.exp(attack[h] - defense[a] + home_adv)
        lambda_a = np.exp(attack[a] - defense[h])
        
        return lambda_h, lambda_a
    
    def predict_scoreline(self, home_team, away_team, max_goals=8):
        """
        Retorna matriz de probabilidades de marcadores (max_goals x max_goals).
        """
        lambda_h, lambda_a = self._get_lambdas(home_team, away_team)
        if lambda_h is None:
            return None
        
        rho = self.params[2*len(self.teams)+1]
        
        matrix = np.zeros((max_goals+1, max_goals+1))
        for gh in range(max_goals+1):
            for ga in range(max_goals+1):
                prob = (poisson.pmf(gh, lambda_h) * 
                        poisson.pmf(ga, lambda_a) * 
                        rho_correction(gh, ga, lambda_h, lambda_a, rho))
                matrix[gh, ga] = max(prob, 0)
        
        # Normalizar
        matrix /= matrix.sum()
        return matrix
    
    def predict_result(self, home_team, away_team, neutral=False):
        """
        Retorna dict con probabilidades de: home_win, draw, away_win,
        expected_goals_h, expected_goals_a.
        """
        matrix = self.predict_scoreline(home_team, away_team)
        if matrix is None:
            return None
        
        home_win = np.sum(np.tril(matrix, -1))
        draw     = np.sum(np.diag(matrix))
        away_win = np.sum(np.triu(matrix, 1))
        
        lambda_h, lambda_a = self._get_lambdas(home_team, away_team)
        
        return {
            'home_win': round(home_win, 4),
            'draw':     round(draw, 4),
            'away_win': round(away_win, 4),
            'xG_home':  round(lambda_h, 2),
            'xG_away':  round(lambda_a, 2),
        }
    
    def get_team_params(self):
        """DataFrame con parámetros de ataque/defensa por equipo."""
        if not self.fitted:
            raise RuntimeError("Modelo no entrenado.")
        n = len(self.teams)
        return pd.DataFrame({
            'team':    self.teams,
            'attack':  self.params[:n],
            'defense': self.params[n:2*n],
            'overall': self.params[:n] - self.params[n:2*n]
        }).sort_values('overall', ascending=False).reset_index(drop=True)
    
    def top_teams(self, n=20):
        """Top N equipos por parámetro overall (ataque - defensa)."""
        return self.get_team_params().head(n)
