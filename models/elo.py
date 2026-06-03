"""
models/elo.py
-------------
Sistema Elo para selecciones de fútbol.

Características:
- Rating inicial configurable (default 1500)
- K-factor dinámico por tipo de torneo y margen de goles
- Corrección por localía
- Decay temporal opcional (partidos viejos pesan menos)
- Exporta ratings finales y evolución histórica
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


class EloSystem:
    """
    Sistema Elo para selecciones nacionales.
    
    Parámetros
    ----------
    initial_rating : float
        Rating de inicio para equipos sin historia (default 1500)
    k_base : float
        Factor K base, escala la velocidad de ajuste (default 32)
    home_advantage : float
        Puntos sumados al equipo local en la predicción (default 100)
    """

    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_base: float = 32.0,
        home_advantage: float = 100.0,
    ):
        self.initial_rating = initial_rating
        self.k_base = k_base
        self.home_advantage = home_advantage
        self.ratings: dict[str, float] = {}
        self.history: list[dict] = []  # para graficar evolución

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------

    def _get_rating(self, team: str) -> float:
        return self.ratings.get(team, self.initial_rating)

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """Probabilidad de que A gane (o empate = 0.5 para A)."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def _goal_index(self, goal_diff: int) -> float:
        """
        Multiplicador por diferencia de goles (FIFA Elo adaptation).
        Evita que goleadas inflen artificialmente los ratings.
        """
        abs_diff = abs(goal_diff)
        if abs_diff <= 1:
            return 1.0
        elif abs_diff == 2:
            return 1.5
        elif abs_diff == 3:
            return 1.75
        else:
            return (11 + abs_diff) / 8.0

    def _k_factor(self, tournament_weight: float, goal_diff: int) -> float:
        """K-factor ajustado por importancia del partido y margen de goles."""
        return self.k_base * tournament_weight * self._goal_index(goal_diff)

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, verbose: bool = True) -> "EloSystem":
        """
        Procesa todos los partidos en orden cronológico y actualiza ratings.
        
        df debe tener: date, home_team, away_team, home_score, away_score,
                       neutral (bool), tournament_weight
        """
        self.ratings = {}
        self.history = []

        for _, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            h_score = row["home_score"]
            a_score = row["away_score"]
            if pd.isna(h_score) or pd.isna(a_score):
                continue
            neutral = row.get("neutral", False)
            t_weight = row.get("tournament_weight", 0.5)

            r_home = self._get_rating(home)
            r_away = self._get_rating(away)

            # Ventaja de localía (no aplica en campo neutral)
            advantage = 0.0 if neutral else self.home_advantage
            exp_home = self._expected_score(r_home + advantage, r_away)

            # Resultado real: 1 = gana local, 0.5 = empate, 0 = gana visitante
            goal_diff = int(h_score) - int(a_score)
            actual_home = 1.0 if goal_diff > 0 else (0.5 if goal_diff == 0 else 0.0)

            # K dinámico
            k = self._k_factor(t_weight, goal_diff)

            # Actualización
            delta = k * (actual_home - exp_home)
            self.ratings[home] = r_home + delta
            self.ratings[away] = r_away - delta

            # Guardar snapshot para evolución
            self.history.append({
                "date": row["date"],
                "home_team": home,
                "away_team": away,
                "home_rating_after": self.ratings[home],
                "away_rating_after": self.ratings[away],
                "result": row.get("result", "?"),
            })

        if verbose:
            top5 = sorted(self.ratings.items(), key=lambda x: -x[1])[:5]
            print(f"✓ Elo entrenado sobre {len(df):,} partidos")
            print(f"  Top 5 ratings: {[(t, round(r)) for t, r in top5]}")

        return self

    # ------------------------------------------------------------------
    # Predicción
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        home: str,
        away: str,
        neutral: bool = False,
    ) -> dict[str, float]:
        """
        Retorna probabilidades estimadas: {'home': p, 'draw': p, 'away': p}
        
        Usamos el método de conversión Elo → 3 resultados via Bradley-Terry
        con ajuste empírico para empates en fútbol.
        """
        r_home = self._get_rating(home)
        r_away = self._get_rating(away)
        advantage = 0.0 if neutral else self.home_advantage

        # P(home wins) en escala Elo
        p_home_win = self._expected_score(r_home + advantage, r_away)

        # Ajuste empírico para empates (≈27% en fútbol internacional)
        # Distribuimos la probabilidad de empate de forma simétrica
        draw_prob = max(0.10, 0.30 - 0.3 * abs(p_home_win - 0.5))
        remaining = 1.0 - draw_prob
        p_home = p_home_win * remaining + draw_prob * 0.5  # re-escalar
        p_away = (1 - p_home_win) * remaining + draw_prob * 0.5

        # Normalizar
        total = p_home + draw_prob + p_away
        return {
            "home": round(p_home / total, 4),
            "draw": round(draw_prob / total, 4),
            "away": round(p_away / total, 4),
        }

    def get_rating(self, team: str) -> float:
        return round(self._get_rating(team), 1)

    # ------------------------------------------------------------------
    # Exportar
    # ------------------------------------------------------------------

    def get_ratings_df(self) -> pd.DataFrame:
        """DataFrame con todos los ratings, ordenado de mayor a menor."""
        df = pd.DataFrame(
            [(team, rating) for team, rating in self.ratings.items()],
            columns=["team", "elo_rating"],
        ).sort_values("elo_rating", ascending=False).reset_index(drop=True)
        df.index += 1
        return df

    def get_history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def save(self, path: Optional[str] = None) -> None:
        path = path or str(PROCESSED_DIR / "elo_ratings.csv")
        self.get_ratings_df().to_csv(path, index=False)
        print(f"✓ Ratings guardados en {path}")


# ------------------------------------------------------------------
# Ejecución directa para testear
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data_loader import download_data, prepare_data

    print("=== ELO SYSTEM - World Cup 2026 Predictor ===\n")

    df_raw = download_data()
    df = prepare_data(df_raw, from_year=1990)

    elo = EloSystem(initial_rating=1500, k_base=32, home_advantage=100)
    elo.fit(df)

    print("\n--- Top 20 Selecciones ---")
    print(elo.get_ratings_df().head(20).to_string())

    print("\n--- Predicciones de ejemplo ---")
    matchups = [
        ("Argentina", "France", True),
        ("Brazil", "Germany", True),
        ("Spain", "England", True),
        ("Argentina", "Brazil", True),
    ]
    for home, away, neutral in matchups:
        proba = elo.predict_proba(home, away, neutral=neutral)
        print(f"  {home} vs {away}: "
              f"{home} {proba['home']*100:.1f}% | "
              f"Empate {proba['draw']*100:.1f}% | "
              f"{away} {proba['away']*100:.1f}%")

    elo.save()
    print("\n✓ Fase 1 completa.")
