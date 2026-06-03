"""
Monte Carlo Simulation — WC 2026 Predictor (versión optimizada)
Pre-calcula todas las probabilidades y simula en memoria pura.
"""

import numpy as np
import pandas as pd
import random
from collections import defaultdict
from itertools import combinations


def build_prob_table(teams, predictor_fn):
    """
    Pre-calcula probabilidades para todos los pares de equipos.
    Retorna dict: {(home, away): {'home_win': x, 'draw': y, 'away_win': z}}
    """
    table = {}
    pairs = list(combinations(teams, 2))
    print(f"   Pre-calculando {len(pairs)*2} matchups...")

    for h, a in pairs:
        pred = predictor_fn(h, a, neutral=True)
        if pred is None:
            pred = {'home_win': 0.38, 'draw': 0.24, 'away_win': 0.38}
        table[(h, a)] = pred
        # Invertido
        table[(a, h)] = {
            'home_win': pred['away_win'],
            'draw':     pred['draw'],
            'away_win': pred['home_win'],
        }

    print(f"   ✓ Tabla de probabilidades lista ({len(table)} pares)")
    return table


def sim_match_from_table(home, away, prob_table):
    """Simula un partido usando la tabla pre-calculada. Retorna (winner, result)."""
    pred = prob_table.get((home, away), {'home_win': 0.38, 'draw': 0.24, 'away_win': 0.38})
    r = random.random()
    if r < pred['home_win']:
        return home, 'H'
    elif r < pred['home_win'] + pred['draw']:
        return None, 'D'
    else:
        return away, 'A'


def sim_knockout(home, away, prob_table):
    """Partido eliminatorio — no hay empate (penales si empate)."""
    pred = prob_table.get((home, away), {'home_win': 0.38, 'draw': 0.24, 'away_win': 0.38})
    r = random.random()
    if r < pred['home_win']:
        return home
    elif r < pred['home_win'] + pred['draw']:
        return home if random.random() < 0.5 else away
    else:
        return away


def simulate_group(teams, prob_table):
    """Simula un grupo. Retorna equipos ordenados por puntos/DG/GF."""
    table = {t: {'pts': 0, 'gd': 0, 'gf': 0} for t in teams}

    for h, a in combinations(teams, 2):
        _, result = sim_match_from_table(h, a, prob_table)
        # Goles simulados rápido
        if result == 'H':
            gh, ga = np.random.poisson(1.6), np.random.poisson(0.9)
            gh = max(gh, ga + 1)
            table[h]['pts'] += 3
        elif result == 'A':
            gh, ga = np.random.poisson(0.9), np.random.poisson(1.6)
            ga = max(ga, gh + 1)
            table[a]['pts'] += 3
        else:
            gh = ga = np.random.poisson(1.1)
            table[h]['pts'] += 1
            table[a]['pts'] += 1

        table[h]['gf'] += gh; table[h]['gd'] += gh - ga
        table[a]['gf'] += ga; table[a]['gd'] += ga - gh

    return sorted(teams, key=lambda t: (table[t]['pts'], table[t]['gd'], table[t]['gf']), reverse=True), table


def simulate_tournament(groups, prob_table):
    """Simula el torneo completo. Retorna campeón y clasificados por fase."""
    # Fase de grupos
    first, second, thirds = [], [], []

    for grp_teams in groups.values():
        ranked, table = simulate_group(grp_teams, prob_table)
        first.append(ranked[0])
        second.append(ranked[1])
        thirds.append((table[ranked[2]]['pts'], table[ranked[2]]['gd'], table[ranked[2]]['gf'], ranked[2]))

    # 8 mejores terceros
    thirds.sort(reverse=True)
    best_thirds = [t[3] for t in thirds[:8]]

    qualified_32 = first + second + best_thirds
    random.shuffle(qualified_32)

    # Knockout
    round_16_teams = set(qualified_32)
    current = qualified_32.copy()

    for _ in range(5):  # R16, QF, SF, Final (32->16->8->4->2->1)
        if len(current) == 1:
            break
        next_round = []
        for i in range(0, len(current) - 1, 2):
            winner = sim_knockout(current[i], current[i+1], prob_table)
            next_round.append(winner)
        if len(current) % 2 == 1:
            next_round.append(current[-1])
        current = next_round

    champion = current[0] if current else None
    return champion, qualified_32


class MonteCarloSimulator:
    def __init__(self, groups, predictor_fn, n_simulations=5000):
        self.groups = groups
        self.predictor_fn = predictor_fn
        self.n_simulations = n_simulations
        self.results = None
        self.prob_table = None

    def run(self):
        all_teams = [t for grp in self.groups.values() for t in grp]
        print(f"🎲 Corriendo {self.n_simulations:,} simulaciones del Mundial 2026...")

        # Pre-calcular probabilidades UNA sola vez
        self.prob_table = build_prob_table(all_teams, self.predictor_fn)

        champion_count = defaultdict(int)
        qualified_count = defaultdict(int)

        for sim in range(self.n_simulations):
            if (sim + 1) % 500 == 0:
                print(f"   {sim+1:,}/{self.n_simulations:,}...")
            try:
                champion, qualified = simulate_tournament(self.groups, self.prob_table)
                if champion:
                    champion_count[champion] += 1
                for t in qualified:
                    qualified_count[t] += 1
            except Exception:
                continue

        N = self.n_simulations
        rows = []
        for team in all_teams:
            rows.append({
                'team':      team,
                'qualified': round(qualified_count[team] / N * 100, 1),
                'champion':  round(champion_count[team]  / N * 100, 1),
            })

        self.results = pd.DataFrame(rows).sort_values('champion', ascending=False).reset_index(drop=True)
        return self.results

    def print_summary(self, top_n=20):
        if self.results is None:
            return
        print(f"\n{'='*55}")
        print(f"  TOP {top_n} — PROBABILIDADES MUNDIALISTAS")
        print(f"{'='*55}")
        print(f"  {'Equipo':<25} {'Clasifica':>10} {'Campeón':>10}")
        print(f"  {'-'*48}")
        for _, row in self.results.head(top_n).iterrows():
            print(f"  {row['team']:<25} {row['qualified']:>9.1f}%  {row['champion']:>9.1f}%")
