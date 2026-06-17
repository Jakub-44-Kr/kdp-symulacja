"""
sobol.py — Globalna analiza wrażliwości metodą indeksów Sobola.

Realizuje metodykę z rozdz. 4.6:
  - próbkowanie Saltelli (quasi-Monte Carlo, sekwencja Sobola)
  - N·(n+2) uruchomień modelu (n=5 parametrów)
  - indeksy pierwszego rzędu S_i (wzór eq:Si) i całkowite S_Ti (wzór eq:STi)
  - test zbieżności: rosnące N, obserwacja stabilizacji indeksów
  - osobno dla systemów AC i DC

Biblioteka: SALib (Herman 2017).

Funkcja celu: jednostkowe zużycie energii E_per_km [kWh/km]
(energia netto na pantografie odniesiona do długości odcinka).
Normalizacja względem L usuwa trywialny, niemal liniowy wzrost energii
całkowitej z długością trasy, dzięki czemu dekompozycja wariancji ujawnia
wpływ pozostałych parametrów (prędkość, pochylenie) zamiast samej długości.

Autor: Jakub Król, PW WE, 2026
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

from energy import compute_energy
from parameters import OUTPUT_DIR, Parameters
from simulation import run_simulation

# ═══════════════════════════════════════════════════════════════════════════
#  DEFINICJA PROBLEMU (zakresy z Tabeli 4, jednostki SI)
# ═══════════════════════════════════════════════════════════════════════════


def build_problem(system: str) -> dict:
    """Problem SALib: 5 parametrów, rozkłady jednostajne, ZAKRESY PER SYSTEM (spójne z OAT).
    AC: v_max 100-400 km/h, P_nom 6-12 MW. DC: v_max 100-250 km/h (sufit), P_nom 6-9 MW."""
    system = system.upper()
    if system == "DC":
        v_bounds = [100 / 3.6, 250 / 3.6]
        P_bounds = [6e6, 9e6]
    else:
        v_bounds = [100 / 3.6, 400 / 3.6]
        P_bounds = [6e6, 12e6]
    return {
        "num_vars": 5,
        "names": ["v_max", "m", "P_nom", "gradient", "L"],
        "bounds": [
            v_bounds,
            [450_000.0, 750_000.0],
            P_bounds,
            [-5.0, 5.0],
            [50_000.0, 400_000.0],
        ],
    }


def _sobol_worker(args: tuple[np.ndarray, str, bool]) -> float:
    """
    Worker multiprocessing. Dostaje wektor parametrów (jeden wiersz macierzy
    próbek Saltelli) + system zasilania, zwraca E_per_km [kWh/km].

    Args:
        args: (param_vector, system, regen) gdzie param_vector to
              [v_max, m, P_nom, gradient, L] w SI.

    Returns:
        E_per_km — jednostkowe zużycie energii netto [kWh/km].
    """
    param_vector, system, regen = args
    v_max, m, P_nom, gradient, L = param_vector

    base = Parameters.base()
    p = base.with_changes(
        power_system=system,
        v_max=float(v_max),
        m=float(m),
        P_nom=float(P_nom),
        gradient=float(gradient),
        L=float(L),
        regen=bool(regen),
    )

    profile = [(0.0, p.L, p.gradient)]
    sim = run_simulation(p, profile)
    energy = compute_energy(sim, p)
    return energy.E_per_km  # kWh/km (E_pant_netto / L)


# ═══════════════════════════════════════════════════════════════════════════
#  EWALUACJA MODELU dla wszystkich próbek (równolegle)
# ═══════════════════════════════════════════════════════════════════════════


def evaluate_samples(
    param_values: np.ndarray,
    system: str,
    regen: bool = True,
    n_workers: int | None = None,
) -> np.ndarray:
    """
    Uruchamia model dla wszystkich wierszy macierzy próbek (równolegle).

    Args:
        param_values: macierz (N·(n+2)) × n próbek z SALib.
        system: "AC" lub "DC".
        n_workers: liczba procesów.

    Returns:
        Wektor wyjść Y (E_per_km [kWh/km]) o długości równej liczbie próbek.
    """
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 1)

    tasks = [(row, system, regen) for row in param_values]
    n_total = len(tasks)
    Y = np.zeros(n_total, dtype=np.float64)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # map zachowuje kolejność - kluczowe dla Sobola!
        for i, result in enumerate(executor.map(_sobol_worker, tasks, chunksize=16)):
            Y[i] = result
            if (i + 1) % 500 == 0 or (i + 1) == n_total:
                print(f"      {i + 1}/{n_total} przejazdów")

    return Y


# ═══════════════════════════════════════════════════════════════════════════
#  GŁÓWNA ANALIZA SOBOLA dla jednego systemu
# ═══════════════════════════════════════════════════════════════════════════


def run_sobol_for_system(
    system: str,
    N: int = 512,
    regen: bool = True,
    n_workers: int | None = None,
    seed: int = 42,
) -> dict:
    """
    Pełna analiza Sobola dla jednego systemu zasilania.

    Args:
        system: "AC" lub "DC".
        N: bazowa liczba próbek (uruchomień = N·(n+2)).
        n_workers: liczba procesów.
        seed: ziarno dla powtarzalności.

    Returns:
        Słownik z indeksami S1, ST, ich błędami (conf) i metadanymi.
    """
    problem = build_problem(system)
    n = problem["num_vars"]

    # Próbkowanie Saltelli (sekwencja Sobola)
    param_values = sobol_sample.sample(problem, N, calc_second_order=True, seed=seed)
    n_runs = len(param_values)
    print(f"    [{system}] N={N} → {n_runs} uruchomień modelu ({N}·({n}+2))")

    # Ewaluacja modelu
    t0 = time.perf_counter()
    Y = evaluate_samples(param_values, system, regen, n_workers)
    print(f"    [{system}] ewaluacja: {time.perf_counter() - t0:.1f} s")

    # Analiza Sobola
    Si = sobol_analyze.analyze(problem, Y, calc_second_order=True, seed=seed)

    return {
        "system": system,
        "regen": regen,
        "N": N,
        "n_runs": n_runs,
        "names": problem["names"],
        "S1": Si["S1"],
        "S1_conf": Si["S1_conf"],
        "ST": Si["ST"],
        "ST_conf": Si["ST_conf"],
        "S2": Si["S2"],  # interakcje drugiego rzędu (macierz)
        "S2_conf": Si["S2_conf"],
        "Y_mean": float(np.mean(Y)),
        "Y_std": float(np.std(Y)),
        "Y": Y,
        "param_values": param_values,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  TEST ZBIEŻNOŚCI — rosnące N
# ═══════════════════════════════════════════════════════════════════════════


def convergence_test(
    system: str = "AC",
    N_values: tuple[int, ...] = (128, 256, 512, 1024),
    n_workers: int | None = None,
) -> pd.DataFrame:
    """
    Test zbieżności indeksów Sobola: oblicza S_i i S_Ti dla rosnących N
    i sprawdza czy się stabilizują (metodyka rozdz. 4.6).

    Returns:
        DataFrame: kolumny N, parameter, S1, ST + błędy.
    """
    rows = []
    for N in N_values:
        print(f"  >>> Test zbieżności N={N}...")
        res = run_sobol_for_system(system, N=N, n_workers=n_workers)
        for i, name in enumerate(res["names"]):
            rows.append(
                {
                    "N": N,
                    "n_runs": res["n_runs"],
                    "parameter": name,
                    "S1": res["S1"][i],
                    "S1_conf": res["S1_conf"][i],
                    "ST": res["ST"][i],
                    "ST_conf": res["ST_conf"][i],
                }
            )
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
#  EKSPORT
# ═══════════════════════════════════════════════════════════════════════════


def export_sobol_indices(results: dict, save_dir=OUTPUT_DIR) -> dict:
    """Zapisuje indeksy Sobola do CSV (S1/ST + interakcje S2)."""
    save_dir.mkdir(parents=True, exist_ok=True)
    system = results["system"]
    names = results["names"]

    # S1 i ST
    df_main = pd.DataFrame(
        {
            "parameter": names,
            "S1": results["S1"],
            "S1_conf": results["S1_conf"],
            "ST": results["ST"],
            "ST_conf": results["ST_conf"],
        }
    )
    path_main = save_dir / f"sobol_indices_{system}.csv"
    df_main.to_csv(path_main, index=False, float_format="%.6g")

    # S2 - interakcje drugiego rzędu (macierz n×n, tylko górny trójkąt)
    S2 = results["S2"]
    rows_s2 = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rows_s2.append(
                {
                    "param_i": names[i],
                    "param_j": names[j],
                    "S2": S2[i, j],
                    "S2_conf": results["S2_conf"][i, j],
                }
            )
    df_s2 = pd.DataFrame(rows_s2)
    path_s2 = save_dir / f"sobol_interactions_{system}.csv"
    df_s2.to_csv(path_s2, index=False, float_format="%.6g")

    return {"main": path_main, "interactions": path_s2}


def print_sobol_report(results: dict) -> None:
    """Drukuje czytelny raport indeksów Sobola."""
    system = results["system"]
    names = results["names"]
    print()
    print("=" * 78)
    print(
        f"INDEKSY SOBOLA — system {system}  (N={results['N']}, "
        f"{results['n_runs']} uruchomień)"
    )
    print(
        f"  E/km: średnia = {results['Y_mean']:.2f} kWh/km, "
        f"odch.std = {results['Y_std']:.2f} kWh/km"
    )
    print("=" * 78)
    print(
        f"{'Parametr':>12} {'S_i (1.rz)':>14} {'S_Ti (całk)':>14} "
        f"{'S_Ti - S_i':>12} {'interakcje?':>12}"
    )
    print("-" * 70)

    # Sortuj wg ST malejąco
    order = np.argsort(results["ST"])[::-1]
    for idx in order:
        S1 = results["S1"][idx]
        ST = results["ST"][idx]
        diff = ST - S1
        interact = "TAK" if diff > 0.05 else "—"
        print(f"{names[idx]:>12} {S1:>14.4f} {ST:>14.4f} {diff:>12.4f} {interact:>12}")

    print("-" * 70)
    print(
        f"  Suma S_i = {np.sum(results['S1']):.4f}  "
        f"(≈1 → model addytywny; <1 → istotne interakcje)"
    )

    # Najsilniejsza interakcja drugiego rzędu
    S2 = results["S2"]
    max_s2 = 0.0
    max_pair = None
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if not np.isnan(S2[i, j]) and S2[i, j] > max_s2:
                max_s2 = S2[i, j]
                max_pair = (names[i], names[j])
    if max_pair:
        print(
            f"  Najsilniejsza interakcja 2.rz: {max_pair[0]} × {max_pair[1]} "
            f"= {max_s2:.4f}"
        )
    print("=" * 78)


# ═══════════════════════════════════════════════════════════════════════════
#  🚀 SZYBKI TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    n_cpu = os.cpu_count() or 4
    print("=" * 78)
    print("GLOBALNA ANALIZA WRAŻLIWOŚCI — INDEKSY SOBOLA")
    print(f"  Procesory: {n_cpu}, używam {n_cpu - 1} procesów")
    print("=" * 78)

    # Na start mniejsze N żeby szybko zobaczyć czy działa
    N_TEST = 256  # → 256·7 = 1792 uruchomień/system

    t0 = time.perf_counter()
    for system in ("AC", "DC"):
        print()
        print(f">>> System {system}...")
        results = run_sobol_for_system(system, N=N_TEST)
        print_sobol_report(results)
        paths = export_sobol_indices(results)
        for key, path in paths.items():
            print(f"    zapisano {key}: {path.name}")

    print()
    print(f"✓ Łączny czas: {time.perf_counter() - t0:.1f} s")
