"""
sensitivity.py — Analiza wrażliwości modelu KDP.

Realizuje dwa komplementarne podejścia OAT (One-At-a-Time) z rozdz. 4.6 i 5.3:

  1. SWEEP OAT — każdy parametr zmieniany po całym zakresie z Tabeli 4,
     daje krzywe E(parametr). Łącznie ~50 przejazdów + bazowy.

  2. ELASTYCZNOŚCI OAT — perturbacja ±10% wokół punktu bazowego,
     wskaźnik S_i^OAT = (ΔE/E₀)/(Δp/p₀) (wzór 4.6), ranking ważności
     + sprawdzenie symetrii (perturbacja + vs -).
     Funkcją celu jest jednostkowe zużycie energii E_per_km [kWh/km].

Analiza prowadzona osobno dla systemów AC i DC.

UWAGA: sweep OAT (część 1) zapisuje w CSV pełen zestaw metryk, w tym
E_pant_netto i E_per_km — wybór metryki do wykresu/tabeli następuje na
etapie post-processingu. Część 2 (elastyczności) liczy wskaźniki wprost
na E_per_km.

Autor: Jakub Król, PW WE, 2026
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import pandas as pd

from energy import compute_energy
from parameters import OUTPUT_DIR, Parameters
from simulation import run_simulation

# ═══════════════════════════════════════════════════════════════════════════
#  DEFINICJA ZAKRESÓW OAT (Tabela 4 + tekst "Liczba scenariuszy")
# ═══════════════════════════════════════════════════════════════════════════


# Zakresy sweepów - klucz = nazwa parametru, wartość = lista wartości w jednostkach SI
def _build_sweep_ranges() -> dict[str, dict]:
    """
    Buduje słownik zakresów OAT. Każdy parametr ma:
      - 'values': lista wartości (SI)
      - 'label': etykieta do wykresów
      - 'unit': jednostka do wyświetlania
      - 'display': funkcja konwersji SI → jednostka wyświetlana
    """
    return {
        "v_max": {
            "values": [
                (v / 3.6) for v in range(250, 401, 10)
            ],  # 250-400 km/h, 16 wartości
            "label": "Prędkość eksploatacyjna $v_{max}$",
            "unit": "km/h",
            "display": lambda x: x * 3.6,
        },
        "m": {
            "values": [450_000.0 + 50_000.0 * i for i in range(7)],
            # 450-750 t co 50 t (7 wartości) — zakres = klasy/granice Sobola
            "label": "Masa składu $m$",
            "unit": "t",
            "display": lambda x: x / 1000.0,
        },
        "P_nom": {
            "values": [6e6 + 1e6 * i for i in range(7)],  # 6-12 MW co 1 MW (7)
            "label": "Moc znamionowa $P$",
            "unit": "MW",
            "display": lambda x: x / 1e6,
            # Dla DC sufit trakcji = 6 MW — wartości > 6 MW pomijane (P_eff stałe).
            "cap_si": {"DC": 6e6},
        },
        "gradient": {
            "values": [float(i) for i in range(-5, 6)],  # -5 do +5‰, 11 wartości
            "label": "Pochylenie trasy $i$",
            "unit": "‰",
            "display": lambda x: x,
        },
        "L": {
            "values": [
                (L * 1000.0) for L in range(50, 401, 25)
            ],  # 50-400 km, 15 wartości
            "label": "Długość odcinka $L$",
            "unit": "km",
            "display": lambda x: x / 1000.0,
        },
    }


def _sweep_values(spec: dict, system: str) -> list:
    """Wartości przemiatu z uwzględnieniem sufitu per system (np. P_nom DC ≤ 6 MW)."""
    cap = spec.get("cap_si", {}).get(system)
    if cap is None:
        return spec["values"]
    return [v for v in spec["values"] if v <= cap + 1.0]


# ═══════════════════════════════════════════════════════════════════════════
#  POMOCNICZE — uruchomienie jednego scenariusza i ekstrakcja metryk
# ═══════════════════════════════════════════════════════════════════════════


def _run_and_extract(p: Parameters) -> dict:
    """
    Uruchamia pełną symulację + bilans energii dla danych parametrów
    i zwraca słownik kluczowych metryk wyjściowych.
    """
    profile = [(0.0, p.L, p.gradient)]
    sim = run_simulation(p, profile)
    energy = compute_energy(sim, p)

    return {
        "E_pant_netto_kWh": energy.E_pant_netto / 3.6e6,
        "E_pant_pobrana_kWh": energy.E_pant_pobrana / 3.6e6,
        "E_rec_pant_kWh": energy.E_rec_pant / 3.6e6,
        "E_per_km_kWh": energy.E_per_km,
        "E_per_btkm_Wh": energy.E_per_btkm,
        "E_per_seat_km_Wh": energy.E_per_seat_km,
        "E_jednostkowa": energy.E_jednostkowa,
        "T_min": sim.T_total / 60.0,
        "v_avg_kmh": sim.v_avg * 3.6,
        "reached_v_set": sim.reached_v_set,
        "F_max_kN": p.F_max / 1000.0,
        "v_breakpoint_kmh": p.v_breakpoint * 3.6,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  WORKER DLA MULTIPROCESSING — musi być na poziomie modułu (pickle)
# ═══════════════════════════════════════════════════════════════════════════


def _sweep_worker(task: dict) -> dict:
    """
    Worker uruchamiany w osobnym procesie. Dostaje opis zadania,
    rekonstruuje parametry, uruchamia symulację, zwraca wiersz wyników.

    Args:
        task: słownik z kluczami: system, parameter, value_SI,
              value_display, unit, base_kwargs.

    Returns:
        Słownik gotowy do wiersza DataFrame.
    """
    # Rekonstrukcja parametrów bazowych + nadpisanie
    base = Parameters.base()
    base_sys = base.with_changes(power_system=task["system"])
    p = base_sys.with_changes(**{task["parameter"]: task["value_SI"]})

    metrics = _run_and_extract(p)
    return {
        "system": task["system"],
        "parameter": task["parameter"],
        "value_SI": task["value_SI"],
        "value_display": task["value_display"],
        "unit": task["unit"],
        **metrics,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  CZĘŚĆ 1: SWEEP OAT
# ═══════════════════════════════════════════════════════════════════════════


def run_oat_sweep(
    base: Parameters | None = None,
    systems: tuple[str, ...] = ("AC", "DC"),
) -> pd.DataFrame:
    """
    Wykonuje sweep OAT: każdy parametr zmieniany po całym zakresie,
    dla każdego systemu zasilania osobno.

    Returns:
        DataFrame z kolumnami:
          system, parameter, value_SI, value_display, unit, + wszystkie metryki.
    """
    if base is None:
        base = Parameters.base()

    ranges = _build_sweep_ranges()
    rows = []

    for system in systems:
        # Dla DC moc ograniczona do 6 MW (chyba że badamy sweep mocy)
        base_sys = base.with_changes(power_system=system)

        for param_name, spec in ranges.items():
            for value in _sweep_values(spec, system):
                # Tworzymy kopię z nadpisanym parametrem
                p = base_sys.with_changes(**{param_name: value})

                metrics = _run_and_extract(p)
                row = {
                    "system": system,
                    "parameter": param_name,
                    "value_SI": value,
                    "value_display": spec["display"](value),
                    "unit": spec["unit"],
                    **metrics,
                }
                rows.append(row)

        print(
            f"  [{system}] przepuszczono {sum(len(s['values']) for s in ranges.values())} scenariuszy"
        )

    df = pd.DataFrame(rows)
    return df


def run_oat_sweep_parallel(
    base: Parameters | None = None,
    systems: tuple[str, ...] = ("AC", "DC"),
    n_workers: int | None = None,
) -> pd.DataFrame:
    """
    Równoległa wersja sweepu OAT (multiprocessing).

    Rozrzuca wszystkie przejazdy na procesy potomne. Każdy przejazd
    jest niezależny (embarrassingly parallel), więc skaluje się prawie
    liniowo z liczbą rdzeni.

    Args:
        base: Parametry bazowe.
        systems: Systemy zasilania do przebadania.
        n_workers: Liczba procesów. Default: liczba rdzeni - 1.

    Returns:
        DataFrame identyczny jak run_oat_sweep (kolejność wierszy może się różnić).
    """
    if base is None:
        base = Parameters.base()
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 1)

    ranges = _build_sweep_ranges()

    # Budujemy listę zadań (lekkie słowniki - łatwe do serializacji)
    tasks = []
    for system in systems:
        for param_name, spec in ranges.items():
            for value in _sweep_values(spec, system):
                tasks.append(
                    {
                        "system": system,
                        "parameter": param_name,
                        "value_SI": value,
                        "value_display": spec["display"](value),
                        "unit": spec["unit"],
                    }
                )

    rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_sweep_worker, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if i % 20 == 0 or i == len(tasks):
                print(f"    postęp: {i}/{len(tasks)} przejazdów")

    df = pd.DataFrame(rows)
    # Sortujemy dla powtarzalności (multiprocessing zwraca w losowej kolejności)
    df = df.sort_values(["system", "parameter", "value_SI"]).reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  CZĘŚĆ 2: ELASTYCZNOŚCI OAT (wskaźnik S_i^OAT, wzór 4.6)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ElasticityResult:
    """Wynik elastyczności dla jednego parametru."""

    parameter: str
    system: str
    S_plus: float  # elastyczność dla perturbacji +10%
    S_minus: float  # elastyczność dla perturbacji -10%
    S_avg: float  # średnia (do rankingu)
    asymmetry: float  # |S_plus - S_minus| (miara nieliniowości)
    E_base: float  # E_per_km bazowe [kWh/km]
    E_plus: float  # E_per_km przy +10%
    E_minus: float  # E_per_km przy -10%


def _elasticity_for_param(
    base: Parameters,
    param_name: str,
    delta: float = 0.10,
) -> ElasticityResult:
    """
    Liczy wskaźnik elastyczności S_i^OAT dla jednego parametru (wzór 4.6):

        S_i^OAT = (ΔE_per_km / E_per_km,0) / (Δp_i / p_i,0)

    Funkcją celu jest jednostkowe zużycie energii E_per_km [kWh/km]
    (energia netto na pantografie / długość odcinka), a nie energia
    całkowita. Dla parametrów innych niż L długość L jest stała podczas
    perturbacji, więc ich elastyczność jest identyczna jak dla energii
    całkowitej; zmienia się jedynie elastyczność względem L, która traci
    składnik trywialnego, niemal liniowego wzrostu energii z długością.

    Perturbacja ±delta wokół wartości bazowej.

    UWAGA dla pochylenia: bazowa wartość = 0‰, więc względna perturbacja
    nie ma sensu (dzielenie przez zero). Dla pochylenia używamy perturbacji
    bezwzględnej (0 → ±0.5‰) i raportujemy jako pseudo-elastyczność.
    """
    # Wartość bazowa parametru (SI)
    p0 = getattr(base, param_name)

    # Bazowy wynik
    E_base = _run_and_extract(base)["E_per_km_kWh"]

    # Specjalna obsługa pochylenia (baza = 0)
    if param_name == "gradient" and abs(p0) < 1e-9:
        # Perturbacja bezwzględna: ±0.5‰
        abs_delta = 0.5
        p_plus = abs_delta
        p_minus = -abs_delta
        E_plus = _run_and_extract(base.with_changes(gradient=p_plus))["E_per_km_kWh"]
        E_minus = _run_and_extract(base.with_changes(gradient=p_minus))["E_per_km_kWh"]
        # Pseudo-elastyczność: (ΔE/E) / (Δi w ‰) - normalizowane na 1‰
        S_plus = ((E_plus - E_base) / E_base) / abs_delta
        S_minus = ((E_minus - E_base) / E_base) / (-abs_delta)
    else:
        # Standardowa perturbacja względna ±delta
        p_plus = p0 * (1.0 + delta)
        p_minus = p0 * (1.0 - delta)
        E_plus = _run_and_extract(base.with_changes(**{param_name: p_plus}))[
            "E_per_km_kWh"
        ]
        E_minus = _run_and_extract(base.with_changes(**{param_name: p_minus}))[
            "E_per_km_kWh"
        ]
        S_plus = ((E_plus - E_base) / E_base) / (delta)
        S_minus = ((E_minus - E_base) / E_base) / (-delta)

    S_avg = 0.5 * (S_plus + S_minus)
    asymmetry = abs(S_plus - S_minus)

    return ElasticityResult(
        parameter=param_name,
        system=base.power_system,
        S_plus=S_plus,
        S_minus=S_minus,
        S_avg=S_avg,
        asymmetry=asymmetry,
        E_base=E_base,
        E_plus=E_plus,
        E_minus=E_minus,
    )


def run_oat_elasticity(
    base: Parameters | None = None,
    systems: tuple[str, ...] = ("AC", "DC"),
    delta: float = 0.10,
) -> pd.DataFrame:
    """
    Liczy elastyczności OAT dla wszystkich parametrów i systemów.

    Returns:
        DataFrame z kolumnami: parameter, system, S_plus, S_minus, S_avg,
        asymmetry, E_base_per_km, E_plus_per_km, E_minus_per_km.
        Funkcją celu jest E_per_km [kWh/km].
    """
    if base is None:
        base = Parameters.base()

    params = ["v_max", "m", "P_nom", "gradient", "L"]
    rows = []

    for system in systems:
        base_sys = base.with_changes(power_system=system)
        for param_name in params:
            res = _elasticity_for_param(base_sys, param_name, delta)
            rows.append(
                {
                    "parameter": param_name,
                    "system": system,
                    "S_plus": res.S_plus,
                    "S_minus": res.S_minus,
                    "S_avg": res.S_avg,
                    "asymmetry": res.asymmetry,
                    "E_base_per_km": res.E_base,
                    "E_plus_per_km": res.E_plus,
                    "E_minus_per_km": res.E_minus,
                }
            )

    df = pd.DataFrame(rows)
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  EKSPORT WYNIKÓW
# ═══════════════════════════════════════════════════════════════════════════


def export_sensitivity(
    df_sweep: pd.DataFrame,
    df_elasticity: pd.DataFrame,
    save_dir=OUTPUT_DIR,
):
    """Zapisuje wyniki analizy wrażliwości do CSV."""
    save_dir.mkdir(parents=True, exist_ok=True)
    path_sweep = save_dir / "sensitivity_sweep.csv"
    path_elast = save_dir / "sensitivity_elasticity.csv"
    df_sweep.to_csv(path_sweep, index=False, float_format="%.6g")
    df_elasticity.to_csv(path_elast, index=False, float_format="%.6g")
    return {"sweep": path_sweep, "elasticity": path_elast}


# ═══════════════════════════════════════════════════════════════════════════
#  🚀 SZYBKI TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    base = Parameters.base()
    print("=" * 70)
    print("ANALIZA WRAŻLIWOŚCI OAT")
    print("=" * 70)
    print()

    t0 = time.perf_counter()
    n_cpu = os.cpu_count() or 4
    print(f">>> CZĘŚĆ 1: Sweep OAT równolegle ({n_cpu - 1} procesów)...")
    df_sweep = run_oat_sweep_parallel(base)
    print(f"    Zebrano {len(df_sweep)} scenariuszy w {time.perf_counter() - t0:.1f} s")
    print()

    t1 = time.perf_counter()
    print(">>> CZĘŚĆ 2: Elastyczności OAT (ranking ważności)...")
    df_elast = run_oat_elasticity(base)
    print(f"    Policzono w {time.perf_counter() - t1:.1f} s")
    print()

    # Ranking elastyczności dla AC
    print("=" * 70)
    print("RANKING WAŻNOŚCI PARAMETRÓW (elastyczność |S_avg|, system AC)")
    print("=" * 70)
    df_ac = df_elast[df_elast["system"] == "AC"].copy()
    df_ac["abs_S"] = df_ac["S_avg"].abs()
    df_ac = df_ac.sort_values("abs_S", ascending=False)
    print(
        f"{'Parametr':>12} {'S_avg':>10} {'S_plus':>10} {'S_minus':>10} {'Asymetria':>12}"
    )
    print("-" * 60)
    for _, r in df_ac.iterrows():
        print(
            f"{r['parameter']:>12} {r['S_avg']:>10.3f} {r['S_plus']:>10.3f} "
            f"{r['S_minus']:>10.3f} {r['asymmetry']:>12.3f}"
        )
    print()

    # Eksport
    print(">>> Eksport CSV...")
    paths = export_sensitivity(df_sweep, df_elast)
    for key, path in paths.items():
        print(f"    {key}: {path.name}")
    print()
    print(f"✓ Łączny czas: {time.perf_counter() - t0:.1f} s")
