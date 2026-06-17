"""
results.py — Eksport wyników symulacji do plików.

Zapisuje wyniki w dwóch formatach:
  - CSV - tablice profilu dynamicznego (do dalszej analizy w Excel, pandas)
  - JSON - skalary, parametry, metryki (do agregacji wyników wielu scenariuszy)

Konwencja nazewnictwa plików:
  scenario_{tag}_profile.csv   - tablica x, t, v, F, P, I
  scenario_{tag}_summary.json  - metadane + skalary
  scenario_{tag}_params.json   - kopia parametrów wejściowych

gdzie {tag} = krótki tag scenariusza (np. "base", "v400_m750_DC").

Autor: Jakub Król, PW WE, 2026
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from energy import J_TO_KWH, EnergyResults
from parameters import OUTPUT_DIR, Parameters
from simulation import SimulationProfile

# ═══════════════════════════════════════════════════════════════════════════
#  CSV - profil dynamiczny
# ═══════════════════════════════════════════════════════════════════════════


def export_profile_csv(
    sim: SimulationProfile,
    energy: EnergyResults,
    tag: str = "base",
    save_dir: Path = OUTPUT_DIR,
) -> Path:
    """
    Eksportuje pełny profil dynamiczny i energetyczny do CSV.

    Każdy wiersz = jeden punkt symulacji (co dx = 1 m).
    Kolumny: x, t, v, a, faza, siły, moce, prąd.
    """
    df = pd.DataFrame(
        {
            "x_m": sim.x,
            "t_s": sim.t,
            "v_ms": sim.v,
            "v_kmh": sim.v * 3.6,
            "a_ms2": sim.a,
            "phase": sim.phase,
            "F_tr_N": sim.F_tr,
            "F_brake_N": sim.F_brake,
            "F_brake_el_N": energy.F_brake_el,
            "F_brake_mech_N": energy.F_brake_mech,
            "F_op_N": sim.F_op,
            "F_grav_N": sim.F_grav,
            "gradient_promille": sim.gradient,
            "P_kolo_W": energy.P_kolo,
            "P_pant_draw_W": energy.P_pant_draw,
            "P_pant_rec_W": energy.P_pant_rec,
            "P_pant_net_W": energy.P_pant_net,
            "I_pant_A": energy.I_pant,
        }
    )

    path = save_dir / f"scenario_{tag}_profile.csv"
    df.to_csv(path, index=False, float_format="%.6g")
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  JSON - podsumowanie skalarne
# ═══════════════════════════════════════════════════════════════════════════


def export_summary_json(
    sim: SimulationProfile,
    energy: EnergyResults,
    p: Parameters,
    tag: str = "base",
    save_dir: Path = OUTPUT_DIR,
) -> Path:
    """
    Eksportuje skalary i metryki + kopię parametrów do JSON.
    Format gotowy do agregacji wyników wielu scenariuszy w pandas.
    """
    summary = {
        "tag": tag,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "parameters": {
            "v_max_kmh": p.v_max * 3.6,
            "m_t": p.m / 1000.0,
            "P_nom_MW": p.P_nom / 1e6,
            "P_eff_max_MW": p.P_eff_max / 1e6,
            "gradient_promille": p.gradient,
            "L_km": p.L / 1000.0,
            "power_system": p.power_system,
            "dx_coast_km": p.dx_coast / 1000.0,
            "F_max_kN": p.F_max / 1000.0,
            "v_breakpoint_kmh": p.v_breakpoint * 3.6,
            "eta_tr": p.eta_tr,
            "P_aux_kW": p.P_aux / 1000.0,
            "a_launch_max": p.a_launch_max,
            "a_brake_max": p.a_brake_max,
            "davis_A": p.davis_A,
            "davis_B": p.davis_B,
            "davis_C": p.davis_C,
        },
        "kinematics": {
            "T_total_s": sim.T_total,
            "T_total_min": sim.T_total / 60.0,
            "v_avg_kmh": sim.v_avg * 3.6,
            "v_max_reached_kmh": sim.v_max_reached * 3.6,
            "reached_v_set": sim.reached_v_set,
            "x1_km": sim.x1 / 1000.0,
            "x2_km": sim.x2 / 1000.0,
            "x3_km": sim.x3 / 1000.0,
        },
        "energy_kWh": {
            "E_trakcja_kolo": energy.E_trakcja_kolo * J_TO_KWH,
            "E_trakcja_pant": energy.E_trakcja_pant * J_TO_KWH,
            "E_aux": energy.E_aux * J_TO_KWH,
            "E_ham_el_kolo": energy.E_ham_el_kolo * J_TO_KWH,
            "E_rec_pant": energy.E_rec_pant * J_TO_KWH,
            "E_ham_mech": energy.E_ham_mech * J_TO_KWH,
            "E_pant_pobrana": energy.E_pant_pobrana * J_TO_KWH,
            "E_pant_netto": energy.E_pant_netto * J_TO_KWH,
        },
        "metrics": {
            "E_per_km_kWh": energy.E_per_km,
            "E_per_btkm_Wh": energy.E_per_btkm,
            "E_per_seat_km_Wh": energy.E_per_seat_km,
            "E_jednostkowa_kWh_per_100km_t": energy.E_jednostkowa,
            "P_pant_max_MW": energy.P_pant_max / 1e6,
            "P_pant_avg_MW": energy.P_pant_avg / 1e6,
            "I_pant_max_A": energy.I_pant_max,
            "I_grid_limit_A": p.I_grid_limit,
        },
    }

    path = save_dir / f"scenario_{tag}_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  GŁÓWNA FUNKCJA EKSPORTU
# ═══════════════════════════════════════════════════════════════════════════


def export_all(
    sim: SimulationProfile,
    energy: EnergyResults,
    p: Parameters,
    tag: str = "base",
    save_dir: Path = OUTPUT_DIR,
) -> dict[str, Path]:
    """
    Eksportuje wszystkie wyniki scenariusza.

    Returns:
        Słownik z kluczami: 'csv', 'json' i ścieżkami do plików.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    return {
        "csv": export_profile_csv(sim, energy, tag, save_dir),
        "json": export_summary_json(sim, energy, p, tag, save_dir),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  🚀 SZYBKI TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from energy import compute_energy
    from simulation import run_simulation

    p = Parameters.base()
    print(">>> Symulacja...")
    sim = run_simulation(p)
    print(">>> Bilans energii...")
    energy = compute_energy(sim, p)
    print(">>> Eksport...")
    paths = export_all(sim, energy, p, tag="base")
    print()
    print("Zapisano:")
    for key, path in paths.items():
        size_kb = path.stat().st_size / 1024.0
        print(f"  - {key.upper():4s}: {path.name}  ({size_kb:.1f} KB)")
