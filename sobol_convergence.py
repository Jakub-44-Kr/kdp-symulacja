"""
sobol_convergence.py — Test zbieżności indeksów Sobola.

Realizuje procedurę z rozdz. 4.6: oblicza indeksy S_i i S_Ti dla rosnącej
liczby próbek N i obserwuje, przy jakim N wartości się stabilizują.
Stabilizacja uzasadnia wybór finalnego N do analizy w rozdz. 7.

Generuje:
  - CSV z indeksami dla każdego N (sobol_convergence_{system}.csv)
  - wykres S_i(N) i S_Ti(N) dla każdego parametru (sobol_convergence.png)

Autor: Jakub Król, PW WE, 2026
"""

from __future__ import annotations

import os
import time

import matplotlib.pyplot as plt
import pandas as pd

from parameters import OUTPUT_DIR
from sobol import run_sobol_for_system

# Styl wykresów (spójny z plotting.py)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# Kolory per parametr (spójne między wykresami)
PARAM_COLORS = {
    "v_max": "#1f77b4",
    "m": "#ff7f0e",
    "P_nom": "#2ca02c",
    "gradient": "#d62728",
    "L": "#9467bd",
}
PARAM_LABELS = {
    "v_max": "$v_{max}$",
    "m": "$m$",
    "P_nom": "$P$",
    "gradient": "$i$",
    "L": "$L$",
}


def run_convergence(
    systems: tuple[str, ...] = ("AC", "DC"),
    N_values: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096),
) -> pd.DataFrame:
    """
    Liczy indeksy Sobola dla rosnących N i obu systemów.

    Returns:
        DataFrame: system, N, n_runs, parameter, S1, S1_conf, ST, ST_conf.
    """
    rows = []
    for system in systems:
        print(f"\n{'=' * 70}")
        print(f"TEST ZBIEŻNOŚCI — system {system}")
        print(f"{'=' * 70}")
        for N in N_values:
            t0 = time.perf_counter()
            res = run_sobol_for_system(system, N=N)
            dt = time.perf_counter() - t0
            print(f"  N={N:>5} ({res['n_runs']:>6} uruchomień): {dt:.1f} s")
            for i, name in enumerate(res["names"]):
                rows.append(
                    {
                        "system": system,
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


def plot_convergence(df: pd.DataFrame, save_dir=OUTPUT_DIR) -> list:
    """
    Rysuje wykresy zbieżności S_i(N) i S_Ti(N) dla każdego systemu.
    Dwa panele (S1, ST) na system.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for system in df["system"].unique():
        df_sys = df[df["system"] == system]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        for param in df_sys["parameter"].unique():
            d = df_sys[df_sys["parameter"] == param].sort_values("N")
            color = PARAM_COLORS.get(param, None)
            label = PARAM_LABELS.get(param, param)

            # S1 z przedziałem ufności
            ax1.errorbar(
                d["N"],
                d["S1"],
                yerr=d["S1_conf"],
                marker="o",
                capsize=3,
                color=color,
                label=label,
                linewidth=1.5,
            )
            # ST
            ax2.errorbar(
                d["N"],
                d["ST"],
                yerr=d["ST_conf"],
                marker="s",
                capsize=3,
                color=color,
                label=label,
                linewidth=1.5,
            )

        for ax, title in [
            (ax1, "$S_i$ (pierwszego rzędu)"),
            (ax2, "$S_{Ti}$ (całkowity)"),
        ]:
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Liczba próbek bazowych $N$")
            ax.set_ylabel("Indeks Sobola")
            ax.set_title(title)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.legend(loc="best", framealpha=0.9)

        fig.suptitle(
            f"Zbieżność indeksów Sobola — system {system}", fontsize=13, y=1.02
        )

        path = save_dir / f"sobol_convergence_{system}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
        print(f"    wykres: {path.name}")

    return paths


if __name__ == "__main__":
    n_cpu = os.cpu_count() or 4
    print(f"Test zbieżności Sobola — {n_cpu - 1} procesów")
    print("Zakres N: 128, 256, 512, 1024, 2048, 4096")
    print("Szacowany czas: ~20-25 min\n")

    t0 = time.perf_counter()
    df = run_convergence()

    # Zapis CSV per system
    for system in df["system"].unique():
        path = OUTPUT_DIR / f"sobol_convergence_{system}.csv"
        df[df["system"] == system].to_csv(path, index=False, float_format="%.6g")
        print(f"\n    CSV: {path.name}")

    # Wykresy
    print("\n>>> Generuję wykresy zbieżności...")
    plot_convergence(df)

    # Tabela końcowa - wartości przy max N
    N_max = df["N"].max()
    print(f"\n{'=' * 70}")
    print(f"WARTOŚCI PRZY N={N_max} (najdokładniejsze)")
    print(f"{'=' * 70}")
    for system in df["system"].unique():
        d = df[(df["system"] == system) & (df["N"] == N_max)].sort_values(
            "ST", ascending=False
        )
        print(f"\n  System {system}:")
        print(f"  {'Parametr':>10} {'S1':>10} {'ST':>10}")
        for _, r in d.iterrows():
            print(f"  {r['parameter']:>10} {r['S1']:>10.4f} {r['ST']:>10.4f}")

    print(f"\n✓ Łączny czas: {(time.perf_counter() - t0) / 60:.1f} min")
