"""
plots_param_influence.py — Wpływ pojedynczych parametrów na zużycie energii.

Generuje serię wykresów do podrozdziałów rozdziału 7. Dla każdego parametru
powstają dwie strony:

  1. CHARAKTERYSTYKA E/km — zależność jednostkowego zużycia energii
     E_pant,netto/L [kWh/km] od wartości parametru. Dwa panele (AC | DC),
     jedna krzywa na długość odcinka (kolor = L). Każdy punkt to końcowe
     E_per_km całego przejazdu — ta sama metryka, co w analizie OAT i Sobola.
       - Wpływ prędkości eksploatacyjnej v_max
       - Wpływ masy składu m
       - Wpływ mocy znamionowej P  (DC: sufit 6 MW → pojedynczy punkt)
       - Wpływ pochylenia trasy i

  2. PROFIL PRĘDKOŚCI v(x) — 4×2 panele długości, profil prędkości wzdłuż
     trasy (kolor = wartość parametru, AC ciągła / DC przerywana).

Dodatkowo wykres wpływu długości odcinka: E/km vs L dla różnych prędkości.

Uruchomienia modelu realizowane równolegle (multiprocessing).

Autor: Jakub Król, PW WE, 2026
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from energy import J_TO_KWH, compute_energy
from parameters import OUTPUT_DIR, Parameters
from simulation import run_simulation

# ═══════════════════════════════════════════════════════════════════════════
#  STYL
# ═══════════════════════════════════════════════════════════════════════════

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# Paleta kolorów (viridis-like, czytelna na druku też w mono)
CMAP_NAME = "plasma"  # plasma daje wyrazne odcienie żółty→pomarańcz→czerwony→fiolet

# Styl linii per system
LINESTYLE_AC = "-"
LINESTYLE_DC = "--"


# ═══════════════════════════════════════════════════════════════════════════
#  DEFINICJE PRZEMIATANYCH PARAMETRÓW
# ═══════════════════════════════════════════════════════════════════════════

# Każdy parametr: (nazwa, wartości w SI, label do legendy, formatter wartości)
PARAM_SWEEPS = {
    "v_max": {
        "values": [v / 3.6 for v in (250, 280, 310, 340, 370)],
        "label": "Wpływ prędkości eksploatacyjnej $v_{max}$",
        "fmt": lambda v: f"{v * 3.6:.0f} km/h",
        "param_name": "$v_{max}$",
        "to_x": lambda v: v * 3.6,  # SI (m/s) -> km/h na osi X
        "xlabel": "Prędkość eksploatacyjna $v_{max}$ [km/h]",
    },
    "m": {
        "values": [m * 1000 for m in (450, 500, 550, 600, 650, 700, 750)],
        "label": "Wpływ masy składu $m$",
        "fmt": lambda v: f"{v / 1000:.0f} t",
        "param_name": "$m$",
        "to_x": lambda v: v / 1000.0,  # kg -> t
        "xlabel": "Masa składu $m$ [t]",
    },
    "P_nom": {
        "values": [P * 1e6 for P in (6, 7, 8, 9, 10, 11, 12)],
        "label": "Wpływ mocy znamionowej $P$",
        "fmt": lambda v: f"{v / 1e6:.0f} MW",
        "param_name": "$P$",
        # Dla DC sufit trakcji = 6 MW — zadania DC dla mocy > 6 MW pomijane.
        "cap_si": {"DC": 9e6},
        "to_x": lambda v: v / 1e6,  # W -> MW
        "xlabel": "Moc znamionowa $P$ [MW]",
    },
    "gradient": {
        "values": [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0],
        "label": "Wpływ pochylenia trasy $i$",
        "fmt": lambda v: f"{v:+.0f} ‰",
        "param_name": "$i$",
        "to_x": lambda v: v,  # ‰ bez zmian
        "xlabel": "Pochylenie trasy $i$ [‰]",
    },
}

# Długości tras dla paneli 4×2 (8 paneli)
TRACK_LENGTHS_KM = [50, 100, 150, 200, 250, 300, 350, 400]

# Systemy zasilania
SYSTEMS = ("AC", "DC")


# ═══════════════════════════════════════════════════════════════════════════
#  WORKER MULTIPROCESSING
# ═══════════════════════════════════════════════════════════════════════════


def _worker(task: dict) -> dict:
    """
    Worker: uruchamia jedną symulację dla zadanego scenariusza.

    task: param_name, value_SI, L_m, system

    Zwraca: x [m], E_cum [J], T_total [s] + metadata.
    """
    base = Parameters.base()
    p = base.with_changes(
        power_system=task["system"],
        L=task["L_m"],
        **{task["param_name"]: task["value_SI"]},
    )
    profile = [(0.0, p.L, p.gradient)]
    sim = run_simulation(p, profile)
    energy = compute_energy(sim, p)

    # Energia kumulowana wzdłuż trasy = całka P_pant_net(t) dt
    # Wektorowo: trapezoidalna całka skumulowana
    dt = np.diff(sim.t)
    P_net = energy.P_pant_net
    # cumulative integration (trapezoid): E[i] = sum_{j<i} 0.5*(P[j]+P[j+1])*dt[j]
    E_cum_increments = 0.5 * (P_net[:-1] + P_net[1:]) * dt  # [J]
    E_cum = np.zeros_like(P_net)
    E_cum[1:] = np.cumsum(E_cum_increments)

    return {
        "param_name": task["param_name"],
        "value_SI": task["value_SI"],
        "L_m": task["L_m"],
        "system": task["system"],
        "x": sim.x,
        "v_ms": sim.v,  # NOWE: profil prędkości
        "E_cum_J": E_cum,
        "T_total_s": sim.T_total,
        "v_max_param": task["value_SI"] if task["param_name"] == "v_max" else None,
        "E_pant_netto_kWh": energy.E_pant_netto * J_TO_KWH,
        "E_per_km_kWh": energy.E_per_km,  # kWh/km (metryka jednostkowa)
        "E_per_btkm_Wh": energy.E_per_btkm,  # Wh/(brutto-tona·km)
    }


def _run_all_tasks(tasks: list[dict], n_workers: int | None = None) -> list[dict]:
    """Uruchamia wszystkie zadania równolegle, zwraca wyniki w kolejności tasków."""
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"    {len(tasks)} przejazdów na {n_workers} procesach...")
    # executor.map zachowuje kolejność zadań → wyniki są zgodne z `tasks`
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_worker, tasks, chunksize=4))
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  WYKRES 4×2: pojedynczy parametr × 8 długości
# ═══════════════════════════════════════════════════════════════════════════


def _format_legend_entry(
    value_SI: float, T_AC: float, T_DC: float, param_name: str
) -> str:
    """Etykieta jednowierszowa: 'v=320 km/h (AC 32 min, DC 35 min)'.

    T_AC / T_DC mogą być None (brak danych dla systemu — np. moc DC > 6 MW);
    wtedy w etykiecie pojawia się '—' zamiast czasu.
    """
    spec = PARAM_SWEEPS[param_name]
    val_str = spec["fmt"](value_SI)
    ac = f"AC {T_AC / 60:.1f}" if T_AC else "AC —"
    dc = f"DC {T_DC / 60:.1f}" if T_DC else "DC —"
    return f"{val_str}  ({ac}, {dc} min)"


def plot_param_influence(
    param_name: str,
    results: list[dict],
    save_dir: Path = OUTPUT_DIR,
) -> Path:
    """
    Charakterystyka wpływu parametru na jednostkowe zużycie energii E/km.

    Jedna strona = dwa panele (AC | DC). Na każdym panelu:
      - oś X: wartość badanego parametru (jednostki wyświetlane),
      - oś Y: jednostkowe zużycie energii netto E_pant,netto/L [kWh/km],
      - jedna krzywa na długość odcinka z TRACK_LENGTHS_KM (kolor = L).

    To zależność f(parametr), a nie profil wzdłuż trasy: każdy punkt krzywej
    to KOŃCOWE E_per_km całego przejazdu — ta sama metryka, co w analizie
    OAT i Sobola. Spadek krzywych wraz z rosnącym L pokazuje amortyzację
    energii rozpędzania/hamowania na dłuższym dystansie.
    """
    spec = PARAM_SWEEPS[param_name]
    cmap = plt.get_cmap(CMAP_NAME)
    n_L = len(TRACK_LENGTHS_KM)
    L_colors = {L: cmap(i / max(1, n_L - 1)) for i, L in enumerate(TRACK_LENGTHS_KM)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    sys_axes = {"AC": axes[0], "DC": axes[1]}
    sys_titles = {"AC": "2×25 kV AC", "DC": "3 kV DC"}

    for system in SYSTEMS:
        ax = sys_axes[system]
        for L_km in TRACK_LENGTHS_KM:
            L_m = L_km * 1000.0
            pts = [
                (r["value_SI"], r["E_per_km_kWh"])
                for r in results
                if r["system"] == system and r["L_m"] == L_m
            ]
            if not pts:
                continue
            pts.sort(key=lambda t: t[0])
            xs = [spec["to_x"](v) for v, _ in pts]
            ys = [e for _, e in pts]
            multi = len(pts) > 1
            ax.plot(
                xs,
                ys,
                color=L_colors[L_km],
                marker="o",
                markersize=4,
                linewidth=1.6 if multi else 0,
                linestyle="-" if multi else "None",
            )
        ax.set_title(sys_titles[system])
        ax.set_xlabel(spec["xlabel"])

    sys_axes["AC"].set_ylabel("Jednostkowe zużycie energii $E_{pant,netto}/L$ [kWh/km]")

    # Adnotacja sufitu mocy DC przy przemiataniu mocy (DC = pojedynczy punkt)
    if param_name == "P_nom":
        sys_axes["DC"].text(
            0.5,
            0.04,
            "sufit trakcji 9 MW",
            transform=sys_axes["DC"].transAxes,
            ha="center",
            fontsize=8,
            style="italic",
            color="0.4",
        )

    # Wspólna legenda długości odcinka (kolor = L)
    from matplotlib.lines import Line2D

    L_handles = [
        Line2D(
            [0],
            [0],
            color=L_colors[L],
            marker="o",
            markersize=4,
            linewidth=1.6,
            label=f"{L} km",
        )
        for L in TRACK_LENGTHS_KM
    ]
    fig.legend(
        handles=L_handles,
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
        title="Długość odcinka $L$",
        frameon=True,
        framealpha=0.95,
        fontsize=9,
    )

    fig.suptitle(spec["label"], fontsize=14, y=1.02, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 0.87, 0.96))

    safe_name = param_name.replace("_", "")
    path = save_dir / f"influence_{safe_name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  WYKRES 4×2: profile prędkości v(x) — analogiczny do plot_param_influence
# ═══════════════════════════════════════════════════════════════════════════


def plot_param_influence_velocity(
    param_name: str,
    results: list[dict],
    save_dir: Path = OUTPUT_DIR,
) -> Path:
    """
    Generuje stronę 4×2 paneli z profilami prędkości v(x) dla wpływu jednego parametru.

    Identyczna struktura jak plot_param_influence, ale na osi Y prędkość v [km/h]
    zamiast energii kumulowanej.
    """
    spec = PARAM_SWEEPS[param_name]
    values = spec["values"]
    n_values = len(values)
    cmap = plt.get_cmap(CMAP_NAME)
    colors = [cmap(i / max(1, n_values - 1)) for i in range(n_values)]

    fig, axes = plt.subplots(4, 2, figsize=(11, 13), sharey=False)
    axes_flat = axes.flatten()

    # Grupowanie wyników po L_m
    by_L = {}
    for r in results:
        L_m = r["L_m"]
        by_L.setdefault(L_m, []).append(r)

    for panel_idx, L_km in enumerate(TRACK_LENGTHS_KM):
        ax = axes_flat[panel_idx]
        L_m = L_km * 1000.0
        panel_results = by_L.get(L_m, [])
        by_key = {(r["value_SI"], r["system"]): r for r in panel_results}

        # Najpierw DC (przerywane, na spodzie), potem AC (ciągłe, na wierzchu)
        # + lekka przezroczystość żeby było widać pokrywające się krzywe DC
        for v_idx, value_SI in enumerate(values):
            color = colors[v_idx]
            r_dc = by_key.get((value_SI, "DC"))
            if r_dc is not None:
                ax.plot(
                    r_dc["x"] / 1000.0,
                    r_dc["v_ms"] * 3.6,
                    color=color,
                    linestyle=LINESTYLE_DC,
                    linewidth=1.8,
                    alpha=0.7,
                )
        for v_idx, value_SI in enumerate(values):
            color = colors[v_idx]
            r_ac = by_key.get((value_SI, "AC"))
            if r_ac is not None:
                ax.plot(
                    r_ac["x"] / 1000.0,
                    r_ac["v_ms"] * 3.6,
                    color=color,
                    linestyle=LINESTYLE_AC,
                    linewidth=1.4,
                )

        ax.set_xlim(0, L_km)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Pozycja $x$ [km]")
        ax.set_ylabel("Prędkość $v$ [km/h]")
        ax.set_title(f"L = {L_km} km")

    # Legenda - identyczna jak w plot_param_influence
    legend_handles = []
    legend_labels = []
    L_ref = TRACK_LENGTHS_KM[3] * 1000.0
    panel_ref = by_L.get(L_ref, [])
    by_key_ref = {(r["value_SI"], r["system"]): r for r in panel_ref}

    from matplotlib.lines import Line2D

    for v_idx, value_SI in enumerate(values):
        color = colors[v_idx]
        T_AC = by_key_ref.get((value_SI, "AC"), {}).get("T_total_s", 0)
        T_DC = by_key_ref.get((value_SI, "DC"), {}).get("T_total_s", 0)
        label = _format_legend_entry(value_SI, T_AC, T_DC, param_name)
        legend_handles.append(Line2D([0], [0], color=color, linewidth=2))
        legend_labels.append(label)

    sys_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=LINESTYLE_AC,
            linewidth=1.5,
            label="AC (2×25 kV)",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=LINESTYLE_DC,
            linewidth=1.5,
            label="DC (3 kV)",
        ),
    ]

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=3,
        title=spec["param_name"],
        frameon=True,
        framealpha=0.95,
        fontsize=8,
    )
    fig.legend(
        handles=sys_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        title="System zasilania",
        frameon=True,
        framealpha=0.95,
    )

    title_velocity = spec["label"].replace("Wpływ", "Profile prędkości — wpływ")
    fig.suptitle(title_velocity, fontsize=14, y=1.04, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.97))

    safe_name = param_name.replace("_", "")
    path = save_dir / f"influence_velocity_{safe_name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  WPŁYW DŁUGOŚCI ODCINKA — jeden panel
# ═══════════════════════════════════════════════════════════════════════════


def plot_length_influence(
    save_dir: Path = OUTPUT_DIR, n_workers: int | None = None
) -> Path:
    """
    Pojedynczy wykres: jednostkowe zużycie E_per_km [kWh/km] vs L,
    dla 5-6 prędkości × 2 systemy. Pokazuje spadek zużycia jednostkowego
    z długością odcinka (amortyzacja energii rozpędzania/hamowania) —
    w przeciwieństwie do energii całkowitej, która z L rośnie ~liniowo.
    """
    # Sweep parametrów
    velocities_kmh = (250, 280, 310, 340, 370, 400)
    lengths_km = list(range(50, 401, 25))  # 50, 75, ..., 400 km (15 punktów)

    tasks = []
    for v_kmh in velocities_kmh:
        for L_km in lengths_km:
            for system in SYSTEMS:
                tasks.append(
                    {
                        "param_name": "v_max",
                        "value_SI": v_kmh / 3.6,
                        "L_m": L_km * 1000.0,
                        "system": system,
                    }
                )

    t0 = time.perf_counter()
    print("  >>> Sweep L (E/km vs L)...")
    results = _run_all_tasks(tasks, n_workers)
    print(f"    {len(tasks)} przejazdów w {time.perf_counter() - t0:.1f} s")

    # Rysowanie
    cmap = plt.get_cmap(CMAP_NAME)
    colors = [
        cmap(i / max(1, len(velocities_kmh) - 1)) for i in range(len(velocities_kmh))
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for v_idx, v_kmh in enumerate(velocities_kmh):
        color = colors[v_idx]
        for system in SYSTEMS:
            linestyle = LINESTYLE_AC if system == "AC" else LINESTYLE_DC
            xs = []
            ys = []
            for L_km in lengths_km:
                # znajdź wynik dla (v_kmh, L_km, system)
                v_SI = v_kmh / 3.6
                L_m = L_km * 1000.0
                for r in results:
                    if (
                        abs(r["value_SI"] - v_SI) < 1e-6
                        and r["L_m"] == L_m
                        and r["system"] == system
                    ):
                        xs.append(L_km)
                        ys.append(r["E_per_km_kWh"])
                        break
            ax.plot(
                xs,
                ys,
                color=color,
                linestyle=linestyle,
                linewidth=1.6,
                marker="o",
                markersize=4,
                label=f"{v_kmh} km/h ({system})",
            )

    ax.set_xlabel("Długość odcinka $L$ [km]", fontsize=11)
    ax.set_ylabel(
        "Jednostkowe zużycie energii $E_{pant,netto}/L$ [kWh/km]", fontsize=11
    )
    ax.set_title(
        "Wpływ długości odcinka na jednostkowe zużycie energii\n"
        "dla różnych prędkości eksploatacyjnych i systemów zasilania",
        fontsize=12,
    )
    ax.legend(loc="upper right", ncol=2, framealpha=0.95, fontsize=9)
    ax.set_xlim(0, max(lengths_km) * 1.02)
    # oś Y autoskalowana — wartości ~22-32 kWh/km, wymuszanie 0 spłaszczyłoby zakres

    path = save_dir / "influence_length.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════


def run_all_param_influence_plots(
    save_dir: Path = OUTPUT_DIR, n_workers: int | None = None
) -> list[Path]:
    """
    Główna funkcja: dla każdego z 4 parametrów generuje stronę 4×2 paneli,
    plus dodatkowy wykres wpływu długości.

    Returns:
        Lista ścieżek do wygenerowanych plików.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1-4: cztery podrozdziały po 8 paneli każdy (energia + prędkość)
    for param_name, spec in PARAM_SWEEPS.items():
        print(f"\n>>> Podrozdział: {spec['label']}")
        tasks = []
        cap = spec.get("cap_si", {})
        for value_SI in spec["values"]:
            for L_km in TRACK_LENGTHS_KM:
                for system in SYSTEMS:
                    # pomiń wartości powyżej sufitu danego systemu (np. moc DC > 6 MW)
                    if system in cap and value_SI > cap[system] + 1.0:
                        continue
                    tasks.append(
                        {
                            "param_name": param_name,
                            "value_SI": value_SI,
                            "L_m": L_km * 1000.0,
                            "system": system,
                        }
                    )

        t0 = time.perf_counter()
        results = _run_all_tasks(tasks, n_workers)

        # Dwa wykresy z tego samego zestawu symulacji
        print("    rysowanie wykresu energii...")
        path_E = plot_param_influence(param_name, results, save_dir)
        paths.append(path_E)
        print(f"    {path_E.name}")

        print("    rysowanie wykresu prędkości...")
        path_v = plot_param_influence_velocity(param_name, results, save_dir)
        paths.append(path_v)
        print(f"    {path_v.name}  (łącznie {time.perf_counter() - t0:.1f} s)")

    # 5: wpływ długości (osobny, prostszy wykres)
    print("\n>>> Podrozdział: Wpływ długości odcinka")
    path = plot_length_influence(save_dir, n_workers)
    paths.append(path)
    print(f"    {path.name}")

    return paths


# ═══════════════════════════════════════════════════════════════════════════
#  🚀 GŁÓWNY PRZEBIEG
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    n_cpu = os.cpu_count() or 4
    print("=" * 78)
    print("WYKRESY WPŁYWU POJEDYNCZYCH PARAMETRÓW")
    print(f"  Procesory: {n_cpu}, używam {n_cpu - 1} procesów")
    print("  Sumaryczne przejazdy:")
    n_total = 0
    for p_name, spec in PARAM_SWEEPS.items():
        n = len(spec["values"]) * len(TRACK_LENGTHS_KM) * len(SYSTEMS)
        n_total += n
        print(
            f"    {p_name}: {len(spec['values'])} × {len(TRACK_LENGTHS_KM)} L × "
            f"{len(SYSTEMS)} sys = {n}"
        )
    n_length = 6 * 15 * 2  # dla wpływu długości
    n_total += n_length
    print(f"    wpływ L: {n_length}")
    print(
        f"  RAZEM: {n_total} przejazdów (~{n_total * 0.012:.0f} s przy obecnej prędkości)"
    )
    print("=" * 78)

    t0 = time.perf_counter()
    paths = run_all_param_influence_plots()

    print()
    print("=" * 78)
    print(f"WYGENEROWANO {len(paths)} WYKRESÓW")
    print("=" * 78)
    for p in paths:
        print(f"  - {p.name}")
    print()
    print(f"✓ Łączny czas: {(time.perf_counter() - t0) / 60:.1f} min")
