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
        "cap_si": {"DC": 250 / 3.6},  # DC: sztywny sufit 250 km/h (A)
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

# Długości tras dla paneli profili prędkości (jedna kolumna na całą stronę)
VELOCITY_PANEL_LENGTHS = [50, 150, 250]

# Panele profili prędkości: 3 prędkości zadane (zamiast 3 długości)
SPEED_PANELS_KMH = [250, 300, 350]
VELOCITY_PANEL_L_KM = 100  # stała długość odcinka dla profili [km]
DC_VELOCITY_CAP_KMH = 250  # DC <= 250 km/h (zalecenie promotora)

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


def _vel_worker(task: dict) -> dict:
    """Worker profili prędkości: ustawia v_max = prędkość panelu ORAZ wartość parametru."""
    changes = {
        "power_system": task["system"],
        "L": VELOCITY_PANEL_L_KM * 1000.0,
        "v_max": task["v_panel_kmh"] / 3.6,
    }
    if task["param_name"] != "v_max":
        changes[task["param_name"]] = task["value_SI"]
    p = Parameters.base().with_changes(**changes)
    sim = run_simulation(p, [(0.0, p.L, p.gradient)])
    return {
        "param_name": task["param_name"],
        "value_SI": task["value_SI"],
        "v_panel_kmh": task["v_panel_kmh"],
        "system": task["system"],
        "x": sim.x,
        "v_ms": sim.v,
        "T_total_s": float(sim.t[-1]),
    }


def _run_vel_tasks(tasks: list[dict], n_workers: int | None = None) -> list[dict]:
    """Jak _run_all_tasks, ale dla profili prędkości (_vel_worker)."""
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 1)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        return list(executor.map(_vel_worker, tasks, chunksize=4))


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
    metric: str = "E_per_km_kWh",
    ylabel: str = "Jednostkowe zużycie energii $E_{pant,netto}/L$ [kWh/km]",
    fname_suffix: str = "",
) -> Path:
    """
    Charakterystyka wpływu parametru na jednostkowe zużycie energii.

    JEDEN panel: AC (linia ciągła) i DC (linia przerywana) razem; kolor = długość L.
    Każdy punkt to końcowe E całego przejazdu (ta sama metryka, co OAT/Sobol).
    metric: "E_per_km_kWh" (kWh/km) albo "E_per_btkm_Wh" (Wh/(bt*km)) — bliźniak.
    """
    from matplotlib.lines import Line2D

    spec = PARAM_SWEEPS[param_name]
    cmap = plt.get_cmap(CMAP_NAME)
    n_L = len(TRACK_LENGTHS_KM)
    L_colors = {L: cmap(i / max(1, n_L - 1)) for i, L in enumerate(TRACK_LENGTHS_KM)}

    fig, ax = plt.subplots(figsize=(10, 6))

    for system in SYSTEMS:
        ls = LINESTYLE_AC if system == "AC" else LINESTYLE_DC
        for L_km in TRACK_LENGTHS_KM:
            L_m = L_km * 1000.0
            pts = [
                (r["value_SI"], r[metric])
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
                linestyle=ls if multi else "None",
            )

    ax.set_xlabel(spec["xlabel"])
    ax.set_ylabel(ylabel)

    if param_name == "P_nom":
        ax.text(
            0.5,
            0.04,
            "DC: sufit trakcji 9 MW",
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
            style="italic",
            color="0.4",
        )
    if param_name == "v_max":
        ax.text(
            0.5,
            0.04,
            "DC: sufit 250 km/h",
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
            style="italic",
            color="0.4",
        )

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
    leg1 = ax.legend(
        handles=L_handles,
        title="Długość odcinka $L$",
        loc="center left",
        bbox_to_anchor=(1.01, 0.65),
        framealpha=0.95,
        fontsize=8,
    )
    ax.add_artist(leg1)
    sys_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=LINESTYLE_AC,
            linewidth=1.6,
            label="AC (2×25 kV)",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=LINESTYLE_DC,
            linewidth=1.6,
            label="DC (3 kV)",
        ),
    ]
    ax.legend(
        handles=sys_handles,
        title="System zasilania",
        loc="center left",
        bbox_to_anchor=(1.01, 0.22),
        framealpha=0.95,
        fontsize=8,
    )

    fig.suptitle(spec["label"], fontsize=14, y=1.00, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 0.80, 0.97))

    safe_name = param_name.replace("_", "")
    path = save_dir / f"influence_{safe_name}{fname_suffix}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  WYKRES 4×2: profile prędkości v(x) — analogiczny do plot_param_influence
# ═══════════════════════════════════════════════════════════════════════════


def plot_param_influence_velocity(
    param_name: str,
    vel_results: list[dict],
    save_dir: Path = OUTPUT_DIR,
) -> Path:
    """
    Profile prędkości v(x) dla wpływu parametru.

    JEDNA KOLUMNA: panele = prędkości zadane 250/300/350 km/h (SPEED_PANELS_KMH)
    przy stałej długości L = VELOCITY_PANEL_L_KM km. AC ciągła, DC przerywana.
    DC tylko do 250 km/h (zalecenie promotora) — w panelach 300/350 sam AC.
    Dla param=v_max panel pokazuje sam profil (prędkość zadana = panel).
    """
    from matplotlib.lines import Line2D

    spec = PARAM_SWEEPS[param_name]
    is_vmax = param_name == "v_max"
    values = spec["values"]
    cmap = plt.get_cmap(CMAP_NAME)
    colors = [cmap(i / max(1, len(values) - 1)) for i in range(len(values))]

    panels = SPEED_PANELS_KMH
    fig, axes = plt.subplots(len(panels), 1, figsize=(8.5, 12.5), sharey=True)
    if len(panels) == 1:
        axes = [axes]

    by_panel = {}
    for r in vel_results:
        by_panel.setdefault(r["v_panel_kmh"], []).append(r)

    for panel_idx, v_panel in enumerate(panels):
        ax = axes[panel_idx]
        rows = by_panel.get(v_panel, [])
        if is_vmax:
            for system, col, ls, lw in (
                ("DC", "#c1121f", LINESTYLE_DC, 1.9),
                ("AC", "#1f5fa6", LINESTYLE_AC, 1.6),
            ):
                rr = next((r for r in rows if r["system"] == system), None)
                if rr is not None:
                    ax.plot(
                        rr["x"] / 1000.0,
                        rr["v_ms"] * 3.6,
                        color=col,
                        linestyle=ls,
                        linewidth=lw,
                    )
        else:
            by_key = {(r["value_SI"], r["system"]): r for r in rows}
            for v_idx, val in enumerate(values):
                rdc = by_key.get((val, "DC"))
                if rdc is not None:
                    ax.plot(
                        rdc["x"] / 1000.0,
                        rdc["v_ms"] * 3.6,
                        color=colors[v_idx],
                        linestyle=LINESTYLE_DC,
                        linewidth=1.8,
                        alpha=0.75,
                    )
            for v_idx, val in enumerate(values):
                rac = by_key.get((val, "AC"))
                if rac is not None:
                    ax.plot(
                        rac["x"] / 1000.0,
                        rac["v_ms"] * 3.6,
                        color=colors[v_idx],
                        linestyle=LINESTYLE_AC,
                        linewidth=1.4,
                    )
        ax.set_xlim(0, VELOCITY_PANEL_L_KM)
        ax.set_ylim(0, max(panels) * 1.06)
        ax.set_xlabel("Pozycja $x$ [km]")
        ax.set_ylabel("Prędkość $v$ [km/h]")
        note = "" if v_panel <= DC_VELOCITY_CAP_KMH else "  (tylko AC — DC ≤ 250 km/h)"
        ax.set_title(f"$v_{{zad}}$ = {v_panel} km/h{note}")

    sys_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=LINESTYLE_AC,
            linewidth=1.6,
            label="AC (2×25 kV)",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=LINESTYLE_DC,
            linewidth=1.6,
            label="DC (3 kV, ≤ 250 km/h)",
        ),
    ]
    if is_vmax:
        sys_handles = [
            Line2D(
                [0],
                [0],
                color="#1f5fa6",
                linestyle=LINESTYLE_AC,
                linewidth=1.8,
                label="AC (2×25 kV)",
            ),
            Line2D(
                [0],
                [0],
                color="#c1121f",
                linestyle=LINESTYLE_DC,
                linewidth=1.8,
                label="DC (3 kV, ≤ 250 km/h)",
            ),
        ]
        fig.legend(
            handles=sys_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=2,
            title="System zasilania",
            frameon=True,
            framealpha=0.95,
        )
    else:
        val_handles = [
            Line2D([0], [0], color=colors[i], linewidth=2) for i in range(len(values))
        ]
        val_labels = [spec["fmt"](v) for v in values]
        fig.legend(
            val_handles,
            val_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=min(len(values), 6),
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
    fig.suptitle(
        f"{title_velocity}  (L = {VELOCITY_PANEL_L_KM} km)",
        fontsize=14,
        y=1.045,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.025, 1.0, 0.95))

    safe_name = param_name.replace("_", "")
    path = save_dir / f"influence_velocity_{safe_name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  WPŁYW DŁUGOŚCI ODCINKA — jeden panel
# ═══════════════════════════════════════════════════════════════════════════


def plot_length_influence(
    save_dir: Path = OUTPUT_DIR,
    n_workers: int | None = None,
    results: list[dict] | None = None,
    metric: str = "E_per_km_kWh",
    ylabel: str = "Jednostkowe zużycie energii $E_{pant,netto}/L$ [kWh/km]",
    fname_suffix: str = "",
) -> tuple[Path, list[dict]]:
    """
    E_per_km vs L dla kilku prędkości × 2 systemy. DC ograniczone do sufitu 250 km/h.

    Zwraca (ścieżka, wyniki) — wyniki można podać ponownie (arg `results`), by
    narysować bliźniaczy wykres w innej metryce bez powtarzania symulacji.
    """
    velocities_kmh = (250, 280, 310, 340, 370, 400)
    DC_CAP_KMH = 250
    lengths_km = list(range(50, 401, 25))

    if results is None:
        tasks = []
        for v_kmh in velocities_kmh:
            for L_km in lengths_km:
                for system in SYSTEMS:
                    if system == "DC" and v_kmh > DC_CAP_KMH:
                        continue
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

    cmap = plt.get_cmap(CMAP_NAME)
    colors = [
        cmap(i / max(1, len(velocities_kmh) - 1)) for i in range(len(velocities_kmh))
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    for v_idx, v_kmh in enumerate(velocities_kmh):
        for system in SYSTEMS:
            if system == "DC" and v_kmh > DC_CAP_KMH:
                continue
            linestyle = LINESTYLE_AC if system == "AC" else LINESTYLE_DC
            xs, ys = [], []
            for L_km in lengths_km:
                v_SI = v_kmh / 3.6
                L_m = L_km * 1000.0
                for r in results:
                    if (
                        abs(r["value_SI"] - v_SI) < 1e-6
                        and r["L_m"] == L_m
                        and r["system"] == system
                    ):
                        xs.append(L_km)
                        ys.append(r[metric])
                        break
            ax.plot(
                xs,
                ys,
                color=colors[v_idx],
                linestyle=linestyle,
                linewidth=1.6,
                marker="o",
                markersize=4,
                label=f"{v_kmh} km/h ({system})",
            )

    ax.set_xlabel("Długość odcinka $L$ [km]", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(
        "Wpływ długości odcinka na jednostkowe zużycie energii\n"
        "dla różnych prędkości eksploatacyjnych i systemów zasilania",
        fontsize=12,
    )
    ax.legend(loc="upper right", ncol=2, framealpha=0.95, fontsize=9)
    ax.set_xlim(0, max(lengths_km) * 1.02)

    path = save_dir / f"influence_length{fname_suffix}.png"
    fig.savefig(path)
    plt.close(fig)
    return path, results


# ═══════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════


def run_all_param_influence_plots(
    save_dir: Path = OUTPUT_DIR, n_workers: int | None = None
) -> list[Path]:
    """
    Dla kazdego parametru: wykres E/km (AC+DC na jednym), jego blizniak Wh/(bt*km)
    oraz profile predkosci (kolumna 50/150/250 km). Plus wplyw dlugosci (E/km + blizniak).
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    BTKM_YLABEL = "Jednostkowe zużycie energii [Wh/(bt·km)]"

    for param_name, spec in PARAM_SWEEPS.items():
        print(f"\n>>> Podrozdzial: {spec['label']}")
        tasks = []
        cap = spec.get("cap_si", {})
        for value_SI in spec["values"]:
            for L_km in TRACK_LENGTHS_KM:
                for system in SYSTEMS:
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

        print("    wykres energii (kWh/km) + blizniak (Wh/bt km)...")
        paths.append(plot_param_influence(param_name, results, save_dir))
        paths.append(
            plot_param_influence(
                param_name,
                results,
                save_dir,
                metric="E_per_btkm_Wh",
                ylabel=BTKM_YLABEL,
                fname_suffix="_btkm",
            )
        )

        print("    profile predkosci (panele 250/300/350 km/h, DC <=250)...")
        vel_tasks = []
        for v_panel in SPEED_PANELS_KMH:
            for system in SYSTEMS:
                if system == "DC" and v_panel > DC_VELOCITY_CAP_KMH:
                    continue
                if param_name == "v_max":
                    vel_tasks.append(
                        {
                            "param_name": "v_max",
                            "value_SI": v_panel / 3.6,
                            "v_panel_kmh": v_panel,
                            "system": system,
                        }
                    )
                else:
                    for value_SI in spec["values"]:
                        vel_tasks.append(
                            {
                                "param_name": param_name,
                                "value_SI": value_SI,
                                "v_panel_kmh": v_panel,
                                "system": system,
                            }
                        )
        vel_results = _run_vel_tasks(vel_tasks, n_workers)
        paths.append(plot_param_influence_velocity(param_name, vel_results, save_dir))
        print(f"    (lacznie {time.perf_counter() - t0:.1f} s)")

    print("\n>>> Podrozdzial: Wplyw dlugosci odcinka")
    path_E, len_results = plot_length_influence(save_dir, n_workers)
    paths.append(path_E)
    path_btkm, _ = plot_length_influence(
        save_dir,
        n_workers,
        results=len_results,
        metric="E_per_btkm_Wh",
        ylabel=BTKM_YLABEL,
        fname_suffix="_btkm",
    )
    paths.append(path_btkm)

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
