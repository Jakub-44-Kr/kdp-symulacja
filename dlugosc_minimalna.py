"""
dlugosc_minimalna.py — Minimalna długość odcinka do osiągnięcia prędkości zadanej.

Wariant DODATKOWY, w pełni samodzielny: korzysta wyłącznie z istniejącego modelu
(parameters/simulation) i NIE modyfikuje żadnego innego pliku ani nie wpływa na
resztę pipeline'u. Dla scenariusza bazowego (m = 600 t, i = 0 ‰), przy MAKSYMALNEJ
mocy każdego systemu (AC 12 MW, DC 9 MW), wyznacza najmniejszą długość odcinka L,
na której pociąg zdąży rozpędzić się do zadanej prędkości (profil "trójkątny":
rozpęd → wierzchołek równy prędkości zadanej → hamowanie do zera).

Cel: podkreślić różnicę AC/DC — niższa moc DC oznacza dłuższy rozbieg do tej samej
prędkości oraz niższy pułap prędkości na płaskim torze.

Wyniki (wszystkie z modelu): tabela (CSV + LaTeX) oraz wykres L_min(v).

Uruchomienie:
    python dlugosc_minimalna.py

Autor: Jakub Król, PW WE, 2026
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from parameters import OUTPUT_DIR, Parameters
from simulation import run_simulation

# ─────────────────────────────────────────────────────────────────────────────
#  KONFIGURACJA
# ─────────────────────────────────────────────────────────────────────────────

# Maksymalna moc per system (twardy limit na pantografie)
P_MAX = {"AC": 12e6, "DC": 9e6}
SYS_LABEL = {"AC": "2×25 kV AC (12 MW)", "DC": "3 kV DC (9 MW)"}

# Prędkości zadane: zestaw do TABELI oraz gęstszy do WYKRESU
TABLE_SPEEDS_KMH = [160, 200, 250, 280, 300, 320, 350]
PLOT_SPEEDS_KMH = list(range(140, 351, 10))

# Tolerancja "osiągnięcia" prędkości i precyzja wyszukiwania binarnego
V_TOL_KMH = 0.4
L_TOL_KM = 0.05
L_SEARCH_HI_KM = 500.0  # górna granica przeszukiwania (powyżej => uznaj za pułap)

# Styl wykresu (spójny z pozostałymi rysunkami pracy)
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
LINESTYLE = {"AC": "-", "DC": "--"}
COLOR = {"AC": "#1f5fa6", "DC": "#c1121f"}


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL — funkcje pomocnicze (tylko ODCZYT istniejącego modelu)
# ─────────────────────────────────────────────────────────────────────────────


def _apex_kmh(L_km: float, vt_kmh: float, system: str) -> float:
    """Maksymalna prędkość osiągnięta na odcinku L [km] przy prędkości zadanej vt.

    Scenariusz bazowy (m, pochylenie itd. z Parameters.base()), moc = sufit systemu.
    """
    p = Parameters.base().with_changes(
        power_system=system,
        v_max=vt_kmh / 3.6,
        L=L_km * 1000.0,
        P_nom=P_MAX[system],
    )
    return float(run_simulation(p).v.max()) * 3.6


def ceiling_kmh(system: str) -> float:
    """Pułap prędkości na płaskim torze (zadanie 450 km/h na bardzo długim odcinku)."""
    return _apex_kmh(600.0, 450.0, system)


def min_length_km(vt_kmh: float, system: str) -> float | None:
    """Najmniejsze L [km], na którym pociąg osiąga prędkość vt.

    Zwraca None, jeżeli vt leży powyżej pułapu systemu (nieosiągalna na żadnym L).
    Wyszukiwanie binarne — funkcja "osiągnięto" jest monotoniczna względem L.
    """
    if _apex_kmh(L_SEARCH_HI_KM, vt_kmh, system) < vt_kmh - V_TOL_KMH:
        return None
    lo, hi = 2.0, L_SEARCH_HI_KM
    while hi - lo > L_TOL_KM:
        mid = 0.5 * (lo + hi)
        if _apex_kmh(mid, vt_kmh, system) >= vt_kmh - V_TOL_KMH:
            hi = mid
        else:
            lo = mid
    return hi


def compute(speeds_kmh: list[int]) -> dict:
    """Zwraca {v: {'AC': L_min|None, 'DC': L_min|None}} dla zadanych prędkości."""
    return {vt: {s: min_length_km(vt, s) for s in ("AC", "DC")} for vt in speeds_kmh}


# ─────────────────────────────────────────────────────────────────────────────
#  TABELA
# ─────────────────────────────────────────────────────────────────────────────


def build_table(data: dict) -> pd.DataFrame:
    rows = []
    for vt in TABLE_SPEEDS_KMH:
        la, ld = data[vt]["AC"], data[vt]["DC"]
        rows.append(
            {
                "v_zadana_kmh": vt,
                "Lmin_AC_km": round(la, 1) if la is not None else None,
                "Lmin_DC_km": round(ld, 1) if ld is not None else None,
                "roznica_DC_AC_km": (
                    round(ld - la, 1) if (la is not None and ld is not None) else None
                ),
                "krotnosc_DC_AC": (
                    round(ld / la, 2) if (la is not None and ld is not None) else None
                ),
            }
        )
    return pd.DataFrame(rows)


def to_latex(df: pd.DataFrame, ceilings: dict) -> str:
    def cell(x) -> str:
        return "{—}" if x is None or pd.isna(x) else f"{x:.1f}"

    body = "\n".join(
        f"    {int(r.v_zadana_kmh)} & {cell(r.Lmin_AC_km)} & {cell(r.Lmin_DC_km)} "
        f"& {cell(r.roznica_DC_AC_km)} \\\\"
        for r in df.itertuples()
    )

    head = (
        r"\begin{table}[htbp]" + "\n  \\centering\n"
        r"  \caption{Minimalna długość odcinka $L_\mathrm{min}$ do osiągnięcia "
        r"prędkości zadanej (scenariusz bazowy: $m=600$~t, $i=0$\textperthousand), "
        r"przy maksymalnej mocy systemu. Pułap prędkości: AC $\approx$"
        + f"{ceilings['AC']:.0f}"
        + r"~km/h, DC $\approx$"
        + f"{ceilings['DC']:.0f}"
        + r"~km/h.}"
        + "\n"
        r"  \label{tab:dlugosc-minimalna}" + "\n"
        r"  \begin{tabular}{S[table-format=3.0] S[table-format=3.1] "
        r"S[table-format=3.1] S[table-format=2.1]}" + "\n"
        r"    \toprule" + "\n"
        r"    {$v_\mathrm{zad}$ [km/h]} & {$L_\mathrm{min}$ AC [km]} "
        r"& {$L_\mathrm{min}$ DC [km]} & {$\Delta$ (DC$-$AC) [km]} \\" + "\n"
        r"    \midrule" + "\n"
    )
    tail = (
        "\n    " + r"\bottomrule" + "\n  " + r"\end{tabular}" + "\n"
        r"  \vspace{2pt}{\footnotesize\\ Symbol „---'' oznacza prędkość powyżej "
        r"pułapu danego systemu (nieosiągalną na żadnej długości odcinka).}" + "\n"
        r"\end{table}" + "\n"
    )
    return head + body + tail


# ─────────────────────────────────────────────────────────────────────────────
#  WYKRES
# ─────────────────────────────────────────────────────────────────────────────


def plot_min_length(data: dict, ceilings: dict, save_dir: Path = OUTPUT_DIR) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for s in ("AC", "DC"):
        xs = [v for v in PLOT_SPEEDS_KMH if data[v][s] is not None]
        ys = [data[v][s] for v in xs]
        ax.plot(
            xs,
            ys,
            color=COLOR[s],
            linestyle=LINESTYLE[s],
            marker="o",
            markersize=4,
            linewidth=1.9,
            label=SYS_LABEL[s],
        )

    ymax = ax.get_ylim()[1]
    for s in ("AC", "DC"):
        ax.axvline(ceilings[s], color=COLOR[s], linestyle=":", linewidth=1.2, alpha=0.8)
        ax.text(
            ceilings[s],
            ymax * 0.97,
            f"pułap {ceilings[s]:.0f} km/h ",
            color=COLOR[s],
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
        )

    ax.set_xlabel(r"Prędkość zadana $v_\mathrm{zad}$ [km/h]")
    ax.set_ylabel(r"Minimalna długość odcinka $L_\mathrm{min}$ [km]")
    ax.set_title(
        "Minimalna długość odcinka do osiągnięcia prędkości zadanej\n"
        "(scenariusz bazowy, maksymalna moc systemu)"
    )
    ax.set_xlim(min(PLOT_SPEEDS_KMH) - 5, max(ceilings.values()) + 8)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", framealpha=0.95)

    path = save_dir / "dlugosc_minimalna.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  GŁÓWNY PRZEBIEG
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ceilings = {s: ceiling_kmh(s) for s in ("AC", "DC")}
    print("Pułap prędkości na płaskim torze:")
    for s in ("AC", "DC"):
        print(f"  {SYS_LABEL[s]:>22}: {ceilings[s]:.0f} km/h")

    speeds = sorted(set(PLOT_SPEEDS_KMH) | set(TABLE_SPEEDS_KMH))
    data = compute(speeds)

    df = build_table(data)
    csv_path = OUTPUT_DIR / "dlugosc_minimalna.csv"
    df.to_csv(csv_path, index=False)
    tex_path = OUTPUT_DIR / "dlugosc_minimalna.tex"
    tex_path.write_text(to_latex(df, ceilings), encoding="utf-8")
    png_path = plot_min_length(data, ceilings)

    print(
        "\nTabela — minimalna długość odcinka L_min [km] (scenariusz bazowy, max moc):"
    )
    print(
        df.to_string(
            index=False,
            formatters={
                "Lmin_AC_km": lambda x: "—" if x is None else f"{x:.1f}",
                "Lmin_DC_km": lambda x: "—" if x is None else f"{x:.1f}",
                "roznica_DC_AC_km": lambda x: "—" if x is None else f"{x:+.1f}",
                "krotnosc_DC_AC": lambda x: "—" if x is None else f"{x:.2f}×",
            },
        )
    )
    print(f"\n  CSV:    {csv_path.name}")
    print(f"  LaTeX:  {tex_path.name}")
    print(f"  Wykres: {png_path.name}")
