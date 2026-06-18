"""
routes_analysis.py — Analiza i wizualizacja przejazdów po trasach z postojami.

Dodatek do pracy: trzy trasy Warszawa–Wrocław o tej samej długości (~378 km),
różniące się liczbą postojów pośrednich (0 / 1 / 3). Pozwala to wyizolować
energetyczny koszt postojów oraz różnicę systemów zasilania (2×25 kV AC / 3 kV DC)
na trasie rzeczywistej, zgodnie z modelem hamowania TSI 4.2.4.6.1.

Generowane rysunki (format wg parameters.FIG_FORMAT — domyślnie PNG):
  WARIANT 1  trasy_E_droga_<param>     E_kumulowana(x), 3 panele=trasy, krzywe=param
  WARIANT 2  trasy_v_E_masa            E/km(v_max), 3 panele=trasy, krzywe=masa
  KOSZT      trasy_koszt_postoju       E/km i energia odzysku wg trasy i systemu
  WRAŻLIWOŚĆ trasy_oat_ranking         elastyczność E/km wg parametru i trasy
Tabele zapisywane do outputs/ jako CSV.

Uruchom:  python routes_analysis.py
Autor: Jakub Król, PW WE, 2026
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.lines import Line2D

from energy import J_TO_KWH, compute_energy
from parameters import OUTPUT_DIR, Parameters, save_figure
from route import ROUTE_3, ROUTES, simulate_route
from simulation import run_simulation

# ───────────────────────────────────────────────────────────────────────────
#  STYL (spójny z plotting.py / plots_param_influence.py)
# ───────────────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

CMAP = colormaps["plasma"]
LS = {"AC": "-", "DC": "--"}  # AC ciągła, DC przerywana (konwencja pracy)
SYS_LABEL = {"AC": "2×25 kV", "DC": "3 kV"}
SYSTEMS = ("AC", "DC")

# Prędkość docelowa per system: AC 320 km/h, DC ograniczone do 250 km/h.
# (zalecenie promotora — DC nie przekracza 250 km/h także na trasach).
ROUTE_VMAX_KMH = {"AC": 320, "DC": 250}


def _route_base(system, **changes):
    """Parametry bazowe tras z prędkością docelową zależną od systemu
    (AC 320 / DC 250 km/h). Jeśli `changes` zawiera v_max (przemiatanie),
    wartość docelowa jest nadpisywana."""
    return (
        Parameters.base()
        .with_changes(v_max=ROUTE_VMAX_KMH[system] / 3.6)
        .with_changes(**changes)
    )


def _colors(values):
    """Mapa wartość→kolor z palety plasma (ucięta do 0.85 dla czytelności)."""
    n = len(values)
    return {v: CMAP(i / max(n - 1, 1) * 0.85) for i, v in enumerate(values)}


# Trasa odniesienia dla czasów w legendzie (bez postojów → czysty czas jazdy)
REF_ROUTE_FOR_TIMES = ROUTE_3


def _ref_times(param_name, value_si):
    """Czas przejazdu (min) na trasie odniesienia, osobno AC i DC."""
    t = {}
    for sysn in SYSTEMS:
        # DC nie przekracza 250 km/h także przy przemiataniu v_max
        v_use = (
            min(value_si, ROUTE_VMAX_KMH["DC"] / 3.6)
            if (param_name == "v_max" and sysn == "DC")
            else value_si
        )
        p = _route_base(sysn, **{param_name: v_use})
        t[sysn] = simulate_route(REF_ROUTE_FOR_TIMES, sysn, p_base=p)["metrics"][
            "T_total_min"
        ]
    return t["AC"], t["DC"]


def _value_label(spec, v):
    """Etykieta wartości + czasy AC/DC na trasie odniesienia."""
    t_ac, t_dc = _ref_times(spec["param"], spec["to_si"](v))
    return f"{spec['fmt'](v)}  (AC {t_ac:.0f} / DC {t_dc:.0f} min)"


# ───────────────────────────────────────────────────────────────────────────
#  KONFIGURACJA PRZEMIATAŃ
# ───────────────────────────────────────────────────────────────────────────
# WARIANT 1: dla każdego parametru — wartości, konwersja do SI, etykieta, tytuł.
# Pozostałe parametry pozostają na wartości bazowej (v_max=320 km/h, m=600 t, P=12 MW).
SWEEP_V1 = {
    "v_max": {
        "param": "v_max",
        "values": [250, 280, 310, 340, 370],
        "to_si": lambda v: v / 3.6,
        "fmt": lambda v: f"{v} km/h",
        "title": r"Wpływ prędkości $v_{max}$ na energię",
        "leg_title": r"$v_{max}$ (kolor) · system (styl)",
    },
    "m": {
        "param": "m",
        "values": [450, 600, 750],
        "to_si": lambda v: v * 1000.0,
        "fmt": lambda v: f"{v} t",
        "title": r"Wpływ masy składu $m$ na energię",
        "leg_title": r"$m$ (kolor) · system (styl)",
    },
    "P_nom": {
        "param": "P_nom",
        "values": [6, 9, 12],
        "to_si": lambda v: v * 1e6,
        "fmt": lambda v: f"{v} MW",
        "title": r"Wpływ mocy znamionowej $P$ na energię",
        "leg_title": r"$P$ (kolor) · system (styl)",
    },
}

# WARIANT 2: oś X = v_max [km/h] (gęściej przy suficie DC ~325), krzywe = masa.
VMAX_AXIS_V2 = [250, 270, 290, 310, 320, 340, 360, 380, 400]
MASS_CLASSES_V2 = [450, 600, 750]  # [t]


# ───────────────────────────────────────────────────────────────────────────
#  ENERGIA KUMULOWANA WZDŁUŻ CAŁEJ TRASY (sklejone odcinki + skok aux na postoju)
# ───────────────────────────────────────────────────────────────────────────
def route_cumulative_energy(route, system, **changes):
    """
    Zwraca (x_km, E_cum_kWh) wzdłuż całej trasy. Na postoju doliczany jest
    pionowy skok energii potrzeb własnych (aux), spójnie z route.simulate_route.
    """
    base = _route_base(system, **changes)
    x_off, E_off, xs, Es = 0.0, 0.0, [], []
    for seg in route.segments:
        p = base.with_changes(
            L=seg.length_km * 1000.0,
            gradient=seg.gradient_promille,
            power_system=system,
        )
        sim = run_simulation(p, [(0.0, p.L, p.gradient)])
        en = compute_energy(sim, p)
        dt = np.diff(sim.t)
        inc = 0.5 * (en.P_pant_net[:-1] + en.P_pant_net[1:]) * dt
        Ecum = np.zeros_like(en.P_pant_net)
        Ecum[1:] = np.cumsum(inc)
        xs.append(sim.x + x_off)
        Es.append(Ecum * J_TO_KWH + E_off)
        x_off = float(sim.x[-1] + x_off)
        E_off = float(Ecum[-1] * J_TO_KWH + E_off)
        if seg.dwell_after_min > 0:  # skok aux na postoju
            E_aux = p.P_aux * seg.dwell_after_min * 60.0 * J_TO_KWH
            xs.append(np.array([x_off, x_off]))
            Es.append(np.array([E_off, E_off + E_aux]))
            E_off += E_aux
    return np.concatenate(xs) / 1000.0, np.concatenate(Es)


# ───────────────────────────────────────────────────────────────────────────
#  WARIANT 1 — E_kumulowana(x), 3 panele = trasy, krzywe = wartości parametru
# ───────────────────────────────────────────────────────────────────────────
def plot_wariant1(param_name, save_dir=OUTPUT_DIR):
    spec = SWEEP_V1[param_name]
    values = spec["values"]
    colors = _colors(values)

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11), sharex=True)
    fig.subplots_adjust(top=0.82, hspace=0.24)

    for ax, route in zip(axes, ROUTES):
        for v in values:
            for sysn in SYSTEMS:
                # DC: zalecenie ≤250 km/h — pomiń krzywe v_max>250 dla DC
                if param_name == "v_max" and sysn == "DC" and v > ROUTE_VMAX_KMH["DC"]:
                    continue
                x, E = route_cumulative_energy(
                    route, sysn, **{param_name: spec["to_si"](v)}
                )
                ax.plot(
                    x, E, color=colors[v], linestyle=LS[sysn], linewidth=1.3, alpha=0.9
                )
        ax.set_title(f"{route.name}   (postoje: {route.n_stops})", fontsize=11)
        ax.set_ylabel(r"$E_{pant}$ [kWh]")
        ax.set_xlim(0, 378)
        ax.margins(y=0.05)
    axes[-1].set_xlabel(r"Pozycja $x$ [km]")

    leg_v = [
        Line2D([0], [0], color=colors[v], lw=2.2, label=_value_label(spec, v))
        for v in values
    ]
    leg_s = [
        Line2D(
            [0], [0], color="dimgray", ls=LS[s], lw=2.2, label=f"{s} ({SYS_LABEL[s]})"
        )
        for s in SYSTEMS
    ]
    fig.legend(
        handles=leg_v + leg_s,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.93),
        frameon=True,
        fancybox=True,
        title=spec["leg_title"] + r"  ·  czas: trasa Wwa–Wrocław (bez postojów)",
    )
    fig.suptitle(
        spec["title"] + r" — zestawienie trzech tras",
        y=0.985,
        fontweight="bold",
        fontsize=13,
    )

    safe = param_name.replace("_", "")
    path = save_figure(fig, save_dir, f"trasy_E_droga_{safe}")
    plt.close(fig)
    return path


# ───────────────────────────────────────────────────────────────────────────
#  WARIANT 3 — profil prędkości v(x), 3 panele = trasy, krzywe = param (m / P)
# ───────────────────────────────────────────────────────────────────────────
def plot_wariant_velocity(param_name, save_dir=OUTPUT_DIR, decim=50):
    """Profil v(x) całej trasy ze spadkami do 0 na postojach. param ∈ {m, P_nom}."""
    spec = SWEEP_V1[param_name]
    values = spec["values"]
    colors = _colors(values)
    v_max_base = max(ROUTE_VMAX_KMH.values())  # górny zakres osi Y (AC 320)

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11), sharex=True)
    fig.subplots_adjust(top=0.82, hspace=0.24)

    for ax, route in zip(axes, ROUTES):
        for v in values:
            for sysn in SYSTEMS:
                p_base = _route_base(sysn, **{param_name: spec["to_si"](v)})
                r = simulate_route(route, sysn, p_base=p_base)
                prof = r["profile"]
                x = prof["x_m"].to_numpy()[::decim] / 1000.0
                vk = prof["v_kmh"].to_numpy()[::decim]
                ax.plot(
                    x,
                    vk,
                    color=colors[v],
                    linestyle=LS[sysn],
                    linewidth=1.2,
                    alpha=0.9,
                    rasterized=True,
                )
        for vref, col in (
            (ROUTE_VMAX_KMH["AC"], "#1f5fa6"),
            (ROUTE_VMAX_KMH["DC"], "#c1121f"),
        ):
            ax.axhline(
                vref, color=col, linestyle=":", linewidth=0.9, alpha=0.55, zorder=0
            )
        ax.set_title(f"{route.name}   (postoje: {route.n_stops})", fontsize=11)
        ax.set_ylabel(r"Prędkość $v$ [km/h]")
        ax.set_xlim(0, 378)
        ax.set_ylim(0, v_max_base * 1.08)
    axes[-1].set_xlabel(r"Pozycja $x$ [km]")

    leg_v = [
        Line2D([0], [0], color=colors[v], lw=2.2, label=_value_label(spec, v))
        for v in values
    ]
    leg_s = [
        Line2D(
            [0], [0], color="dimgray", ls=LS[s], lw=2.2, label=f"{s} ({SYS_LABEL[s]})"
        )
        for s in SYSTEMS
    ]
    fig.legend(
        handles=leg_v + leg_s,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.93),
        frameon=True,
        fancybox=True,
        title=spec["leg_title"] + r"  ·  czas: trasa Wwa–Wrocław (bez postojów)",
    )
    fig.suptitle(
        spec["title"].replace("na energię", "na profil prędkości") + r" — trzy trasy",
        y=0.985,
        fontweight="bold",
        fontsize=13,
    )

    safe = param_name.replace("_", "")
    path = save_figure(fig, save_dir, f"trasy_v_droga_{safe}")
    plt.close(fig)
    return path


# ───────────────────────────────────────────────────────────────────────────
#  WARIANT 2 — E/km(v_max), 3 panele = trasy, krzywe = masa (DC urywa się ~325)
# ───────────────────────────────────────────────────────────────────────────
def plot_wariant2_mass(save_dir=OUTPUT_DIR):
    colors = _colors(MASS_CLASSES_V2)

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11), sharex=True)
    fig.subplots_adjust(top=0.80, hspace=0.22)

    for ax, route in zip(axes, ROUTES):
        for m_t in MASS_CLASSES_V2:
            # AC: pełne przemiatanie v_max
            ekm = []
            for vk in VMAX_AXIS_V2:
                p_base = Parameters.base().with_changes(v_max=vk / 3.6, m=m_t * 1000.0)
                r = simulate_route(route, "AC", p_base=p_base)
                ekm.append(r["metrics"]["E_per_km_kWh"])
            ax.plot(
                VMAX_AXIS_V2,
                ekm,
                color=colors[m_t],
                linestyle=LS["AC"],
                linewidth=1.4,
                marker="o",
                markersize=3,
                alpha=0.9,
            )
            # DC: zalecenie promotora <=250 km/h - pojedynczy punkt pracy (marker)
            vk_dc = ROUTE_VMAX_KMH["DC"]
            p_dc = Parameters.base().with_changes(v_max=vk_dc / 3.6, m=m_t * 1000.0)
            r_dc = simulate_route(route, "DC", p_base=p_dc)
            ax.plot(
                [vk_dc],
                [r_dc["metrics"]["E_per_km_kWh"]],
                color=colors[m_t],
                marker="s",
                markersize=8,
                linestyle="None",
                markeredgecolor="black",
                markeredgewidth=0.6,
                zorder=4,
            )
        ax.set_title(f"{route.name}   (postoje: {route.n_stops})", fontsize=11)
        ax.set_ylabel(r"$E_{pant}$/km [kWh/km]")
        # granica zalecenia DC (250 km/h): powyżej - zakres tylko dla AC
        dc_cap = ROUTE_VMAX_KMH["DC"]
        ax.axvline(
            dc_cap, color="dimgray", linestyle=":", linewidth=1.0, alpha=0.7, zorder=1
        )
        ax.text(
            dc_cap - 3,
            ax.get_ylim()[1],
            "DC: zalecenie ≤250 km/h (marker)",
            fontsize=8,
            color="dimgray",
            ha="right",
            va="top",
            rotation=90,
        )
        ax.text(
            (dc_cap + VMAX_AXIS_V2[-1]) / 2,
            ax.get_ylim()[0],
            "zakres tylko dla AC",
            fontsize=7.5,
            color="gray",
            ha="center",
            va="bottom",
            style="italic",
        )
    axes[-1].set_xlabel(
        r"Prędkość zadana $v_{max}$ (zadana, nie zawsze osiągnięta) [km/h]"
    )

    leg_m = [
        Line2D([0], [0], color=colors[m], lw=2.2, marker="o", ms=4, label=f"{m} t")
        for m in MASS_CLASSES_V2
    ]
    leg_s = [
        Line2D(
            [0],
            [0],
            color="dimgray",
            ls=LS["AC"],
            lw=2.2,
            label=f"AC ({SYS_LABEL['AC']}) - przemiatanie",
        ),
        Line2D(
            [0],
            [0],
            color="dimgray",
            marker="s",
            ms=7,
            ls="None",
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=f"DC ({SYS_LABEL['DC']}) - punkt 250 km/h",
        ),
    ]
    fig.legend(
        handles=leg_m + leg_s,
        loc="upper center",
        ncol=len(MASS_CLASSES_V2) + 2,
        bbox_to_anchor=(0.5, 0.91),
        frameon=True,
        fancybox=True,
        title=r"masa $m$ (kolor) · system (styl)",
    )
    fig.suptitle(
        r"Wpływ prędkości i masy na jednostkowe zużycie energii"
        r" — trzy trasy",
        y=0.985,
        fontweight="bold",
        fontsize=13,
    )

    path = save_figure(fig, save_dir, "trasy_v_E_masa")
    plt.close(fig)
    return path


# ───────────────────────────────────────────────────────────────────────────
#  KOSZT POSTOJU — E/km i energia odzysku wg trasy i systemu
# ───────────────────────────────────────────────────────────────────────────
def analyze_cost_of_stops(save_dir=OUTPUT_DIR):
    rows = []
    for route in ROUTES:
        for sysn in SYSTEMS:
            r = simulate_route(route, sysn, p_base=_route_base(sysn))
            m = r["metrics"]
            rows.append(
                {
                    "trasa": route.name,
                    "system": sysn,
                    "postoje": route.n_stops,
                    "E_per_km_kWh": m["E_per_km_kWh"],
                    "E_netto_kWh": r["E_kWh"]["E_pant_netto"],
                    "E_rec_kWh": r["E_kWh"]["E_rec_pant"],
                    "v_avg_total_kmh": m["v_avg_total_kmh"],
                    "T_total_min": m["T_total_min"],
                }
            )
    df = pd.DataFrame(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    n_stops = [r.n_stops for r in ROUTES]
    width = 0.35
    xpos = np.arange(len(ROUTES))
    col = {"AC": "#1f77b4", "DC": "#d62728"}
    for i, sysn in enumerate(SYSTEMS):
        sub = df[df.system == sysn].set_index("postoje").loc[n_stops]
        ax1.bar(
            xpos + (i - 0.5) * width,
            sub["E_per_km_kWh"],
            width,
            label=f"{sysn} ({SYS_LABEL[sysn]})",
            color=col[sysn],
            alpha=0.85,
        )
        ax2.bar(
            xpos + (i - 0.5) * width,
            sub["E_rec_kWh"],
            width,
            label=f"{sysn} ({SYS_LABEL[sysn]})",
            color=col[sysn],
            alpha=0.85,
        )
    for ax, ttl, ylab in [
        (ax1, "Jednostkowe zużycie energii", r"$E_{pant}$/km [kWh/km]"),
        (ax2, "Energia odzyskana (rekuperacja)", r"$E_{rec}$ [kWh]"),
    ]:
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"{n} post." for n in n_stops])
        ax.set_title(ttl)
        ax.set_ylabel(ylab)
        ax.legend()
    fig.suptitle(
        "Energetyczny koszt postojów (L = 378 km, ten sam dla wszystkich tras)",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    path = save_figure(fig, save_dir, "trasy_koszt_postoju")
    plt.close(fig)
    return path, df


# ───────────────────────────────────────────────────────────────────────────
#  PEŁNY AnalizaOAT per trasa (schemat jak sensitivity.py + kolumna trasa)
# ───────────────────────────────────────────────────────────────────────────
# Na trasie L jest ustalone (378 km), a pochylenie jest własnością odcinków,
# więc przemiatamy tylko parametry niezależne od geometrii trasy.
SWEEP_RANGES_ROUTE = {
    "v_max": {
        "values": [v / 3.6 for v in range(250, 401, 10)],  # 250-400 km/h, 16
        "label": r"Prędkość $v_{max}$",
        "unit": "km/h",
        "display": lambda x: x * 3.6,
    },
    "m": {
        "values": [450_000.0 + 50_000.0 * i for i in range(7)],  # 450-750/50t (7)
        "label": r"Masa składu $m$",
        "unit": "t",
        "display": lambda x: x / 1000.0,
    },
    "P_nom": {
        "values": [6e6 + 1e6 * i for i in range(7)],  # 6-12 MW/1MW (7)
        "label": r"Moc znamionowa $P$",
        "unit": "MW",
        "display": lambda x: x / 1e6,
        # Dla DC sufit trakcji = 6 MW (P_eff_max nie rośnie powyżej) — nie
        # przemiatamy w zakresie pozbawionym fizycznego sensu.
        "cap_si": {"DC": 6e6},
    },
}


def _sweep_values(spec: dict, system: str) -> list:
    """Wartości przemiatu z uwzględnieniem sufitu per system (np. P_nom DC ≤ 6 MW)."""
    cap = spec.get("cap_si", {}).get(system)
    if cap is None:
        return spec["values"]
    return [v for v in spec["values"] if v <= cap + 1.0]


def run_oat_sweep_per_route(save_dir=OUTPUT_DIR):
    """
    Pełny AnalizaOAT dla 3 tras × 2 systemy. Każdy parametr zmieniany po
    całym zakresie (pozostałe na wartości bazowej). Zwraca DataFrame ze
    schematem zgodnym z sensitivity_sweep.csv, rozszerzonym o 'trasa'/'postoje'.
    """
    rows = []
    for route in ROUTES:
        for sysn in SYSTEMS:
            for pname, spec in SWEEP_RANGES_ROUTE.items():
                for val in _sweep_values(spec, sysn):
                    if (
                        pname == "v_max"
                        and sysn == "DC"
                        and val > ROUTE_VMAX_KMH["DC"] / 3.6
                    ):
                        continue
                    p_base = _route_base(sysn, **{pname: val})
                    r = simulate_route(route, sysn, p_base=p_base)
                    m = r["metrics"]
                    seg_reached = all(s["reached_v_set"] for s in r["segments"])
                    rows.append(
                        {
                            "trasa": route.name,
                            "postoje": route.n_stops,
                            "system": sysn,
                            "parameter": pname,
                            "value_SI": val,
                            "value_display": spec["display"](val),
                            "unit": spec["unit"],
                            "E_pant_netto_kWh": r["E_kWh"]["E_pant_netto"],
                            "E_pant_pobrana_kWh": r["E_kWh"]["E_pant_pobrana"],
                            "E_rec_pant_kWh": r["E_kWh"]["E_rec_pant"],
                            "E_per_km_kWh": m["E_per_km_kWh"],
                            "E_per_seat_km_Wh": m["E_per_seat_km_Wh"],
                            "T_min": m["T_total_min"],
                            "v_avg_kmh": m["v_avg_total_kmh"],
                            "reached_v_set": seg_reached,
                        }
                    )
    return pd.DataFrame(rows)


def plot_oat_sweep(df_sweep, save_dir=OUTPUT_DIR):
    """
    Wykres wpływu: E/km(wartość parametru), panele = parametry (poziomo),
    krzywe = trasy (kolor) × system (styl). Pokazuje KSZTAŁT zależności i jak
    liczba postojów go modyfikuje (dopełnienie elastyczności z run_oat_per_route).
    """
    params = list(SWEEP_RANGES_ROUTE)
    route_colors = _colors([r.n_stops for r in ROUTES])

    fig, axes = plt.subplots(1, len(params), figsize=(13.5, 4.6))
    for ax, pname in zip(axes, params):
        spec = SWEEP_RANGES_ROUTE[pname]
        sub_p = df_sweep[df_sweep.parameter == pname]
        for route in ROUTES:
            for sysn in SYSTEMS:
                s = sub_p[(sub_p.trasa == route.name) & (sub_p.system == sysn)]
                s = s.sort_values("value_display")
                if len(s) == 1:  # np. moc DC (sufit 6 MW) — jeden punkt
                    ax.plot(
                        s["value_display"],
                        s["E_per_km_kWh"],
                        color=route_colors[route.n_stops],
                        marker="s",
                        markersize=7,
                        markeredgecolor="white",
                        markeredgewidth=0.7,
                        linestyle="none",
                        alpha=0.9,
                        zorder=5,
                    )
                else:
                    ax.plot(
                        s["value_display"],
                        s["E_per_km_kWh"],
                        color=route_colors[route.n_stops],
                        linestyle=LS[sysn],
                        linewidth=1.5,
                        marker="o",
                        markersize=3,
                        alpha=0.9,
                    )
        ax.set_xlabel(f"{spec['label']} [{spec['unit']}]")
        ax.set_ylabel(r"$E_{pant}$/km [kWh/km]")
        ax.set_title(spec["label"])
        if pname == "P_nom":  # adnotacja: DC tylko 6 MW
            ax.text(
                6.0,
                ax.get_ylim()[1],
                "DC: tylko 6 MW\n(limit trakcji)",
                fontsize=7.5,
                color="dimgray",
                ha="left",
                va="top",
            )
    # strefa limitu DC tylko na panelu v_max
    axes[0].axvspan(325, 400, color="gray", alpha=0.06, zorder=0)
    axes[0].text(
        363,
        axes[0].get_ylim()[0],
        "DC: limit mocy",
        fontsize=7.5,
        color="gray",
        ha="center",
        va="bottom",
    )

    leg_r = [
        Line2D(
            [0],
            [0],
            color=route_colors[r.n_stops],
            lw=2.2,
            marker="o",
            ms=4,
            label=f"{r.n_stops} post.",
        )
        for r in ROUTES
    ]
    leg_s = [
        Line2D(
            [0], [0], color="dimgray", ls=LS[s], lw=2.2, label=f"{s} ({SYS_LABEL[s]})"
        )
        for s in SYSTEMS
    ]
    fig.legend(
        handles=leg_r + leg_s,
        loc="upper center",
        ncol=len(ROUTES) + 2,
        bbox_to_anchor=(0.5, 1.13),
        frameon=True,
        fancybox=True,
        title="liczba postojów (kolor) · system (styl)",
    )
    fig.suptitle(
        "AnalizaOAT jednostkowego zużycia energii — trzy trasy",
        y=1.0,
        fontweight="bold",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = save_figure(fig, save_dir, "trasy_oat_sweep")
    plt.close(fig)
    return path


# ───────────────────────────────────────────────────────────────────────────
#  WRAŻLIWOŚĆ OAT per trasa — elastyczność E/km (porównanie z analizą bazową)
# ───────────────────────────────────────────────────────────────────────────
OAT_PARAMS = {
    "v_max": {"base": 320.0 / 3.6, "to_si": lambda v: v, "name": r"$v_{max}$"},
    "m": {"base": 600.0 * 1000, "to_si": lambda v: v, "name": r"$m$"},
    "P_nom": {"base": 12.0 * 1e6, "to_si": lambda v: v, "name": r"$P$"},
}
OAT_DELTA = 0.10  # ±10 % — spójnie z sensitivity.py (analiza bazowa)


def _ekm(route, system, **changes):
    p = _route_base(system, **changes)
    return simulate_route(route, system, p_base=p)["metrics"]["E_per_km_kWh"]


def run_oat_per_route(save_dir=OUTPUT_DIR):
    rows = []
    for route in ROUTES:
        for sysn in SYSTEMS:
            E0 = _ekm(route, sysn)
            for pname, spec in OAT_PARAMS.items():
                # baza v_max zależna od systemu (AC 320 / DC 250 km/h)
                p0 = (ROUTE_VMAX_KMH[sysn] / 3.6) if pname == "v_max" else spec["base"]
                Ep = _ekm(route, sysn, **{pname: p0 * (1 + OAT_DELTA)})
                Em = _ekm(route, sysn, **{pname: p0 * (1 - OAT_DELTA)})
                s_plus = (Ep - E0) / E0 / OAT_DELTA
                s_minus = (E0 - Em) / E0 / OAT_DELTA
                rows.append(
                    {
                        "trasa": route.name,
                        "postoje": route.n_stops,
                        "system": sysn,
                        "parametr": pname,
                        "S_plus": s_plus,
                        "S_minus": s_minus,
                        "S_avg": 0.5 * (s_plus + s_minus),
                        "asymetria": s_plus - s_minus,
                        "E0_per_km": E0,
                    }
                )
    df = pd.DataFrame(rows)

    # Wykres: |S_avg| wg parametru, grupowane po liczbie postojów (system AC)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, sysn in zip(axes, SYSTEMS):
        sub = df[df.system == sysn]
        params = list(OAT_PARAMS)
        xpos = np.arange(len(params))
        width = 0.25
        stop_levels = sorted(sub["postoje"].unique())
        scol = _colors(stop_levels)
        for j, ns in enumerate(stop_levels):
            vals = [
                sub[(sub.postoje == ns) & (sub.parametr == pp)]["S_avg"].abs().iloc[0]
                for pp in params
            ]
            ax.bar(
                xpos + (j - 1) * width,
                vals,
                width,
                label=f"{ns} postojów",
                color=scol[ns],
                alpha=0.85,
            )
        ax.set_xticks(xpos)
        ax.set_xticklabels([OAT_PARAMS[p]["name"] for p in params])
        ax.set_title(f"System {sysn} ({SYS_LABEL[sysn]})")
        ax.set_ylabel(r"$|S_{avg}|$  (elastyczność $E$/km)")
        ax.legend(title="liczba postojów")
    fig.suptitle(
        "Wrażliwość OAT jednostkowego zużycia energii — wpływ liczby postojów",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    path = save_figure(fig, save_dir, "trasy_oat_ranking")
    plt.close(fig)
    return path, df


# ───────────────────────────────────────────────────────────────────────────
#  ORKIESTRACJA
# ───────────────────────────────────────────────────────────────────────────
def main():
    from parameters import FIG_FORMAT

    print(f"Analiza tras — format rysunków: {FIG_FORMAT}\n" + "=" * 64)

    print("WARIANT 1 — E_kumulowana(x):")
    for pn in SWEEP_V1:
        print(f"  {plot_wariant1(pn).name}")

    print("WARIANT 3 — profil prędkości v(x) (masa, moc):")
    for pn in ("m", "P_nom"):
        print(f"  {plot_wariant_velocity(pn).name}")

    print("WARIANT 2 — E/km(v_max), krzywe = masa:")
    print(f"  {plot_wariant2_mass().name}")

    print("KOSZT POSTOJU:")
    path_c, df_c = analyze_cost_of_stops()
    df_c.to_csv(OUTPUT_DIR / "trasy_koszt_postoju.csv", index=False)
    print(f"  {path_c.name}  +  trasy_koszt_postoju.csv")
    print(
        df_c.to_string(
            index=False,
            formatters={
                "E_per_km_kWh": "{:.2f}".format,
                "E_netto_kWh": "{:.0f}".format,
                "E_rec_kWh": "{:.0f}".format,
                "v_avg_total_kmh": "{:.1f}".format,
                "T_total_min": "{:.1f}".format,
            },
        )
    )

    print("\nWRAŻLIWOŚĆ — pełny analiza OAT per trasa:")
    df_sweep = run_oat_sweep_per_route()
    df_sweep.to_csv(OUTPUT_DIR / "trasy_sensitivity_sweep.csv", index=False)
    print(f"  trasy_sensitivity_sweep.csv  ({len(df_sweep)} wierszy)")
    print(f"  {plot_oat_sweep(df_sweep).name}")

    print("\nWRAŻLIWOŚĆ OAT per trasa:")
    path_o, df_o = run_oat_per_route()
    df_o.to_csv(OUTPUT_DIR / "trasy_oat_elasticity.csv", index=False)
    print(f"  {path_o.name}  +  trasy_oat_elasticity.csv")
    piv = df_o[df_o.system == "AC"].pivot_table(
        index="parametr", columns="postoje", values="S_avg"
    )
    print("  Elastyczność S_avg (AC) wg liczby postojów:")
    print(piv.to_string(float_format="%+.3f"))


if __name__ == "__main__":
    main()
