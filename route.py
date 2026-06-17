"""
route.py — Symulacja przejazdu po trasie z postojami pośrednimi.

Warstwa orkiestracji nad run_simulation(): każdy odcinek liczony jest jako
pełny cykl rozpędzanie→jazda→wybieg→hamowanie (od v=0 do v=0) istniejącym
silnikiem, z podmienionymi L, pochyleniem i systemem zasilania. Profile
odcinków są sklejane (przesunięcie x i t), między nimi wstawiany jest postój
5 min z czynnymi potrzebami własnymi (aux), a energia sumowana po odcinkach
plus aux postojów. Zwalidowany rdzeń symulacji nie jest modyfikowany.

Trzy trasy (każda ~378 km Warszawa-Wrocław, różna liczba postojów):
  1. Wwa-Łódź-Sieradz-Kępno-Wrocław   (3 postoje)
  2. Wwa-Sieradz-Wrocław               (1 postój)
  3. Wwa-Wrocław                       (0 postojów)

Pochylenia odcinków policzone z różnic wysokości miast (PRZYBLIŻONE):
  Warszawa 113 | Łódź 200 | Sieradz 140 | Kępno 180 | Wrocław 120 [m n.p.m.]

Autor: Jakub Król, PW WE, 2026
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from energy import J_TO_KWH, compute_energy
from parameters import Parameters
from simulation import run_simulation

# Pola energii [J] sumowane addytywnie po odcinkach (zgodne z EnergyResults)
_ENERGY_FIELDS = [
    "E_trakcja_kolo",
    "E_trakcja_pant",
    "E_aux",
    "E_ham_el_kolo",
    "E_rec_pant",
    "E_ham_mech",
    "E_pant_pobrana",
    "E_pant_netto",
]

N_SEATS: int = 500  # liczba miejsc (jak w energy.compute_energy)
DWELL_MIN: float = 5.0  # postój na stacji pośredniej [min], aux czynne
PHASE_DWELL: int = 5  # nowa faza profilu: postój (0-3 ruch, 4 hamowanie)


# ===========================================================================
#  STRUKTURY TRASY
# ===========================================================================
@dataclass
class Segment:
    """Odcinek między dwiema stacjami."""

    name: str
    length_km: float
    gradient_promille: float  # stałe pochylenie netto odcinka
    dwell_after_min: float = 0.0  # postój NA KOŃCU odcinka (0 dla stacji końcowej)


@dataclass
class Route:
    name: str
    segments: list[Segment]

    @property
    def length_km(self) -> float:
        return sum(s.length_km for s in self.segments)

    @property
    def n_stops(self) -> int:
        return max(len(self.segments) - 1, 0)


ROUTE_1 = Route(
    "Wwa-Łódź-Sieradz-Kępno-Wrocław",
    [
        Segment("Warszawa->Łódź", 133.0, +0.65, DWELL_MIN),
        Segment("Łódź->Sieradz", 65.0, -0.92, DWELL_MIN),
        Segment("Sieradz->Kępno", 95.0, +0.42, DWELL_MIN),
        Segment("Kępno->Wrocław", 85.0, -0.71, 0.0),
    ],
)

ROUTE_2 = Route(
    "Wwa-Sieradz-Wrocław",
    [
        Segment(
            "Warszawa->Sieradz", 198.0, +0.14, DWELL_MIN
        ),  # 133+65, bez postoju w Łodzi
        Segment("Sieradz->Wrocław", 180.0, -0.11, 0.0),  # 95+85, bez postoju w Kępnie
    ],
)

ROUTE_3 = Route(
    "Wwa-Wrocław",
    [
        Segment("Warszawa->Wrocław", 378.0, +0.02, 0.0),  # netto ~ płaska
    ],
)

ROUTES = [ROUTE_1, ROUTE_2, ROUTE_3]


# ===========================================================================
#  SYMULACJA POJEDYNCZEGO ODCINKA + SKLEJANIE
# ===========================================================================
def _run_segment(p_base: Parameters, seg: Segment, system: str):
    """Pełny cykl v=0->v=0 dla jednego odcinka. Zwraca (sim, energy)."""
    p = p_base.with_changes(
        L=seg.length_km * 1000.0,
        gradient=seg.gradient_promille,
        power_system=system,
    )
    profile = [(0.0, p.L, p.gradient)]
    sim = run_simulation(p, profile)
    energy = compute_energy(sim, p)
    return sim, energy


def _stitch_profile(seg_results, route: Route, n_dwell: int = 60) -> pd.DataFrame:
    """Skleja profile kinematyczne odcinków: przesuwa x i t, wstawia postoje."""
    parts, x_off, t_off = [], 0.0, 0.0
    for (sim, _), seg in zip(seg_results, route.segments):
        df = pd.DataFrame(
            {
                "x_m": sim.x + x_off,
                "t_s": sim.t + t_off,
                "v_kmh": sim.v * 3.6,
                "a_ms2": sim.a,
                "phase": sim.phase,
            }
        )
        parts.append(df)
        x_end = float(sim.x[-1] + x_off)
        t_end = float(sim.t[-1] + t_off)
        if seg.dwell_after_min > 0:
            dur = seg.dwell_after_min * 60.0
            dwell = pd.DataFrame(
                {
                    "x_m": x_end,
                    "t_s": t_end + np.linspace(0.0, dur, n_dwell),
                    "v_kmh": 0.0,
                    "a_ms2": 0.0,
                    "phase": PHASE_DWELL,
                }
            )
            parts.append(dwell)
            t_off = t_end + dur
        else:
            t_off = t_end
        x_off = x_end
    return pd.concat(parts, ignore_index=True)


def simulate_route(
    route: Route, system: str, p_base: Parameters | None = None, n_seats: int = N_SEATS
) -> dict:
    """
    Pełna symulacja trasy: odcinki -> sumowanie energii -> metryki -> profil.

    Zwraca słownik: E_kWh (bilans), metrics, profile (DataFrame), segments.
    """
    if p_base is None:
        p_base = Parameters.base()

    seg_results = [_run_segment(p_base, s, system) for s in route.segments]

    # --- Sumowanie energii (w dżulach) + aux postojów ---
    E = {k: 0.0 for k in _ENERGY_FIELDS}
    for _, en in seg_results:
        for k in _ENERGY_FIELDS:
            E[k] += float(getattr(en, k))
    t_dwell_s = route.n_stops * DWELL_MIN * 60.0
    E_aux_dwell = p_base.P_aux * t_dwell_s  # P_aux*t postojów [J]
    E["E_aux"] += E_aux_dwell
    E["E_pant_pobrana"] += E_aux_dwell
    E["E_pant_netto"] += E_aux_dwell

    # --- Metryki trasy ---
    L = route.length_km
    t_drive_s = sum(float(sim.t[-1]) for sim, _ in seg_results)
    e_net_kWh = E["E_pant_netto"] * J_TO_KWH
    metrics = {
        "L_km": L,
        "n_stops": route.n_stops,
        "E_per_km_kWh": e_net_kWh / L,
        "E_per_seat_km_Wh": e_net_kWh * 1000.0 / (L * n_seats),
        "E_jednostkowa_kWh_100km_t": e_net_kWh / (L * (p_base.m / 1000.0) / 100.0),
        "T_drive_min": t_drive_s / 60.0,
        "T_dwell_min": t_dwell_s / 60.0,
        "T_total_min": (t_drive_s + t_dwell_s) / 60.0,
        "v_avg_drive_kmh": L / (t_drive_s / 3600.0),
        "v_avg_total_kmh": L / ((t_drive_s + t_dwell_s) / 3600.0),
        "E_aux_dwell_kWh": E_aux_dwell * J_TO_KWH,
    }

    profile = _stitch_profile(seg_results, route)
    reached = all(sim.reached_v_set for sim, _ in seg_results)
    segments = [
        {
            "name": s.name,
            "L_km": s.length_km,
            "gradient": s.gradient_promille,
            "v_max_reached_kmh": float(sim.v_max_reached * 3.6),
            "reached_v_set": bool(sim.reached_v_set),
            "E_pant_netto_kWh": float(en.E_pant_netto * J_TO_KWH),
            "T_min": float(sim.t[-1] / 60.0),
        }
        for s, (sim, en) in zip(route.segments, seg_results)
    ]

    return {
        "route": route.name,
        "system": system,
        "E_kWh": {k: v * J_TO_KWH for k, v in E.items()},
        "metrics": metrics,
        "profile": profile,
        "reached_v_set_all": reached,
        "segments": segments,
    }


# ===========================================================================
#  SZYBKI TEST — uruchom: python route.py
# ===========================================================================
if __name__ == "__main__":
    print("Symulacja tras z postojami (silnik rzeczywisty)\n" + "=" * 70)
    for route in ROUTES:
        print(
            f"\n### {route.name}  (L={route.length_km:.0f} km, postoje={route.n_stops})"
        )
        for system in ("AC", "DC"):
            r = simulate_route(route, system)
            m = r["metrics"]
            flag = (
                ""
                if r["reached_v_set_all"]
                else "  [!] nie wszystkie odcinki osiagaja v_max"
            )
            print(
                f"  {system}: E_netto={r['E_kWh']['E_pant_netto']:7.0f} kWh | "
                f"E/km={m['E_per_km_kWh']:5.2f} | "
                f"v_avg(z post.)={m['v_avg_total_kmh']:5.1f} | "
                f"T={m['T_total_min']:5.1f} min | "
                f"rec={r['E_kWh']['E_rec_pant']:5.0f} kWh{flag}"
            )

    print("\n" + "=" * 70)
    print("### Przypadek 400 km/h — Wwa-Wrocław (bez postojów)")
    p400 = Parameters.base().with_changes(v_max=400.0 / 3.6)
    for system in ("AC", "DC"):
        r = simulate_route(ROUTE_3, system, p_base=p400)
        seg = r["segments"][0]
        print(
            f"  {system}: v_max osiagniete = {seg['v_max_reached_kmh']:.1f} km/h "
            f"(zadane 400) | E/km={r['metrics']['E_per_km_kWh']:.2f} kWh/km | "
            f"reached={seg['reached_v_set']}"
        )
