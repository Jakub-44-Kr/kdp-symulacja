"""
physics.py — Funkcje fizyczne modelu pociągu KDP.

Moduł zawiera czyste funkcje matematyczne opisujące mechanikę
ruchu pociągu: charakterystykę trakcyjną, opory ruchu, hamowanie
oraz składową grawitacyjną. Wszystkie wartości w SI.

Odwołania do wzorów odnoszą się do pracy magisterskiej.

Autor: Jakub Król, PW WE, 2026
"""

from __future__ import annotations

from typing import Sequence

from parameters import G, Parameters

# ───────────────────────────────────────────────────────────────────────────
#  Typowanie profilu trasy
# ───────────────────────────────────────────────────────────────────────────
# Profil to lista segmentów: [(x_start_m, x_end_m, gradient_promille), ...]
# Dla scenariusza bazowego: [(0.0, L, 0.0)]
TrackProfile = Sequence[tuple[float, float, float]]


# ═══════════════════════════════════════════════════════════════════════════
#  CHARAKTERYSTYKA TRAKCYJNA F_tr(v) — rozdz. 4.2, rys. 3
# ═══════════════════════════════════════════════════════════════════════════


def F_traction(v: float, p: Parameters) -> float:
    """
    Siła trakcyjna w funkcji prędkości [N].

    Charakterystyka dwuczęściowa (uproszczenie - bez regionu osłabienia pola):
      - Region 1 (rozruchowy): v < v_b  →  F = F_max
      - Region 2 (stała moc):  v ≥ v_b  →  F = P_eff_max / v

    gdzie v_b = P_eff_max / F_max  (prędkość łamania).

    Dla v → 0 zwracamy F_max (zabezpieczenie przed dzieleniem przez zero).

    Args:
        v: Aktualna prędkość pociągu [m/s], v ≥ 0.
        p: Parametry symulacji.

    Returns:
        Siła trakcyjna [N], zawsze ≥ 0.
    """
    if v <= p.v_breakpoint:
        # Region stałej siły rozruchowej
        return p.F_max
    # Region stałej mocy: F = P/v
    return p.P_eff_max / v


# ═══════════════════════════════════════════════════════════════════════════
#  OPORY RUCHU — równanie Davisa (rozdz. 3.2)
# ═══════════════════════════════════════════════════════════════════════════


def F_davis(v: float, p: Parameters) -> float:
    """
    Opór ruchu wg równania Davisa: F_op = A + B·v + C·v² [N].

    Dla v < 0 zwracamy F_op(|v|) - opory zawsze działają przeciwnie do ruchu,
    ale moduł |v| nie ma fizycznego znaczenia w naszym modelu (v ≥ 0 zawsze).

    Args:
        v: Prędkość pociągu [m/s].
        p: Parametry symulacji (zawierają współczynniki A, B, C).

    Returns:
        Opór ruchu [N], zawsze ≥ 0.
    """
    return p.davis_A + p.davis_B * v + p.davis_C * v * v


# ═══════════════════════════════════════════════════════════════════════════
#  PROFIL TRASY I SKŁADOWA GRAWITACYJNA
# ═══════════════════════════════════════════════════════════════════════════


def gradient_at(x: float, profile: TrackProfile) -> float:
    """
    Zwraca pochylenie trasy [‰] w punkcie x [m].

    Profil to lista segmentów (x_start, x_end, gradient_promille).
    Zakładamy że segmenty są niepokrywające się i pokrywają cały zakres [0, L].

    Tolerancja TOL na końcach segmentów chroni przed błędami zaokrągleń
    float w siatce np.arange (ostatni punkt może minimalnie wyjść poza L).

    Args:
        x: Pozycja na trasie [m].
        profile: Lista segmentów trasy.

    Returns:
        Pochylenie [‰] w punkcie x. Dodatnie = pod górę, ujemne = z górki.
    """
    TOL = 1.0  # [m] tolerancja na zaokrąglenia float

    for x_start, x_end, i_promille in profile:
        if x_start - TOL <= x <= x_end + TOL:
            return i_promille

    # Fallback: jeśli x tuż poza zakresem, zwróć pochylenie najbliższego segmentu
    if x < profile[0][0]:
        return profile[0][2]
    return profile[-1][2]


def F_gravity(x: float, p: Parameters, profile: TrackProfile) -> float:
    """
    Składowa grawitacyjna oporu ruchu [N].

    Wzór: F_g = m · g · sin(α(x)) ≈ m · g · i(x) / 1000

    Przybliżenie sin(α) ≈ tan(α) ≈ i/1000 jest dopuszczalne dla i ≤ 35‰
    (TSI INF, rozdz. 2.4.2 pracy) - błąd < 0.06%.

    Dodatnia wartość = wzniesienie (siła przeciwna ruchowi).
    Ujemna wartość = spadek (siła wspomagająca ruch).

    Args:
        x: Pozycja na trasie [m].
        p: Parametry symulacji.
        profile: Profil trasy.

    Returns:
        Siła grawitacyjna [N]. Znak: dodatni = pod górę, ujemny = z górki.
    """
    i_promille = gradient_at(x, profile)
    return p.m * G * i_promille / 1000.0


# ═══════════════════════════════════════════════════════════════════════════
#  HAMOWANIE — model mieszany (rozdz. 4.2)
# ═══════════════════════════════════════════════════════════════════════════


def F_brake_max_electric(v: float, p: Parameters) -> float:
    """
    Maksymalna elektryczna siła hamulcowa [N] - charakterystyka symetryczna
    do trakcji.

    Założenie modelu: napęd w trybie generatorowym ma tę samą charakterystykę
    co w trybie napędowym (Steimel 2008):
      - v ≥ v_brake_min ∧ v < v_b   →  F_ham,el = F_max
      - v ≥ v_breakpoint            →  F_ham,el = P_eff_max / v
      - v < v_brake_min             →  F_ham,el = 0  (poniżej progu, tylko hamulec mechaniczny)

    Args:
        v: Prędkość pociągu [m/s].
        p: Parametry symulacji.

    Returns:
        Maksymalna elektryczna siła hamulcowa [N], ≥ 0.
        (Znak ujemny obsługiwany w równaniu ruchu — tu zwracamy wartość bezwzględną.)
    """
    if v < p.v_brake_min:
        return 0.0
    if v < p.v_breakpoint:
        return p.F_max
    return p.P_eff_max / v


def F_brake_required(
    v: float,
    x: float,
    p: Parameters,
    profile: TrackProfile,
    target_decel: float | None = None,
) -> float:
    """
    Wymagana CAŁKOWITA siła hamulcowa (elektryczna + mechaniczna) [N]
    aby osiągnąć zadane opóźnienie target_decel.

    Wzór (17) z pracy:
        F_ham,zad = (m + m_rot) · a_decel - F_op(v) - m·g·sin(α(x))

    Logika:
      - Opory ruchu F_op WSPOMAGAJĄ hamowanie (zmniejszają wymaganą siłę)
      - Składowa grawitacyjna:
          * wzniesienie (+) wspomaga hamowanie (zmniejsza wymaganą siłę)
          * spadek (-) utrudnia hamowanie (zwiększa wymaganą siłę)

    Args:
        v: Prędkość [m/s].
        x: Pozycja [m].
        p: Parametry.
        profile: Profil trasy.
        target_decel: Zadane opóźnienie [m/s²], > 0. Default: a_brake_max.

    Returns:
        Wymagana siła hamulcowa [N] (wartość bezwzględna).
        Może być 0 jeśli opory + grawitacja same wystarczą do osiągnięcia opóźnienia.
    """
    if target_decel is None:
        target_decel = p.a_brake_max

    F_op = F_davis(v, p)
    F_g = F_gravity(x, p, profile)  # + dla wzniesienia, - dla spadku

    # Wymagana siła z równania ruchu: m_eff·a_decel = F_brake + F_op + F_g
    # (na wzniesieniu F_g pomaga hamować)
    F_required = p.m_eff * target_decel - F_op - F_g

    # Jeśli opory + grawitacja same wystarczą - hamowanie nie jest potrzebne
    return max(0.0, F_required)


def split_brake_force(
    F_required: float,
    v: float,
    p: Parameters,
) -> tuple[float, float]:
    """
    Rozdziela wymaganą siłę hamulcową na elektryczną i mechaniczną.

    Strategia (rozdz. 4.2, "hamowanie mieszane"):
      - Najpierw używamy maksymalnie hamulca elektrycznego (do limitu F_brake_max_electric)
      - Resztę dokłada hamulec mechaniczny
      - Poniżej v_brake_min: cały moc hamowania mechanicznie

    Args:
        F_required: Wymagana całkowita siła hamulcowa [N].
        v: Prędkość [m/s].
        p: Parametry.

    Returns:
        Tuple (F_el, F_mech) - elektryczna i mechaniczna składowa [N], obie ≥ 0.
    """
    F_el_max = F_brake_max_electric(v, p)
    F_el = min(F_required, F_el_max)
    F_mech = F_required - F_el
    return F_el, F_mech


# ═══════════════════════════════════════════════════════════════════════════
#  POMOCNICZE — siła wypadkowa w fazach sterowania (wzór 20)
# ═══════════════════════════════════════════════════════════════════════════


def F_resultant_in_phase(
    phase: int,
    v: float,
    x: float,
    p: Parameters,
    profile: TrackProfile,
) -> float:
    """
    Wypadkowa siła sterująca u(v, x) wg wzoru (20) z pracy.

    Args:
        phase: Numer fazy: 1=rozpędzanie, 2=jazda ustalona, 3=wybieg, 4=hamowanie.
        v: Prędkość [m/s].
        x: Pozycja [m].
        p: Parametry.
        profile: Profil trasy.

    Returns:
        Wypadkowa siła sterująca [N]. Dodatnia = ciągnięcie, ujemna = hamowanie.

    Raises:
        ValueError: dla nieprawidłowego numeru fazy.
    """
    if phase == 1:
        # Rozpędzanie: pełna siła trakcyjna
        return F_traction(v, p)
    if phase == 2:
        # Jazda ustalona: napęd kompensuje opory + grawitację
        return F_davis(v, p) + F_gravity(x, p, profile)
    if phase == 3:
        # Wybieg (coasting): napęd wyłączony
        return 0.0
    if phase == 4:
        # Hamowanie: siła wynikająca z modelu hamowania mieszanego (z minusem)
        F_req = F_brake_required(v, x, p, profile)
        return -F_req
    raise ValueError(f"Nieznana faza sterowania: {phase}")


# ═══════════════════════════════════════════════════════════════════════════
#  🚀 SZYBKI TEST — uruchom: python physics.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from parameters import Parameters

    p = Parameters.base()
    profile: TrackProfile = [(0.0, p.L, p.gradient)]

    print("=== Test physics.py — scenariusz bazowy ===")
    print(f"v_breakpoint = {p.v_breakpoint * 3.6:.1f} km/h")
    print()
    print(
        f"{'v [km/h]':>10} {'F_tr [kN]':>12} {'F_davis [kN]':>14} {'F_brake_el_max [kN]':>22}"
    )
    print("-" * 60)
    for v_kmh in [0, 30, 50, 71, 100, 150, 200, 250, 320, 400]:
        v_ms = v_kmh / 3.6
        Ft = F_traction(v_ms, p) / 1000.0
        Fd = F_davis(v_ms, p) / 1000.0
        Fb = F_brake_max_electric(v_ms, p) / 1000.0
        print(f"{v_kmh:>10} {Ft:>12.1f} {Fd:>14.2f} {Fb:>22.1f}")

    print()
    print("=== Test pochyleń ===")
    profile_alt: TrackProfile = [
        (0.0, 50_000.0, 0.0),  # 0-50 km płasko
        (50_000.0, 100_000.0, 5.0),  # 50-100 km pod górę 5‰
        (100_000.0, p.L, -3.0),  # 100-180 km z górki -3‰
    ]
    for x_km in [0, 25, 50, 75, 100, 125, 180]:
        x_m = x_km * 1000.0
        i = gradient_at(x_m, profile_alt)
        Fg = F_gravity(x_m, p, profile_alt) / 1000.0
        print(f"x = {x_km:>4} km   i = {i:+.1f}‰   F_g = {Fg:+8.2f} kN")
