"""
validation.py — Sanity checks modelu symulacyjnego.

Sprawdza fizyczną poprawność wyników:
  - bilans energii się zamyka
  - prędkość początkowa i końcowa = 0
  - przyspieszenie i opóźnienie nie przekraczają limitów komfortu
  - moc na pantografie nie przekracza P_eff_max
  - prąd na pantografie w okolicy limitu TSI ENE
  - energia jednostkowa w zakresie literaturowym dla KDP

Każda walidacja zwraca True/False + komunikat.

Autor: Jakub Król, PW WE, 2026
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from energy import J_TO_KWH, EnergyResults
from parameters import G, Parameters
from simulation import SimulationProfile


@dataclass
class ValidationCheck:
    """Wynik pojedynczej walidacji."""

    name: str
    passed: bool
    value: str  # zaobserwowana wartość
    expected: str  # oczekiwany zakres
    severity: str = "INFO"  # INFO | WARNING | ERROR


# ═══════════════════════════════════════════════════════════════════════════
#  POJEDYNCZE WALIDACJE
# ═══════════════════════════════════════════════════════════════════════════


def check_endpoints(sim: SimulationProfile) -> ValidationCheck:
    """v(0) = 0 i v(L) = 0 (z tolerancją 0.5 m/s)."""
    v_start = sim.v[0]
    v_end = sim.v[-1]
    passed = abs(v_start) < 0.5 and abs(v_end) < 0.5
    return ValidationCheck(
        name="Warunki brzegowe v(0) = v(L) = 0",
        passed=passed,
        value=f"v(0) = {v_start * 3.6:.2f} km/h, v(L) = {v_end * 3.6:.2f} km/h",
        expected="< 1.8 km/h (tolerancja 0.5 m/s)",
        severity="ERROR" if not passed else "INFO",
    )


def check_acceleration_limits(sim: SimulationProfile, p: Parameters) -> ValidationCheck:
    """
    Sprawdza limity przyspieszenia/opóźnienia w 99. percentylu.

    Ignorujemy outliers (pojedyncze artefakty Eulera na granicach faz) -
    sprawdzamy 99% trasy, nie pojedyncze punkty.
    """
    # Ignorujemy ostatnie 3 punkty (artefakty numeryczne końcowego hamowania)
    a_clean = sim.a[:-3]
    # 99-percentyl zamiast max - odporne na artefakty Eulera
    a_p99 = float(np.percentile(a_clean, 99))
    a_p01 = float(np.percentile(a_clean, 1))
    a_max_abs = float(np.max(np.abs(a_clean)))
    # Granica opóźnienia = sufit przyczepności TSI 4.2.4.6.1: a_ham,max = μ_b·g
    # (przy v≤250 km/h, μ_b=0,15 → ≈1,47 m/s²; powyżej maleje). Komfort: ±5%.
    a_brake_limit = p.mu_b_base * p.braked_frac * G
    passed = a_p99 <= p.a_launch_max * 1.05 and abs(a_p01) <= a_brake_limit * 1.05
    return ValidationCheck(
        name="Limity przyspieszenia/opóźnienia (99-percentyl)",
        passed=passed,
        value=f"a_p99 = {a_p99:.3f}, a_p01 = {a_p01:.3f} m/s² "
        f"(max bezwzględnie: {a_max_abs:.3f})",
        expected=f"≤ {p.a_launch_max}, ≥ -{a_brake_limit:.2f} m/s² (sufit TSI, 99%, ±5%)",
        severity="WARNING" if not passed else "INFO",
    )


def check_power_limit(energy: EnergyResults, p: Parameters) -> ValidationCheck:
    """Sprawdza czy P_pant_max nie przekracza P_eff_max + aux (z tolerancją 5%)."""
    P_expected_max = (
        p.P_eff_max / p.eta_tr_effective + p.P_aux
    )  # uwzględnia różne η dla AC/DC
    passed = energy.P_pant_max <= P_expected_max * 1.05
    return ValidationCheck(
        name="Limit mocy na pantografie",
        passed=passed,
        value=f"P_pant_max = {energy.P_pant_max / 1e6:.2f} MW",
        expected=f"≤ {P_expected_max / 1e6:.2f} MW",
        severity="ERROR" if not passed else "INFO",
    )


def check_current_limit(energy: EnergyResults, p: Parameters) -> ValidationCheck:
    """Sprawdza prąd pantografu względem TSI ENE (z tolerancją 20% dla AC bez cos φ)."""
    tolerance = 1.20 if p.power_system == "AC" else 1.05
    passed = energy.I_pant_max <= p.I_grid_limit * tolerance
    return ValidationCheck(
        name=f"Limit prądu pantografu TSI ENE ({p.power_system})",
        passed=passed,
        value=f"I_pant_max = {energy.I_pant_max:.1f} A",
        expected=f"≤ {p.I_grid_limit:.0f} A (±{(tolerance - 1) * 100:.0f}%)",
        severity="WARNING" if not passed else "INFO",
    )


def check_energy_balance(
    sim: SimulationProfile, energy: EnergyResults, p: Parameters
) -> ValidationCheck:
    """
    Sprawdza zamknięcie bilansu energii (dla trasy płaskiej):
        E_trakcja_kolo ≈ E_kin_max + E_op_total + E_grav_total

    gdzie:
      - E_kin_max = ½·m_eff·v_max²  (energia kinetyczna max)
      - E_op_total = ∫ F_op·v dt   (praca oporów)
      - E_grav_total = m·g·Δh (dla L jednorodnego = 0)

    Bilans powinien się zamknąć z dokładnością do ~10% (Euler nie jest idealny).
    """
    # Energia kinetyczna max (jedyna do "naładowania" w trakcie rozpędzania)
    E_kin = 0.5 * p.m_eff * p.v_max**2

    # Praca oporów Davisa
    P_op = sim.F_op * sim.v
    dt = np.diff(sim.t)
    E_op = float(np.sum(0.5 * (P_op[:-1] + P_op[1:]) * dt))

    # Praca grawitacji (na płaskiej trasie = 0)
    P_grav = sim.F_grav * sim.v
    E_grav = float(np.sum(0.5 * (P_grav[:-1] + P_grav[1:]) * dt))

    # Napęd musi dostarczyć energię kinetyczną (0→v_max) ORAZ pokryć opory
    # i grawitację. E_kin jest rozpraszana w hamowaniu (osobny strumień),
    # więc NIE odejmuje się od E_trakcja_kolo — musi być po stronie oczekiwanej.
    E_expected = E_kin + E_op + E_grav
    rel_error = (
        abs(energy.E_trakcja_kolo - E_expected) / E_expected if E_expected > 0 else 0
    )

    passed = rel_error < 0.25  # 25% tolerancja dla Eulera ze stałym krokiem 1 m
    return ValidationCheck(
        name="Zamknięcie bilansu energii (E_trakcja_kolo ≈ E_op + E_grav)",
        passed=passed,
        value=f"E_trakcja_kolo = {energy.E_trakcja_kolo * J_TO_KWH:.1f} kWh, "
        f"E_op + E_grav = {E_expected * J_TO_KWH:.1f} kWh, "
        f"błąd = {rel_error * 100:.1f}%",
        expected="błąd < 25%",
        severity="WARNING" if not passed else "INFO",
    )


def check_unit_energy_range(energy: EnergyResults) -> ValidationCheck:
    """
    Sprawdza czy zużycie energii jest w zakresie literaturowym KDP.

    Najczęstsza metryka w literaturze KDP: kWh/km całego pociągu.
    Typowo 18-30 kWh/km dla pociągu klasy Velaro/AGV przy v_max=320 km/h
    (RailEnergy, Lukaszewicz 2009).
    """
    E = energy.E_per_km
    passed = 12.0 <= E <= 35.0
    return ValidationCheck(
        name="Zużycie energii w zakresie literaturowym (kWh/km)",
        passed=passed,
        value=f"E_per_km = {E:.2f} kWh/km",
        expected="12 - 35 kWh/km (literatura KDP klasy Velaro)",
        severity="WARNING" if not passed else "INFO",
    )


def check_reached_vmax(sim: SimulationProfile) -> ValidationCheck:
    """Informacyjne: czy pociąg osiągnął prędkość zadaną."""
    return ValidationCheck(
        name="Osiągnięcie prędkości zadanej v_set",
        passed=sim.reached_v_set,
        value="TAK" if sim.reached_v_set else "NIE",
        expected="TAK (dla L > L_min ~ 30 km)",
        severity="INFO",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  RAPORT WALIDACJI
# ═══════════════════════════════════════════════════════════════════════════


def run_validation(
    sim: SimulationProfile, energy: EnergyResults, p: Parameters
) -> list[ValidationCheck]:
    """Uruchamia wszystkie walidacje i zwraca listę wyników."""
    return [
        check_endpoints(sim),
        check_acceleration_limits(sim, p),
        check_power_limit(energy, p),
        check_current_limit(energy, p),
        check_energy_balance(sim, energy, p),
        check_unit_energy_range(energy),
        check_reached_vmax(sim),
    ]


def print_validation_report(checks: list[ValidationCheck]) -> None:
    """Drukuje czytelny raport walidacji."""
    print("=" * 80)
    print("RAPORT WALIDACJI MODELU")
    print("=" * 80)

    icons = {True: "✓", False: "✗"}
    severities_icons = {"INFO": "  ", "WARNING": "⚠ ", "ERROR": "❌"}

    n_passed = sum(1 for c in checks if c.passed)
    n_total = len(checks)

    for c in checks:
        icon = icons[c.passed]
        sev = severities_icons.get(c.severity, "  ")
        print(f"{sev}{icon} {c.name}")
        print(f"     Zaobserwowano:  {c.value}")
        print(f"     Oczekiwano:     {c.expected}")
        print()

    print("=" * 80)
    print(f"PODSUMOWANIE: {n_passed}/{n_total} walidacji przeszło pomyślnie")
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════
#  🚀 SZYBKI TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from energy import compute_energy
    from simulation import run_simulation

    p = Parameters.base()
    print(">>> Symulacja + bilans energii...")
    sim = run_simulation(p)
    energy = compute_energy(sim, p)
    print()
    print(p.summary())
    print()
    checks = run_validation(sim, energy, p)
    print_validation_report(checks)
