"""
main.py — Punkt wejścia symulacji ruchu pociągu KDP.

Uruchamia kompletny pipeline:
  1. Wczytanie parametrów scenariusza bazowego (parameters.py)
  2. Symulacja przejazdu (simulation.py)
  3. Bilans energetyczny (energy.py)
  4. Walidacja (validation.py)
  5. Eksport wyników CSV + JSON (results.py)
  6. Wykresy (plotting.py)

Aby uruchomić inny scenariusz - edytuj parameters.py (sekcja "STREFA EDYCJI")
lub przekaż argumenty przez kod (zobacz przykład na dole).

Autor: Jakub Król, PW WE, 2026
"""

from __future__ import annotations 

import sys 
import time 

from energy import compute_energy 
from parameters import OUTPUT_DIR ,Parameters 
from plotting import plot_all 
from results import export_all 
from simulation import run_simulation 
from validation import print_validation_report ,run_validation 


def run_scenario (p :Parameters ,tag :str ="base",make_plots :bool =True )->dict :
    """
    Uruchamia pełny pipeline dla zadanego zestawu parametrów.

    Args:
        p: Parametry symulacji.
        tag: Krótka etykieta scenariusza do nazw plików wyjściowych.
        make_plots: Czy generować wykresy (wolne dla batchowych analiz).

    Returns:
        Słownik z wynikami (sim, energy, validation_checks, output_paths).
    """
    print ()
    print ("╔"+"═"*78 +"╗")
    print (f"║  SCENARIUSZ: {tag :<63s} ║")
    print ("╚"+"═"*78 +"╝")
    print ()
    print (p .summary ())
    print ()


    t0 =time .perf_counter ()
    print (">>> [1/5] Symulacja ruchu...")
    sim =run_simulation (p )
    print (f"    czas obliczeń: {time .perf_counter ()-t0 :.2f} s")
    print (
    f"    T_przejazdu: {sim .T_total /60 :.2f} min, v_avg = {sim .v_avg *3.6 :.1f} km/h"
    )
    print ()


    print (">>> [2/5] Bilans energetyczny...")
    energy =compute_energy (sim ,p )
    print (
    f"    E_pant_netto = {energy .E_pant_netto /3.6e6 :.2f} kWh, "
    f"E_jedn = {energy .E_jednostkowa :.2f} kWh/(100km·t)"
    )
    print ()


    print (">>> [3/5] Eksport wyników...")
    output_paths =export_all (sim ,energy ,p ,tag =tag )
    for key ,path in output_paths .items ():
        print (f"    {key .upper ()}: {path .name }")
    print ()


    if make_plots :
        print (">>> [4/5] Generowanie wykresów...")
        plot_paths =plot_all (sim ,energy ,p )
        print (f"    Zapisano {len (plot_paths )} wykresów do {OUTPUT_DIR }/")
        print ()
    else :
        plot_paths =[]


    print (">>> [5/5] Walidacja modelu...")
    print ()
    checks =run_validation (sim ,energy ,p )
    print_validation_report (checks )

    return {
    "sim":sim ,
    "energy":energy ,
    "checks":checks ,
    "output_paths":output_paths ,
    "plot_paths":plot_paths ,
    }


def main ()->int :
    """Główna funkcja - uruchamia scenariusz bazowy."""

    p_base =Parameters .base ()
    result =run_scenario (p_base ,tag ="base",make_plots =True )


    print ()
    print (f"✓ Zakończono. Wszystkie wyniki w: {OUTPUT_DIR .absolute ()}")


    has_errors =any (c .severity =="ERROR"and not c .passed for c in result ["checks"])
    return 1 if has_errors else 0 


if __name__ =="__main__":
    sys .exit (main ())
