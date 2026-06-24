"""
energy.py — Bilans energetyczny przejazdu pociągu KDP.

Moduł post-processingu: dostaje gotowy SimulationProfile i wylicza:
  - momentalne moce na kole i pantografie (tablice)
  - prąd na pantografie (uproszczony, bez cos φ dla AC)
  - całkowite energie w fazach trakcyjnych, hamowania, rekuperacji
  - energię jednostkową E [kWh/(100 km·t)] - metryka porównawcza KDP

Konwencja znaków:
  - P_pant > 0 = pobór z sieci (trakcja, aux)
  - P_pant < 0 = zwrot do sieci (rekuperacja, tylko gdy η_grid > 0)
  - E_netto = pobrana - zwrócona

Autor: Jakub Król, PW WE, 2026
"""

from __future__ import annotations 

from dataclasses import dataclass 

import numpy as np 

from parameters import Parameters 
from simulation import SimulationProfile 


J_TO_KWH :float =1.0 /3.6e6 







def _F_brake_max_electric_vec (v :np .ndarray ,p :Parameters )->np .ndarray :
    """
    Wektorowa wersja physics.F_brake_max_electric.

    Dla każdego v w tablicy:
      - v < v_brake_min        → 0 (poniżej progu, tylko mechaniczny)
      - v_brake_min ≤ v ≤ v_1  → F_max (region stałej siły)
      - v_1 < v ≤ v_2          → P_eff_max / v (region stałej mocy)
      - v > v_2                → P_eff_max · v_2 / v² (osłabianie pola)
    """
    F =np .zeros_like (v )

    mask_const =(v >=p .v_brake_min )&(v <=p .v_breakpoint )
    F [mask_const ]=p .F_max 

    mask_power =(v >p .v_breakpoint )&(v <=p .v_field_weak )
    F [mask_power ]=p .P_eff_max /v [mask_power ]

    mask_fw =v >p .v_field_weak 
    F [mask_fw ]=p .P_eff_max *p .v_field_weak /(v [mask_fw ]*v [mask_fw ])
    return F 







@dataclass 
class EnergyResults :
    """
    Kompletny bilans energetyczny przejazdu.

    Atrybuty - momentalne (tablice, długość N):
        P_kolo:        Moc na kole napędowym [W] (≥0 w trakcji, ≤0 w hamowaniu el.)
        P_pant_draw:   Moc pobierana z pantografu [W] (zawsze ≥0)
        P_pant_rec:    Moc zwracana do sieci [W] (zawsze ≥0)
        P_pant_net:    Moc netto = draw - rec [W] (może być ujemna)
        I_pant:        Prąd na pantografie [A] (wartość bezwzględna)
        F_brake_el:    Siła hamulca elektrycznego [N]
        F_brake_mech:  Siła hamulca mechanicznego [N]

    Atrybuty - skalary energii [J]:
        E_trakcja_kolo:   Energia mechaniczna na kole w fazach trakcyjnych
        E_trakcja_pant:   Energia na pantografie zużyta na trakcję (E_kolo/η_tr)
        E_aux:            Energia potrzeb własnych (P_aux · T)
        E_ham_el_kolo:    Energia hamowania elektrycznego (mierzona na kole)
        E_rec_pant:       Energia rzeczywiście zwrócona do sieci
        E_ham_mech:       Energia rozproszona w hamulcu mechanicznym
        E_pant_pobrana:   E_trakcja_pant + E_aux (suma poborów)
        E_pant_netto:     E_pant_pobrana - E_rec_pant (główna metryka)

    Atrybuty - metryki:
        E_jednostkowa:    [kWh/(100 km · t)] - standard porównawczy KDP
        P_pant_max:       Maksymalna chwilowa moc poboru [W]
        P_pant_avg:       Średnia moc poboru w czasie przejazdu [W]
        I_pant_max:       Maksymalny prąd pantografu [A]
    """


    P_kolo :np .ndarray 
    P_pant_draw :np .ndarray 
    P_pant_rec :np .ndarray 
    P_pant_net :np .ndarray 
    I_pant :np .ndarray 
    F_brake_el :np .ndarray 
    F_brake_mech :np .ndarray 


    E_trakcja_kolo :float 
    E_trakcja_pant :float 
    E_aux :float 
    E_ham_el_kolo :float 
    E_rec_pant :float 
    E_ham_mech :float 
    E_pant_pobrana :float 
    E_pant_netto :float 


    E_jednostkowa :float 

    E_jednostkowa :float 
    E_per_km :float 
    E_per_seat_km :float 
    E_per_btkm :float 
    P_pant_max :float 
    P_pant_avg :float 
    I_pant_max :float 

    def summary (self ,p :Parameters )->str :
        """Czytelne podsumowanie bilansu energii."""
        return (
        f"=== Bilans energetyczny ===\n"
        f"  System zasilania       : {p .power_system } (η_grid = {p .eta_grid :.2f})\n"
        f"  --- Energie na pantografie ---\n"
        f"  E_trakcja (pant)       : {self .E_trakcja_pant *J_TO_KWH :>8.2f} kWh\n"
        f"  E_aux                  : {self .E_aux *J_TO_KWH :>8.2f} kWh\n"
        f"  E_pant_pobrana (Σ)     : {self .E_pant_pobrana *J_TO_KWH :>8.2f} kWh\n"
        f"  E_rec_pant (zwrot)     : {self .E_rec_pant *J_TO_KWH :>8.2f} kWh\n"
        f"  E_pant_NETTO           : {self .E_pant_netto *J_TO_KWH :>8.2f} kWh  ⭐\n"
        f"  --- Energie na kole ---\n"
        f"  E_trakcja_kolo         : {self .E_trakcja_kolo *J_TO_KWH :>8.2f} kWh\n"
        f"  E_ham_el_kolo          : {self .E_ham_el_kolo *J_TO_KWH :>8.2f} kWh\n"
        f"  E_ham_mech             : {self .E_ham_mech *J_TO_KWH :>8.2f} kWh\n"
        f"  --- Metryki ---\n"
        f"  E_per_km               : {self .E_per_km :>8.2f} kWh/km          (typ. KDP: 20-30)\n"
        f"  E_per_seat_km          : {self .E_per_seat_km :>8.2f} Wh/(seat·km)    (typ. KDP: 30-60)\n"
        f"  E_jednostkowa          : {self .E_jednostkowa :>8.2f} kWh/(100km·t)   (dawniej)\n"
        f"  P_pant_max             : {self .P_pant_max /1e6 :>8.2f} MW\n"
        f"  P_pant_avg             : {self .P_pant_avg /1e6 :>8.2f} MW\n"
        f"  I_pant_max             : {self .I_pant_max :>8.1f} A  (limit TSI: {p .I_grid_limit :.0f} A)\n"
        )







def compute_energy (sim :SimulationProfile ,p :Parameters )->EnergyResults :
    """
    Wylicza bilans energetyczny z gotowego profilu symulacji.

    Args:
        sim: Profil dynamiczny z run_simulation().
        p: Parametry symulacji.

    Returns:
        EnergyResults zawierający tablice i całkowite energie.
    """
    N =len (sim .x )
    v =sim .v 
    phase =sim .phase 


    mask_traction =(phase ==1 )|(phase ==2 )
    mask_coast =phase ==3 
    mask_brake =phase ==4 


    P_kolo =np .zeros (N ,dtype =np .float64 )
    P_pant_draw =np .zeros (N ,dtype =np .float64 )
    P_pant_rec =np .zeros (N ,dtype =np .float64 )
    F_brake_el =np .zeros (N ,dtype =np .float64 )
    F_brake_mech =np .zeros (N ,dtype =np .float64 )


    P_kolo [mask_traction ]=sim .F_tr [mask_traction ]*v [mask_traction ]
    P_pant_draw [mask_traction ]=P_kolo [mask_traction ]/p .eta_tr_effective +p .P_aux 


    P_pant_draw [mask_coast ]=p .P_aux 



    F_el_max =_F_brake_max_electric_vec (v ,p )
    F_brake_el_full =np .minimum (
    sim .F_brake ,F_el_max 
    )
    F_brake_mech_full =sim .F_brake -F_brake_el_full 

    F_brake_el [mask_brake ]=F_brake_el_full [mask_brake ]
    F_brake_mech [mask_brake ]=F_brake_mech_full [mask_brake ]


    P_kolo [mask_brake ]=-F_brake_el [mask_brake ]*v [mask_brake ]

    P_pant_rec [mask_brake ]=(
    F_brake_el [mask_brake ]*v [mask_brake ]*p .eta_rec_eff 
    )

    P_pant_draw [mask_brake ]=p .P_aux 


    P_pant_net =P_pant_draw -P_pant_rec 


    I_pant =np .abs (P_pant_net )/p .U_grid 



    dt =np .diff (sim .t )


    P_trakcja_kolo =np .where (np .isin (sim .phase ,[1 ,2 ]),P_kolo ,0.0 )
    P_trakcja_pant =np .where (np .isin (sim .phase ,[1 ,2 ]),P_pant_draw -p .P_aux ,0.0 )


    E_trakcja_kolo =float (
    np .sum (0.5 *(P_trakcja_kolo [:-1 ]+P_trakcja_kolo [1 :])*dt )
    )
    E_trakcja_pant =float (
    np .sum (0.5 *(P_trakcja_pant [:-1 ]+P_trakcja_pant [1 :])*dt )
    )


    E_aux =float (p .P_aux *sim .t [-1 ])


    P_ham_el_kolo =F_brake_el *sim .v 
    E_ham_el_kolo =float (np .sum (0.5 *(P_ham_el_kolo [:-1 ]+P_ham_el_kolo [1 :])*dt ))


    P_ham_mech =F_brake_mech *sim .v 
    E_ham_mech =float (np .sum (0.5 *(P_ham_mech [:-1 ]+P_ham_mech [1 :])*dt ))


    E_rec_pant =float (np .sum (0.5 *(P_pant_rec [:-1 ]+P_pant_rec [1 :])*dt ))


    E_pant_pobrana =E_trakcja_pant +E_aux 
    E_pant_netto =E_pant_pobrana -E_rec_pant 


    L_km =sim .x [-1 ]/1000.0 
    m_t =p .m /1000.0 
    n_seats =500 
    E_netto_kWh =E_pant_netto *J_TO_KWH 

    E_jednostkowa =E_netto_kWh /(L_km *m_t )*100.0 
    E_per_km =E_netto_kWh /L_km 
    E_per_seat_km =E_netto_kWh *1000.0 /(L_km *n_seats )
    E_per_btkm =E_netto_kWh *1000.0 /(L_km *m_t )

    P_pant_max =float (np .max (P_pant_draw ))
    P_pant_avg =E_pant_netto /sim .t [-1 ]
    I_pant_max =float (np .max (I_pant ))

    return EnergyResults (
    P_kolo =P_kolo ,
    P_pant_draw =P_pant_draw ,
    P_pant_rec =P_pant_rec ,
    P_pant_net =P_pant_net ,
    I_pant =I_pant ,
    F_brake_el =F_brake_el ,
    F_brake_mech =F_brake_mech ,
    E_trakcja_kolo =E_trakcja_kolo ,
    E_trakcja_pant =E_trakcja_pant ,
    E_aux =E_aux ,
    E_ham_el_kolo =E_ham_el_kolo ,
    E_rec_pant =E_rec_pant ,
    E_ham_mech =E_ham_mech ,
    E_pant_pobrana =E_pant_pobrana ,
    E_pant_netto =E_pant_netto ,
    E_jednostkowa =E_jednostkowa ,
    E_per_km =E_per_km ,
    E_per_seat_km =E_per_seat_km ,
    E_per_btkm =E_per_btkm ,
    P_pant_max =P_pant_max ,
    P_pant_avg =P_pant_avg ,
    I_pant_max =I_pant_max ,
    )






if __name__ =="__main__":
    from simulation import run_simulation 

    p =Parameters .base ()
    print (p .summary ())
    print ()

    print (">>> Uruchamiam symulację...")
    sim =run_simulation (p )
    print (f"  Czas przejazdu: {sim .T_total /60 :.2f} min")
    print ()

    print (">>> Obliczam bilans energii...")
    energy =compute_energy (sim ,p )
    print ()
    print (energy .summary (p ))


    print ()
    print (">>> Porównanie: ten sam przejazd, ale system DC")
    p_dc =p .with_changes (power_system ="DC",P_nom =6e6 )
    sim_dc =run_simulation (p_dc )
    energy_dc =compute_energy (sim_dc ,p_dc )
    print (energy_dc .summary (p_dc ))

    print ()
    print ("--- PORÓWNANIE ---")
    diff =(energy_dc .E_pant_netto -energy .E_pant_netto )*J_TO_KWH 
    print (
    f"  AC: {energy .E_pant_netto *J_TO_KWH :.1f} kWh   "
    f"DC: {energy_dc .E_pant_netto *J_TO_KWH :.1f} kWh   "
    f"Różnica: {diff :+.1f} kWh ({diff /(energy .E_pant_netto *J_TO_KWH )*100 :+.1f}%)"
    )
