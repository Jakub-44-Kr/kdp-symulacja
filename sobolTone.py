from __future__ import annotations 

import os 
import sys 
import time 
from pathlib import Path 

import numpy as np 
import pandas as pd 

from parameters import OUTPUT_DIR ,Parameters 


WH_PER_KWH =1e6 
M_BASE =float (Parameters .base ().m )





PARAM_MATH ={
"v_max":"v_{max}",
"m":"m",
"P_nom":"P",
"gradient":"i",
"L":"L",
}
COLOR_AC ="#1f77b4"
COLOR_DC ="#d62728"


def _apply_style ()->None :
    import matplotlib .pyplot as plt 

    plt .rcParams .update (
    {
    "font.family":"serif",
    "font.size":11 ,
    "axes.titlesize":12 ,
    "axes.labelsize":11 ,
    "xtick.labelsize":10 ,
    "ytick.labelsize":10 ,
    "legend.fontsize":9 ,
    "axes.grid":True ,
    "grid.alpha":0.3 ,
    "grid.linestyle":"--",
    "axes.axisbelow":True ,
    "axes.spines.top":False ,
    "axes.spines.right":False ,
    "savefig.dpi":300 ,
    "savefig.bbox":"tight",
    }
    )







def transform_oat_to_btkm (
elasticity_csv :Path |None =None ,
save_dir :Path =OUTPUT_DIR ,
)->pd .DataFrame :
    """
    Przelicza elastyczności OAT z metryki kWh/km na Wh/(b·tkm), operując
    wyłącznie na istniejącym pliku sensitivity_elasticity.csv (bez symulacji).

    Dla parametrów ≠ masa: masa pozostaje na poziomie bazowym (stały dzielnik),
    więc względna zmiana wyjścia i elastyczności są identyczne jak dla kWh/km;
    przeliczane są tylko kolumny energii (× 1e6 / m_base).
    Dla masy: dzielnik zmienia się wraz z perturbacją (m_base, m_plus, m_minus),
    więc elastyczność jest przeliczana od nowa z wartości energii w Wh/btkm.
    Krok względny perturbacji jest odzyskiwany z definicji elastyczności km
    (dla masy S_plus ≠ 0), dzięki czemu metoda jest niezależna od wartości delta.
    """
    if elasticity_csv is None :
        elasticity_csv =save_dir /"sensitivity_elasticity.csv"
    if not Path (elasticity_csv ).exists ():
        raise FileNotFoundError (
        f"Brak {elasticity_csv }. Uruchom najpierw sensitivity.py "
        f"(generuje sensitivity_elasticity.csv)."
        )

    df =pd .read_csv (elasticity_csv )
    rows =[]
    for _ ,r in df .iterrows ():
        p =r ["parameter"]
        Eb_km ,Ep_km ,Em_km =(
        float (r ["E_base_per_km"]),
        float (r ["E_plus_per_km"]),
        float (r ["E_minus_per_km"]),
        )
        if p =="m":

            delta =(Ep_km /Eb_km -1.0 )/float (r ["S_plus"])
            m_b ,m_p ,m_m =M_BASE ,M_BASE *(1 +delta ),M_BASE *(1 -delta )
            Eb =Eb_km *WH_PER_KWH /m_b 
            Ep =Ep_km *WH_PER_KWH /m_p 
            Em =Em_km *WH_PER_KWH /m_m 
            S_plus =(Ep /Eb -1.0 )/delta 
            S_minus =(Em /Eb -1.0 )/(-delta )
            S_avg =0.5 *(S_plus +S_minus )
            asymmetry =abs (S_plus -S_minus )
        else :

            factor =WH_PER_KWH /M_BASE 
            Eb ,Ep ,Em =Eb_km *factor ,Ep_km *factor ,Em_km *factor 
            S_plus ,S_minus =float (r ["S_plus"]),float (r ["S_minus"])
            S_avg =float (r ["S_avg"])
            asymmetry =float (r ["asymmetry"])

        rows .append (
        {
        "parameter":p ,
        "system":r ["system"],
        "S_plus":S_plus ,
        "S_minus":S_minus ,
        "S_avg":S_avg ,
        "asymmetry":asymmetry ,
        "E_base_per_btkm":Eb ,
        "E_plus_per_btkm":Ep ,
        "E_minus_per_btkm":Em ,
        }
        )

    out =pd .DataFrame (rows )


    merged =out .merge (
    df [["parameter","system","S_avg"]].rename (columns ={"S_avg":"S_avg_km"}),
    on =["parameter","system"],
    )
    mask =merged ["parameter"]!="m"
    max_dev =float (
    (merged .loc [mask ,"S_avg"]-merged .loc [mask ,"S_avg_km"]).abs ().max ()
    )
    assert max_dev <1e-4 ,(
    f"Niespójność: wiersze ≠ masa różnią się od km (Δ={max_dev :.2e})"
    )

    save_dir .mkdir (parents =True ,exist_ok =True )
    path =save_dir /"sensitivity_elasticity_btkm.csv"
    out .to_csv (path ,index =False ,float_format ="%.6g")


    print ("  [OAT] sensitivity_elasticity_btkm.csv zapisany.")
    print (
    "        Kontrola: wiersze ≠ masa identyczne jak km (max Δ S_avg = "
    f"{max_dev :.1e}). ✓"
    )
    for sysname in ("AC","DC"):
        rk =df [(df .parameter =="m")&(df .system ==sysname )]["S_avg"].values 
        rb =out [(out .parameter =="m")&(out .system ==sysname )]["S_avg"].values 
        if len (rk )and len (rb ):
            print (
            f"        masa {sysname }: S_avg km = {rk [0 ]:+.3f}  →  "
            f"btkm = {rb [0 ]:+.3f}  (odwrócenie znaku)"
            )
    return out 







def _load_or_compute_Y (
system :str ,N :int ,seed :int ,n_workers :int |None 
)->tuple [np .ndarray ,np .ndarray ]:
    """
    Zwraca (Y, param_values) dla wariantu regen=True przy zadanym N.
    Korzysta z cache sobol_raw_{system}.npz, jeśli istnieje i zgadza się
    (N, seed). W przeciwnym razie liczy raz i zapisuje cache.
    """
    from SALib .sample import sobol as sobol_sample 

    from sobol import build_problem ,evaluate_samples 

    problem =build_problem (system )
    cache =OUTPUT_DIR /f"sobol_raw_{system }.npz"

    if cache .exists ():
        d =np .load (cache ,allow_pickle =False )
        if int (d ["N"])==N and int (d ["seed"])==seed :
            print (
            f"    [{system }] cache trafiony ({cache .name }, N={N }) — bez symulacji."
            )
            return d ["Y"],d ["param_values"]
        print (f"    [{system }] cache niezgodny (N={int (d ['N'])}≠{N }) — przeliczanie.")

    param_values =sobol_sample .sample (problem ,N ,calc_second_order =True ,seed =seed )
    print (f"    [{system }] N={N } → {len (param_values )} uruchomień (regen=True)...")
    t0 =time .perf_counter ()
    Y =evaluate_samples (param_values ,system ,regen =True ,n_workers =n_workers )
    print (f"    [{system }] ewaluacja: {time .perf_counter ()-t0 :.1f} s")

    np .savez (
    cache ,
    Y =Y ,
    param_values =param_values ,
    N =np .int64 (N ),
    seed =np .int64 (seed ),
    )
    print (f"    [{system }] cache zapisany: {cache .name }")
    return Y ,param_values 


def sobol_btkm_for_system (
system :str ,N :int =1024 ,seed :int =42 ,n_workers :int |None =None 
)->dict :
    """
    Indeksy Sobola dla funkcji celu Wh/(b·tkm) = E_per_km · 1e6 / m.
    Ta sama macierz próbek i ten sam wektor Y (regen=True) co analiza km —
    zmienia się wyłącznie normalizacja wielkości wyjściowej.
    """
    from SALib .analyze import sobol as sobol_analyze 

    from sobol import build_problem 

    problem =build_problem (system )
    Y ,param_values =_load_or_compute_Y (system ,N ,seed ,n_workers )

    mass =param_values [
    :,1 
    ]
    Y_btkm =Y *WH_PER_KWH /mass 

    Si =sobol_analyze .analyze (problem ,Y_btkm ,calc_second_order =True ,seed =seed )

    return {
    "system":system ,
    "N":N ,
    "n_runs":len (param_values ),
    "names":problem ["names"],
    "S1":Si ["S1"],
    "S1_conf":Si ["S1_conf"],
    "ST":Si ["ST"],
    "ST_conf":Si ["ST_conf"],
    "S2":Si ["S2"],
    "S2_conf":Si ["S2_conf"],
    "Y_mean":float (np .mean (Y_btkm )),
    "Y_std":float (np .std (Y_btkm )),
    }


def _report_sobol (res :dict )->None :
    print ()
    print (
    f"  [SOBOL {res ['system']}]  Wh/btkm: średnia = {res ['Y_mean']:.2f}, "
    f"odch.std = {res ['Y_std']:.2f}  (N={res ['N']}, {res ['n_runs']} przebiegów)"
    )
    names =res ["names"]
    order =np .argsort (res ["ST"])[::-1 ]
    print (f"    {'parametr':>10} {'S_i':>9} {'S_Ti':>9}")
    for i in order :
        print (f"    {names [i ]:>10} {res ['S1'][i ]:>9.4f} {res ['ST'][i ]:>9.4f}")







def plot_sobol_btkm (res_ac :dict ,res_dc :dict ,save_dir :Path =OUTPUT_DIR )->Path :
    """
    Dwupanelowy wykres kolumnowy dla metryki Wh/(b·tkm):
      - panel górny:  S_i (słupek pełny) i S_Ti (słupek jaśniejszy), AC vs DC,
      - panel dolny:  interakcje S_ij dla wszystkich 10 par, AC vs DC,
                      posortowane malejąco wg max(|S_ij^AC|, |S_ij^DC|).
    """
    import matplotlib .pyplot as plt 
    from matplotlib .patches import Patch 

    _apply_style ()
    names =res_ac ["names"]
    n =len (names )
    w =0.38 


    importance =np .maximum (res_ac ["ST"],res_dc ["ST"])
    order =list (np .argsort (importance )[::-1 ])
    xlabels =[f"${PARAM_MATH [names [i ]]}$"for i in order ]
    x =np .arange (n )

    fig ,(ax1 ,ax2 )=plt .subplots (2 ,1 ,figsize =(10.5 ,9 ))


    for res ,color ,off in ((res_ac ,COLOR_AC ,-w /2 ),(res_dc ,COLOR_DC ,+w /2 )):
        S1 =np .asarray (res ["S1"])[order ]
        ST =np .asarray (res ["ST"])[order ]
        S1c =np .asarray (res ["S1_conf"])[order ]
        STc =np .asarray (res ["ST_conf"])[order ]
        ax1 .bar (
        x +off ,
        ST ,
        w ,
        color =color ,
        alpha =0.32 ,
        yerr =STc ,
        capsize =2 ,
        error_kw =dict (lw =0.8 ,alpha =0.6 ),
        )
        ax1 .bar (
        x +off ,
        S1 ,
        w ,
        color =color ,
        alpha =0.95 ,
        yerr =S1c ,
        capsize =2 ,
        error_kw =dict (lw =0.8 ,alpha =0.6 ),
        )
    ax1 .axhline (0 ,color ="0.5",lw =0.6 )
    ax1 .set_xticks (x )
    ax1 .set_xticklabels (xlabels )
    ax1 .set_ylabel (r"wartość wskaźnika [$-$]")
    ax1 .set_title (
    r"Indeksy Sobola — $S_i$ (słupek ciemny) i $S_{Ti}$ (jaśniejszy), "
    r"Wh/(b$\cdot$tkm)"
    )
    handles =[
    Patch (facecolor =COLOR_AC ,alpha =0.95 ,label =r"AC  $S_i$"),
    Patch (facecolor =COLOR_AC ,alpha =0.32 ,label =r"AC  $S_{Ti}$"),
    Patch (facecolor =COLOR_DC ,alpha =0.95 ,label =r"DC  $S_i$"),
    Patch (facecolor =COLOR_DC ,alpha =0.32 ,label =r"DC  $S_{Ti}$"),
    ]
    ax1 .legend (handles =handles ,ncol =2 ,loc ="upper right")


    pairs =[(i ,j )for i in range (n )for j in range (i +1 ,n )]
    s2_ac =np .array ([res_ac ["S2"][i ,j ]for i ,j in pairs ])
    s2_dc =np .array ([res_dc ["S2"][i ,j ]for i ,j in pairs ])
    s2c_ac =np .array ([res_ac ["S2_conf"][i ,j ]for i ,j in pairs ])
    s2c_dc =np .array ([res_dc ["S2_conf"][i ,j ]for i ,j in pairs ])
    key =np .maximum (np .abs (s2_ac ),np .abs (s2_dc ))
    po =np .argsort (key )[::-1 ]

    plabels =[
    f"${PARAM_MATH [names [pairs [k ][0 ]]]}\\times {PARAM_MATH [names [pairs [k ][1 ]]]}$"
    for k in po 
    ]
    xx =np .arange (len (pairs ))
    ax2 .bar (
    xx -w /2 ,
    s2_ac [po ],
    w ,
    color =COLOR_AC ,
    alpha =0.9 ,
    label ="AC",
    yerr =s2c_ac [po ],
    capsize =2 ,
    error_kw =dict (lw =0.8 ,alpha =0.5 ),
    )
    ax2 .bar (
    xx +w /2 ,
    s2_dc [po ],
    w ,
    color =COLOR_DC ,
    alpha =0.9 ,
    label ="DC",
    yerr =s2c_dc [po ],
    capsize =2 ,
    error_kw =dict (lw =0.8 ,alpha =0.5 ),
    )
    ax2 .axhline (0 ,color ="0.5",lw =0.6 )
    ax2 .set_xticks (xx )
    ax2 .set_xticklabels (plabels ,rotation =30 ,ha ="right")
    ax2 .set_ylabel (r"$S_{ij}$ [$-$]")
    ax2 .set_title (r"Interakcje drugiego rzędu $S_{ij}$ — Wh/(b$\cdot$tkm)")
    ax2 .legend (loc ="upper right")

    fig .tight_layout ()
    path =save_dir /"sobol_btkm_S1_S2.png"
    fig .savefig (path )
    plt .close (fig )
    print (f"  [WYKRES] {path .name } zapisany (300 dpi).")
    return path 







def main ()->int :
    args =sys .argv [1 :]
    oat_only ="--oat-only"in args 
    N_vals =[int (a )for a in args if a .isdigit ()]
    N =N_vals [0 ]if N_vals else 1024 
    n_cpu =os .cpu_count ()or 4 

    print ("="*78 )
    print (
    "ANALIZA WRAŻLIWOŚCI — WARIANT Wh/(b·tkm)  (rekuperacja 100%: AC 0,80 / DC 0,15)"
    )
    print ("="*78 )


    print ("\n[1/3] OAT — przekształcenie sensitivity_elasticity.csv → _btkm")
    transform_oat_to_btkm ()

    if oat_only :
        print ("\n--oat-only: pominięto Sobola i wykres.")
        print ("="*78 )
        return 0 


    print (f"\n[2/3] SOBOL — indeksy dla Wh/btkm  (N={N }, {max (1 ,n_cpu -1 )} procesów)")
    if N ==1024 :
        print (
        "      (N=1024 = wartość utrwalona w sobol_indices_*.csv; "
        "dla zgodności z rysunkami km na 2048 użyj: python sobolTone.py 2048)"
        )
    results ={}
    for system in ("AC","DC"):
        res =sobol_btkm_for_system (system ,N =N )
        _report_sobol (res )
        from sobol import export_sobol_indices 

        export_sobol_indices (res ,suffix ="_btkm")
        print (
        f"    [{system }] zapisano: sobol_indices_{system }_btkm.csv, "
        f"sobol_interactions_{system }_btkm.csv"
        )
        results [system ]=res 

    print ("\n[3/3] WYKRES — sobol_btkm_S1_S2.png")
    plot_sobol_btkm (results ["AC"],results ["DC"])

    print ("\n"+"="*78 )
    print ("Gotowe. Nowe pliki (sufiks _btkm) — istniejące wyniki nietknięte.")
    print ("="*78 )
    return 0 


if __name__ =="__main__":
    raise SystemExit (main ())
