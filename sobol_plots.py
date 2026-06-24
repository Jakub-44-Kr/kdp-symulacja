"""
sobol_plots.py — Finalne wykresy globalnej analizy wrażliwości Sobola.

Generuje zestaw rysunków do rozdziału 7 pracy magisterskiej:

  1. Bar charty S_i vs S_T_i (osobno dla AC i DC)
  2. Bar chart porównawczy AC vs DC dla S_T_i
  3. Heatmapy interakcji S2 - trzy warianty:
     a) klasyczna kwadratowa 5x5
     b) trójkątna górna (oszczędność miejsca)
     c) bar chart top-3 najsilniejszych par

Używa wyników z N=2048 (świeży run) - kompromis między dokładnością a czasem.
Wykresy zapisywane w outputs/ jako PNG (300 dpi, gotowe do pracy).

Autor: Jakub Król, PW WE, 2026
"""

from __future__ import annotations 

import os 
import sys 
import time 
from pathlib import Path 

import matplotlib .pyplot as plt 
import numpy as np 

from parameters import OUTPUT_DIR 
from sobol import export_sobol_indices ,run_sobol_for_system 





plt .rcParams .update (
{
"font.family":"serif",
"font.size":11 ,
"axes.titlesize":12 ,
"axes.labelsize":11 ,
"xtick.labelsize":10 ,
"ytick.labelsize":10 ,
"legend.fontsize":10 ,
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

PARAM_LABELS ={
"v_max":"$v_{max}$",
"m":"$m$",
"P_nom":"$P$",
"gradient":"$i$",
"L":"$L$",
}

COLOR_AC ="#1f77b4"
COLOR_DC ="#d62728"
COLOR_S1 ="#4c72b0"
COLOR_ST ="#dd8452"


def _sort_by_ST (results :dict )->list [int ]:
    """Indeksy parametrów posortowane malejąco wg S_Ti (dla rankingu)."""
    return list (np .argsort (results ["ST"])[::-1 ])







def plot_indices_single_system (results :dict ,save_dir :Path =OUTPUT_DIR )->Path :
    """Bar chart S_i obok S_T_i dla jednego systemu, posortowane wg S_T_i."""
    system =results ["system"]
    names =results ["names"]
    order =_sort_by_ST (results )
    labels =[PARAM_LABELS [names [i ]]for i in order ]
    S1 =[results ["S1"][i ]for i in order ]
    ST =[results ["ST"][i ]for i in order ]

    fig ,ax =plt .subplots (figsize =(9 ,5 ))
    x =np .arange (len (labels ))
    w =0.4 

    bars1 =ax .bar (
    x -w /2 ,
    S1 ,
    w ,
    color =COLOR_S1 ,
    edgecolor ="black",
    linewidth =0.8 ,
    label ="$S_i$ (samodzielny)",
    alpha =0.9 ,
    )
    bars2 =ax .bar (
    x +w /2 ,
    ST ,
    w ,
    color =COLOR_ST ,
    edgecolor ="black",
    linewidth =0.8 ,
    label ="$S_{Ti}$ (całkowity)",
    alpha =0.9 ,
    )


    for bars ,vals in [(bars1 ,S1 ),(bars2 ,ST )]:
        for bar ,val in zip (bars ,vals ):
            ax .text (
            bar .get_x ()+bar .get_width ()/2 ,
            bar .get_height ()+0.015 ,
            f"{val :.3f}",
            ha ="center",
            fontsize =9 ,
            )

    ax .set_xticks (x )
    ax .set_xticklabels (labels ,fontsize =12 )
    ax .set_ylabel ("Indeks Sobola")
    ax .set_title (
    f"Globalna analiza wrażliwości — system {system }  "
    f"(N={results ['N']}, {results ['n_runs']} uruchomień)"
    )
    ax .set_ylim (0 ,max (max (ST )*1.15 ,0.1 ))
    ax .axhline (0 ,color ="black",linewidth =0.5 )
    ax .legend (loc ="upper right",framealpha =0.95 )

    path =save_dir /f"sobol_indices_{system }.png"
    fig .savefig (path )
    plt .close (fig )
    return path 







def plot_comparison_AC_DC (
results_AC :dict ,results_DC :dict ,save_dir :Path =OUTPUT_DIR 
)->Path :
    """
    Wykres porównawczy: dla każdego parametru S_T_i w AC obok S_T_i w DC.
    Posortowane wg max(AC, DC).
    """
    names =results_AC ["names"]
    n =len (names )
    ST_AC =results_AC ["ST"]
    ST_DC =results_DC ["ST"]


    max_st =np .maximum (ST_AC ,ST_DC )
    order =list (np .argsort (max_st )[::-1 ])
    labels =[PARAM_LABELS [names [i ]]for i in order ]
    ac =[ST_AC [i ]for i in order ]
    dc =[ST_DC [i ]for i in order ]

    fig ,ax =plt .subplots (figsize =(9 ,5 ))
    x =np .arange (n )
    w =0.4 
    ax .bar (
    x -w /2 ,
    ac ,
    w ,
    color =COLOR_AC ,
    edgecolor ="black",
    linewidth =0.8 ,
    label ="AC (2×25 kV)",
    alpha =0.9 ,
    )
    ax .bar (
    x +w /2 ,
    dc ,
    w ,
    color =COLOR_DC ,
    edgecolor ="black",
    linewidth =0.8 ,
    label ="DC (3 kV)",
    alpha =0.9 ,
    )


    for xi ,(a ,d )in enumerate (zip (ac ,dc )):
        ax .text (xi -w /2 ,a +0.02 ,f"{a :.3f}",ha ="center",fontsize =9 )
        ax .text (xi +w /2 ,d +0.02 ,f"{d :.3f}",ha ="center",fontsize =9 )

    ax .set_xticks (x )
    ax .set_xticklabels (labels ,fontsize =12 )
    ax .set_ylabel ("$S_{Ti}$ — indeks całkowity")
    ax .set_title (f"Porównanie wrażliwości — system AC vs DC  (N={results_AC ['N']})")
    ax .set_ylim (0 ,max (max (ac ),max (dc ))*1.18 )
    ax .axhline (0 ,color ="black",linewidth =0.5 )
    ax .legend (loc ="upper right",framealpha =0.95 )

    path =save_dir /"sobol_comparison_AC_vs_DC.png"
    fig .savefig (path )
    plt .close (fig )
    return path 







def plot_z_vs_bez (
results_z :dict ,results_bez :dict ,save_dir :Path =OUTPUT_DIR 
)->Path :
    """
    Porownanie indeksow Sobola: wariant z rekuperacja vs bez rekuperacji,
    dla jednego systemu. Dwa panele: S_i (pierwszego rzedu) oraz S_Ti (calkowity).
    Sluzy do pokazania, ze rekuperacja praktycznie nie zmienia rankingu czynnikow.
    """
    system =results_z ["system"]
    names =results_z ["names"]
    order =_sort_by_ST (results_z )
    labels =[PARAM_LABELS [names [i ]]for i in order ]
    x =np .arange (len (labels ))
    w =0.4 
    C_Z ,C_BEZ ="#1f5fa6","#9ecae1"

    fig ,(ax1 ,ax2 )=plt .subplots (1 ,2 ,figsize =(12 ,5 ))
    for ax ,key ,ttl in (
    (ax1 ,"S1","Indeks pierwszego rzędu $S_i$"),
    (ax2 ,"ST","Indeks całkowity $S_{Ti}$"),
    ):
        z =[results_z [key ][i ]for i in order ]
        b =[results_bez [key ][i ]for i in order ]
        ax .bar (
        x -w /2 ,
        z ,
        w ,
        color =C_Z ,
        edgecolor ="black",
        linewidth =0.8 ,
        label ="z rekuperacją",
        alpha =0.95 ,
        )
        ax .bar (
        x +w /2 ,
        b ,
        w ,
        color =C_BEZ ,
        edgecolor ="black",
        linewidth =0.8 ,
        label ="bez rekuperacji",
        alpha =0.95 ,
        )
        ax .set_xticks (x )
        ax .set_xticklabels (labels ,fontsize =12 )
        ax .set_ylabel ("Indeks Sobola")
        ax .set_title (ttl )
        top =max (max (z ),max (b ))
        ax .set_ylim (0 ,top *1.18 if top >0 else 0.1 )
        ax .axhline (0 ,color ="black",linewidth =0.5 )
        ax .legend (loc ="upper right",framealpha =0.95 )

    fig .suptitle (
    f"Wpływ rekuperacji na indeksy Sobola — system {system }  (N={results_z ['N']})",
    fontsize =13 ,
    )
    fig .tight_layout (rect =(0.0 ,0.0 ,1.0 ,0.96 ))
    path =save_dir /f"sobol_z_vs_bez_{system }.png"
    fig .savefig (path )
    plt .close (fig )
    return path 







def _draw_heatmap (
S2 :np .ndarray ,names :list [str ],title :str ,ax ,triangular :bool =False 
)->None :
    """Helper: rysuje heatmapę S2 na danej osi (kwadratową lub trójkątną)."""
    n =len (names )
    matrix =np .array (S2 ,dtype =np .float64 )

    if triangular :

        mask =np .tril (np .ones_like (matrix ,dtype =bool ),k =-1 )
        matrix_display =np .where (mask ,np .nan ,matrix )
    else :
        matrix_display =matrix 


    vmax =np .nanmax (np .abs (matrix_display ))
    if vmax <1e-6 :
        vmax =0.05 
    im =ax .imshow (matrix_display ,cmap ="YlOrRd",vmin =0 ,vmax =vmax ,aspect ="equal")


    for i in range (n ):
        for j in range (n ):
            val =matrix [i ,j ]
            if triangular and i >j :
                continue 
            if i ==j :

                ax .text (j ,i ,"—",ha ="center",va ="center",color ="black",fontsize =10 )
                continue 
            if not np .isnan (val ):
                color ="white"if val >vmax *0.5 else "black"
                ax .text (
                j ,
                i ,
                f"{val :.3f}",
                ha ="center",
                va ="center",
                color =color ,
                fontsize =9 ,
                )

    ax .set_xticks (range (n ))
    ax .set_yticks (range (n ))
    ax .set_xticklabels ([PARAM_LABELS [n_ ]for n_ in names ],fontsize =11 )
    ax .set_yticklabels ([PARAM_LABELS [n_ ]for n_ in names ],fontsize =11 )
    ax .set_title (title )
    plt .colorbar (im ,ax =ax ,fraction =0.046 ,pad =0.04 ,label ="$S_{2,ij}$")


def plot_heatmap_full (results :dict ,save_dir :Path =OUTPUT_DIR )->Path :
    """Wariant A: pełna kwadratowa heatmapa 5x5."""
    system =results ["system"]
    fig ,ax =plt .subplots (figsize =(7 ,6 ))
    _draw_heatmap (
    results ["S2"],
    results ["names"],
    f"Interakcje drugiego rzędu $S_{{2,ij}}$ — system {system }",
    ax ,
    triangular =False ,
    )
    path =save_dir /f"sobol_S2_heatmap_full_{system }.png"
    fig .savefig (path )
    plt .close (fig )
    return path 


def plot_heatmap_triangular (results :dict ,save_dir :Path =OUTPUT_DIR )->Path :
    """Wariant B: heatmapa trójkątna (tylko górny trójkąt, bez duplikacji)."""
    system =results ["system"]
    fig ,ax =plt .subplots (figsize =(7 ,6 ))
    _draw_heatmap (
    results ["S2"],
    results ["names"],
    f"Interakcje $S_{{2,ij}}$ (trójkąt górny) — system {system }",
    ax ,
    triangular =True ,
    )
    path =save_dir /f"sobol_S2_heatmap_triangular_{system }.png"
    fig .savefig (path )
    plt .close (fig )
    return path 


def plot_top_interactions (
results :dict ,top_n :int =3 ,save_dir :Path =OUTPUT_DIR 
)->Path :
    """Wariant C: bar chart top-N najsilniejszych par interakcji."""
    system =results ["system"]
    names =results ["names"]
    S2 =results ["S2"]
    S2_conf =results ["S2_conf"]
    n =len (names )


    pairs =[]
    for i in range (n ):
        for j in range (i +1 ,n ):
            val =S2 [i ,j ]
            err =S2_conf [i ,j ]
            if not np .isnan (val ):
                pairs .append ((names [i ],names [j ],val ,err ))


    pairs .sort (key =lambda t :abs (t [2 ]),reverse =True )
    top =pairs [:top_n ]

    labels =[f"{PARAM_LABELS [p [0 ]]} × {PARAM_LABELS [p [1 ]]}"for p in top ]
    vals =[p [2 ]for p in top ]
    errs =[p [3 ]for p in top ]

    fig ,ax =plt .subplots (figsize =(8 ,4.5 ))
    colors =["#d62728","#ff7f0e","#9467bd"][:top_n ]+["#7f7f7f"]*max (0 ,top_n -3 )
    bars =ax .bar (
    range (top_n ),
    vals ,
    color =colors ,
    edgecolor ="black",
    linewidth =0.8 ,
    alpha =0.9 ,
    )

    for bar ,val in zip (bars ,vals ):
        ax .text (
        bar .get_x ()+bar .get_width ()/2 ,
        bar .get_height ()+max (vals )*0.04 ,
        f"{val :.4f}",
        ha ="center",
        fontsize =10 ,
        fontweight ="bold",
        )

    ax .set_xticks (range (top_n ))
    ax .set_xticklabels (labels ,fontsize =11 )
    ax .set_ylabel ("$S_{2,ij}$ — interakcja drugiego rzędu")
    ax .set_title (
    f"Top {top_n } najsilniejszych interakcji — system {system }  (N={results ['N']})"
    )
    ax .set_ylim (0 ,max (vals )*1.25 if max (vals )>0 else 0.1 )
    ax .axhline (0 ,color ="black",linewidth =0.5 )

    path =save_dir /f"sobol_S2_top{top_n }_{system }.png"
    fig .savefig (path )
    plt .close (fig )
    return path 







def make_all_plots (
results_AC :dict ,results_DC :dict ,save_dir :Path =OUTPUT_DIR 
)->list [Path ]:
    """Generuje pełen zestaw wykresów Sobola do pracy magisterskiej."""
    paths =[]

    print ("  >>> Bar charty S_i vs S_Ti (osobno AC, DC)...")
    paths .append (plot_indices_single_system (results_AC ,save_dir ))
    paths .append (plot_indices_single_system (results_DC ,save_dir ))

    print ("  >>> Wykres porównawczy AC vs DC...")
    paths .append (plot_comparison_AC_DC (results_AC ,results_DC ,save_dir ))

    print ("  >>> Heatmapy interakcji (3 warianty na system)...")
    for results in (results_AC ,results_DC ):
        paths .append (plot_heatmap_full (results ,save_dir ))
        paths .append (plot_heatmap_triangular (results ,save_dir ))
        paths .append (plot_top_interactions (results ,top_n =3 ,save_dir =save_dir ))

    return paths 






if __name__ =="__main__":
    n_cpu =os .cpu_count ()or 4 


    N_FINAL =int (sys .argv [1 ])if len (sys .argv )>1 else 1024 

    print ("="*78 )
    print ("FINALNE WYKRESY SOBOLA — rozdział 7 (warianty: z / bez rekuperacji)")
    print (f"  N={N_FINAL }, procesory: {max (1 ,n_cpu -1 )}")
    print ("="*78 )
    print ()

    t0 =time .perf_counter ()


    results ={}
    for system in ("AC","DC"):
        print (f">>> Sobol {system } — z rekuperacją...")
        rz =run_sobol_for_system (system ,N =N_FINAL ,regen =True )
        print (f">>> Sobol {system } — bez rekuperacji...")
        rb =run_sobol_for_system (system ,N =N_FINAL ,regen =False )
        export_sobol_indices (rz ,suffix ="")
        export_sobol_indices (rb ,suffix ="_bez")
        results [system ]=(rz ,rb )

    results_AC =results ["AC"][0 ]
    results_DC =results ["DC"][0 ]


    print ("\n>>> Generuję wykresy...")
    paths =make_all_plots (results_AC ,results_DC )
    for system in ("AC","DC"):
        rz ,rb =results [system ]
        paths .append (plot_z_vs_bez (rz ,rb ))

    print ()
    print ("="*78 )
    print (f"WYGENEROWANO {len (paths )} WYKRESÓW")
    print ("="*78 )
    for p in paths :
        print (f"  - {p .name }")
    print ()
    print (f"✓ Łączny czas: {(time .perf_counter ()-t0 )/60 :.1f} min")
