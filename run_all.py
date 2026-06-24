
"""
run_all.py — Sekwencyjny launcher pełnego pipeline'u symulacji KDP.

Uruchamia po kolei skrypty analizy z paskiem postępu, pomiarem czasu
i logiem per-krok (outputs/_pipeline_logs/). Przy błędzie zatrzymuje
pipeline i pokazuje ogon logu, żeby od razu było widać przyczynę.

Użycie:
    python run_all.py                       # cały pipeline (kroki 1–8)
    python run_all.py --verbose             # pokazuj output skryptów na żywo
    python run_all.py --from sobol.py       # zacznij od wskazanego kroku
    python run_all.py --skip plots_param_influence.py,sobol_plots.py
    python run_all.py --only sobol.py,sobol_plots.py
    python run_all.py --keep-going          # nie zatrzymuj się na błędzie
    python run_all.py --list                # tylko wypisz kroki i wyjdź

UWAGA: nie nazywaj tego pliku main.py — w projekcie main.py to scenariusz bazowy.

Autor: Jakub Król, PW WE, 2026
"""

from __future__ import annotations 

import argparse 
import shutil 
import subprocess 
import sys 
import time 
from pathlib import Path 

ROOT =Path (__file__ ).parent .resolve ()
LOG_DIR =ROOT /"outputs"/"_pipeline_logs"


STEPS :list [tuple [str ,str ]]=[
("validation.py","sanity-check modelu (7/7)"),
("main.py","scenariusz bazowy + wykresy 01–08"),
("sensitivity.py","sweep OAT (sensitivity_*.csv)"),
("plots_param_influence.py","wykresy wpływu parametrów (influence_*)"),
("sobol.py","wskaźniki Sobola S1/ST/S2"),
("sobol_convergence.py","zbieżność Sobola"),
("sobol_plots.py","wykresy Sobola"),
("routes_analysis.py","analiza tras (trasy_*)"),
]

USE_TTY =sys .stdout .isatty ()


def c (code :str ,text :str )->str :
    """Koloruj tekst (ANSI) tylko gdy wyjście to terminal."""
    return f"\033[{code }m{text }\033[0m"if USE_TTY else text 


def fmt_dur (seconds :float )->str :
    s =int (round (seconds ))
    h ,rem =divmod (s ,3600 )
    m ,s =divmod (rem ,60 )
    return f"{h :d}:{m :02d}:{s :02d}"if h else f"{m :02d}:{s :02d}"


def bar (done :int ,total :int ,width :int =22 )->str :
    filled =int (width *done /total )if total else 0 
    return "["+"█"*filled +"░"*(width -filled )+"]"


def term_width ()->int :
    return shutil .get_terminal_size ((80 ,20 )).columns 


def run_step (idx :int ,total :int ,script :str ,desc :str ,verbose :bool ):
    """Uruchamia jeden skrypt. Zwraca (status, elapsed), status∈{ok,fail,skip}."""
    path =ROOT /script 
    head =f"{c ('1;36',f'[{idx }/{total }]')} {c ('1',script )} — {desc }"


    if not path .exists ()or path .stat ().st_size ==0 :
        print (f"▶ {head }")
        print (f"  {c ('33','⚠ pominięto')} — plik nie istnieje lub jest pusty\n")
        return "skip",0.0 

    print (f"▶ {head }")
    start =time .perf_counter ()

    if verbose :
        rc =subprocess .run ([sys .executable ,script ],cwd =ROOT ).returncode 
        elapsed =time .perf_counter ()-start 
        log_path =None 
    else :
        LOG_DIR .mkdir (parents =True ,exist_ok =True )
        log_path =LOG_DIR /f"{path .stem }.log"
        spin ="|/-\\"
        i =0 
        with open (log_path ,"w",encoding ="utf-8")as logf :
            proc =subprocess .Popen (
            [sys .executable ,script ],
            cwd =ROOT ,
            stdout =logf ,
            stderr =subprocess .STDOUT ,
            )
            while proc .poll ()is None :
                if USE_TTY :
                    elapsed =time .perf_counter ()-start 
                    line =f"  {bar (idx -1 ,total )} {spin [i %4 ]} {fmt_dur (elapsed )}"
                    sys .stdout .write ("\r"+line [:term_width ()-1 ])
                    sys .stdout .flush ()
                i +=1 
                time .sleep (0.25 )
        rc =proc .returncode 
        elapsed =time .perf_counter ()-start 
        if USE_TTY :
            sys .stdout .write ("\r"+" "*(term_width ()-1 )+"\r")

    if rc ==0 :
        tail =""if log_path is None else f"  ·  log: {log_path .relative_to (ROOT )}"
        print (f"  {c ('1;32','✓')} ukończono w {fmt_dur (elapsed )}{tail }\n")
        return "ok",elapsed 

    print (f"  {c ('1;31','✗ BŁĄD')} (kod {rc }) po {fmt_dur (elapsed )}")
    if log_path is not None and log_path .exists ():
        lines =log_path .read_text (encoding ="utf-8",errors ="replace").splitlines ()[
        -30 :
        ]
        print (c ("31","  ─ ostatnie linie logu "+"─"*40 ))
        for ln in lines :
            print ("  "+ln )
        print (c ("31","  "+"─"*63 )+f"\n  pełny log: {log_path }\n")
    return "fail",elapsed 


def main ()->int :
    ap =argparse .ArgumentParser (description ="Launcher pipeline'u symulacji KDP.")
    ap .add_argument (
    "--verbose",action ="store_true",help ="pokazuj output skryptów na żywo"
    )
    ap .add_argument (
    "--from",dest ="start_from",metavar ="SCRIPT",help ="zacznij od tego kroku"
    )
    ap .add_argument (
    "--skip",metavar ="A,B",help ="pomiń wskazane skrypty (po przecinku)"
    )
    ap .add_argument ("--only",metavar ="A,B",help ="uruchom tylko wskazane skrypty")
    ap .add_argument (
    "--keep-going",action ="store_true",help ="nie zatrzymuj się na błędzie"
    )
    ap .add_argument ("--list",action ="store_true",help ="wypisz kroki i zakończ")
    args =ap .parse_args ()

    steps =list (STEPS )

    if args .list :
        print ("Kroki pipeline'u:")
        for i ,(s ,d )in enumerate (steps ,1 ):
            print (f"  {i }. {s :<26} {d }")
        return 0 

    skip ={x .strip ()for x in args .skip .split (",")}if args .skip else set ()
    only ={x .strip ()for x in args .only .split (",")}if args .only else None 
    if only :
        steps =[st for st in steps if st [0 ]in only ]
    if args .start_from :
        names =[s for s ,_ in steps ]
        if args .start_from not in names :
            print (
            f"✗ --from: nie znam kroku '{args .start_from }'. Dostępne: {', '.join (names )}"
            )
            return 2 
        steps =steps [names .index (args .start_from ):]
    steps =[st for st in steps if st [0 ]not in skip ]

    if not steps :
        print ("Brak kroków do uruchomienia (sprawdź --only/--skip/--from).")
        return 0 

    total =len (steps )
    print (c ("1",f"\n═══ Pipeline KDP — {total } kroków ")+c ("1","═"*28 ))
    print (f"interpreter: {sys .executable }")
    print (f"katalog:     {ROOT }")
    if not args .verbose :
        print (f"logi:        {LOG_DIR .relative_to (ROOT )}/")
    print (c ("90","─"*63 )+"\n")

    t0 =time .perf_counter ()
    results :list [tuple [str ,str ,float ]]=[]
    aborted =False 

    for idx ,(script ,desc )in enumerate (steps ,1 ):
        status ,elapsed =run_step (idx ,total ,script ,desc ,args .verbose )
        results .append ((script ,status ,elapsed ))
        if status =="fail"and not args .keep_going :
            aborted =True 
            break 

    total_elapsed =time .perf_counter ()-t0 

    print (c ("1","═══ Podsumowanie ")+c ("1","═"*46 ))
    mark ={"ok":c ("32","✓"),"fail":c ("31","✗"),"skip":c ("33","⚠")}
    for script ,status ,elapsed in results :
        print (f"  {mark .get (status ,'?')} {script :<26} {fmt_dur (elapsed ):>8}  {status }")
    n_ok =sum (1 for _ ,s ,_ in results if s =="ok")
    print (c ("90","─"*63 ))
    print (f"  {n_ok }/{total } ukończone  ·  łączny czas {fmt_dur (total_elapsed )}")

    if aborted :
        print (c ("1;31","\n⛔ Pipeline zatrzymany na błędzie. Napraw i wznów np.:"))
        print (f"   python {Path (__file__ ).name } --from {results [-1 ][0 ]}\n")
        return 1 

    print (c ("1;32","\n✓ Gotowe.")+f" Pliki wynikowe w: {ROOT /'outputs'}")
    print ("  (podmień nimi kopie danych w projekcie)\n")
    return 0 


if __name__ =="__main__":
    try :
        sys .exit (main ())
    except KeyboardInterrupt :
        print ("\n\n⛔ Przerwano przez użytkownika (Ctrl-C).")
        sys .exit (130 )
