# Symulacja zużycia energii pociągu dużej prędkości

Praca dyplomowa magisterska — Politechnika Warszawska, Wydział Elektryczny.

**Autor:** inż. Jakub Król
**Promotor:** prof. dr hab. inż. Adam Szeląg
**Temat:** Wpływ parametrów ruchu pociągów dużej prędkości na zużycie energii

## Opis

Model symulacyjny przejazdu pociągu KDP na odcinku eksploatacyjnym
pomiędzy dwoma postojami, służący jako podstawa analizy wrażliwości
zużycia energii trakcyjnej na parametry: prędkość eksploatacyjną,
masę składu, moc znamionową, pochylenie trasy, długość odcinka
oraz system zasilania (3 kV DC / 2×25 kV AC).

## Wymagania

- Python 3.13+
- Biblioteki w `requirements.txt`

## Uruchomienie

    py -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python main.py

## Struktura

- `parameters.py` — wszystkie parametry modelu (zmienne + stałe)
- `physics.py` — funkcje fizyczne (charakterystyka trakcyjna, opory Davisa, hamowanie)
- `simulation.py` — silnik symulacji (forward + backward Euler)
- `energy.py` — bilans energii, moc i prąd na pantografie
- `results.py` — struktura wyników, eksport CSV/JSON
- `plotting.py` — wykresy
- `validation.py` — sanity checks
- `main.py` — punkt wejścia
