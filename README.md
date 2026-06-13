# Student Predictor

Aplikacja webowa do analizy i predykcji wyników studentów z matematyki. Projekt wykorzystuje uczenie maszynowe (regresja logistyczna i las losowy) do przewidywania, czy student zda egzamin z matematyki (≥ 50 punktów).

## Funkcje

- Wczytywanie i czyszczenie danych z pliku CSV
- Wizualizacje: histogram ocen, boxplot płeć vs matematyka, scatterplot czytanie vs pisanie
- Trening modeli ML z metrykami: Accuracy, Precision, Recall, F1-Score
- Predykcja na przykładowym studencie
- Wzorzec Strategy do łatwej wymiany algorytmów

## Wymagania

- Python 3.12 (zalecane; projekt wspiera 3.10+)
- [uv](https://docs.astral.sh/uv/) — menedżer wersji Pythona, środowisk i pakietów

## Instalacja

```bash
git clone https://github.com/HooDaTi/student_predictor.git
cd student_predictor

uv python install 3.12
uv venv --python 3.12
uv python pin 3.12
uv pip install -r requirements.txt
```

Aktywacja venv (`source .venv/bin/activate`) nie jest wymagana — poniżej używaj `uv run`.

## Uruchomienie

```bash
uv run streamlit run app.py
```

Aplikacja otworzy się w przeglądarce (domyślnie `http://localhost:8501`).

## Testy

```bash
uv run pytest
```

## Struktura projektu

```
student_predictor/
├── .venv/                  # Wirtualne środowisko (uv)
├── app.py                  # Aplikacja Streamlit
├── data/
│   └── StudentsPerformance.csv
├── src/
│   ├── data_handler.py     # Wczytywanie i czyszczenie danych
│   ├── visualizer.py       # Wykresy i eksploracja danych
│   ├── model_strategy.py   # Wzorzec Strategy (Logistic, Random Forest)
│   └── trainer.py          # Trening i predykcja modeli
├── tests/
│   ├── test_data.py
│   └── test_model.py
└── requirements.txt
```

## Zbiór danych

Projekt korzysta z publicznego zbioru [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) (Kaggle). Zawiera wyniki egzaminów z matematyki, czytania i pisania wraz z danymi demograficznymi studentów.

## Technologie

- [Streamlit](https://streamlit.io/) — interfejs użytkownika
- [pandas](https://pandas.pydata.org/) — przetwarzanie danych
- [scikit-learn](https://scikit-learn.org/) — modele ML
- [matplotlib](https://matplotlib.org/) / [seaborn](https://seaborn.pydata.org/) — wizualizacje
- [pytest](https://pytest.org/) — testy jednostkowe

## Licencja

Projekt jest udostępniany na licencji [MIT](LICENSE).
