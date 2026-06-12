import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            print("Udało się wczytać plik CSV.")
            return self.df
        except FileNotFoundError:
            print("[BŁĄD] Nie znaleziono pliku. Sprawdź ścieżkę!")
            raise 
        except Exception as e:
            print(f"[BŁĄD] Coś poszło nie tak przy wczytywaniu: {e}")
            raise

    def clean_data(self):
        self.df = self.df.drop_duplicates()
        self.df = self.df.dropna()
        return self.df

    def get_basic_statistics(self):
        stats = {}
        
        stats['opis_danych'] = self.df.describe()
        
        stats['korelacje'] = self.df.corr(numeric_only=True)
        
        pierwsza_kolumna = self.df.columns[0]
        stats['rozklad_kategorii'] = self.df[pierwsza_kolumna].value_counts()
        
        return stats


if __name__ == "__main__":
    loader = DataLoader("./data/StudentsPerformance.csv")
    try:
        loader.load_data()
        loader.clean_data()
        wyniki = loader.get_basic_statistics()
        
        print("\nKorelacje w danych:")
        print(wyniki['korelacje'])
    except:
        print("[BŁĄD] Program napotkał błąd i zakończył działanie.")