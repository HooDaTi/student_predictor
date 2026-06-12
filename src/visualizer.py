import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class DataExplorer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        sns.set_theme(style="whitegrid")

    def plot_math_scores_histogram(self):
        # Histogram ocen z matematyki
        fig, ax = plt.subplots(figsize=(8, 5))
        
        sns.histplot(data=self.df, x='math score', bins=20, kde=True, ax=ax)
        
        ax.set_title("Rozkład ocen z matematyki", fontsize=14)
        ax.set_xlabel("Punkty")
        ax.set_ylabel("Liczba studentów")
        
        return fig

    def plot_gender_vs_math(self):
        # Boxplot oceny z matematyki z podziałem na płeć.
        fig, ax = plt.subplots(figsize=(8, 5))
        
        sns.boxplot(data=self.df, x='gender', y='math score', ax=ax)
        
        ax.set_title("Oceny z matematyki a płeć studenta", fontsize=14)
        ax.set_xlabel("Płeć")
        ax.set_ylabel("Punkty z matematyki")
        
        return fig

    def plot_reading_vs_writing(self):
        # Scatterplot zależność wyników z czytania i pisania.
        fig, ax = plt.subplots(figsize=(8, 5))

        sns.scatterplot(data=self.df, x='reading score', y='writing score', hue='gender', ax=ax)
        
        ax.set_title("Zależność: Czytanie vs Pisanie", fontsize=14)
        ax.set_xlabel("Punkty z czytania")
        ax.set_ylabel("Punkty z pisania")
        
        return fig


if __name__ == "__main__":
    from data_handler import DataLoader

    loader = DataLoader("./data/StudentsPerformance.csv")
    df = loader.load_data()
    df = loader.clean_data()

    explorer = DataExplorer(df)

    fig1 = explorer.plot_math_scores_histogram()
    fig2 = explorer.plot_gender_vs_math()
    fig3 = explorer.plot_reading_vs_writing()
    plt.show()