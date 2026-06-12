import streamlit as st
import pandas as pd

from src.data_handler import DataLoader
from src.visualizer import DataExplorer
from src.trainer import ModelTrainer
from src.model_strategy import LogisticStrategy, RandomForestStrategy

def main():
    st.set_page_config(page_title="Predykcja Ocen Studentów",
                        layout="wide")
    st.title("Analiza i Predykcja Wyników Studentów")
    
    st.header("1. Wczytywanie i podgląd danych")
    
    @st.cache_data
    def load_and_clean_data():
        loader = DataLoader("data/StudentsPerformance.csv")
        loader.load_data()
        df = loader.clean_data()
        stats = loader.get_basic_statistics()
        return df, stats
    
    try:
        df, stats = load_and_clean_data()
        st.success("Udało się pomyślnie wczytać i oczyścić dane!")
        st.dataframe(df.head())
        
        with st.expander("Rozwiń, aby zobaczyć macierz korelacji"):
            st.write(stats['korelacje'])
            
    except Exception as e:
        st.error(f"[BŁĄD] Wystąpił błąd podczas wczytywania danych. Szczegóły: {e}")
        st.stop()

    st.header("2. Wizualizacje, eksploracja danych")
    explorer = DataExplorer(df)

    tab1, tab2, tab3 = st.tabs(["Histogram ocen", "Płeć a matematyka", "Czytanie vs Pisanie"])
    
    with tab1:
        st.pyplot(explorer.plot_math_scores_histogram())
    with tab2:
        st.pyplot(explorer.plot_gender_vs_math())
    with tab3:
        st.pyplot(explorer.plot_reading_vs_writing())

    st.header("3. ML, Przewidywanie ryzyka niezdania")
    st.markdown("Model uczy się na podstawie cech studenta (płeć, kursy przygotowawcze itp.) i przewiduje, czy student zda matematykę (zdobędzie min. 50 punktów).")
    
    model_choice = st.selectbox(
        "Wybierz algorytm:",
        ["Regresja Logistyczna", "Las Losowy"]
    )
    
    if st.button("Trenuj model i sprawdź dokładność"):
        if model_choice == "Regresja Logistyczna":
            strategy = LogisticStrategy()
        else:
            strategy = RandomForestStrategy()

        trainer = ModelTrainer(strategy)
        metrics = trainer.run_training(df)

        st.success("Trening zakończony!")
        
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(label="Accuracy", value=f"{metrics['Accuracy'] * 100:.2f}%")
        col2.metric(label="Precision", value=f"{metrics['Precision'] * 100:.2f}%")
        col3.metric(label="Recall", value=f"{metrics['Recall'] * 100:.2f}%")
        col4.metric(label="F1-Score", value=f"{metrics['F1-Score'] * 100:.2f}%")

        st.divider()
        st.subheader("Test na pojedynczym studencie")
        
        dummy_student = {
            'gender': "female",
            'race/ethnicity': "group D",
            'parental level of education': "some high school",
            'lunch': "free/reduced",
            'test preparation course': "none",
            'writing score': 20,
            'reading score': 20
        }

        st.write("Dane studenta:", dummy_student)

        wynik = trainer.real_prediction(dummy_student)
        if wynik == 1:
            st.write("Student zdał matematykę! :)")
        else:
            st.write("Student nie zdał matematyki :(")


if __name__ == "__main__":
    main()