import streamlit as st
import pandas as pd
from datetime import datetime

@st.cache_data(ttl=3600)  # Cache por 1 hora
def load_predictions():
    return pd.read_csv('https://raw.githubusercontent.com/joselucaspj/poissonpredictor/refs/heads/main/data/latest_predictions.csv', parse_dates=['Date'])

def main():
    st.title('Poisson com aprendizado de máquina')
    st.subheader('Jogos do dia')

    # Carrega apenas os dados processados
    try:
        df = load_predictions()

        # Interface lateral com filtros
        with st.sidebar:
            st.header('⚙️ Filtros')
            ligas = st.multiselect('Ligas', options=df['League'].unique(), default=df['League'].unique())

            min_date, max_date = df['Date'].min(), df['Date'].max()
            date_range = st.date_input(
                'Intervalo de Datas',
                value=[min_date, max_date],
                min_value=min_date,
                max_value=max_date
            )

            conf_range = st.slider('Confiança Mínima', 0.0, 1.0, 0.7, 0.05)

        # Aplicar filtros
        if len(date_range) == 2:
            mask = (
                (df['League'].isin(ligas)) &
                (df['Date'] >= pd.to_datetime(date_range[0])) &
                (df['Date'] <= pd.to_datetime(date_range[1])) &
                (df['prediction_confidence'] >= conf_range)
            )
            filtered_df = df[mask]
        else:
            st.warning("Selecione um intervalo de datas válido")
            filtered_df = df

        # Formatar a data para o padrão brasileiro
        filtered_df['Date'] = filtered_df['Date'].dt.strftime('%d/%m/%Y')

        st.dataframe(filtered_df)

    except Exception as e:
        st.error(f"Erro ao carregar previsões: {str(e)}")
        st.info("As previsões podem estar sendo atualizadas. Tente novamente em alguns minutos.")

if __name__ == "__main__":
    main()