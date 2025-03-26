import streamlit as st
import requests
from io import StringIO
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pdb
#!pip install pycaret
from scipy.stats import poisson
import joblib
import warnings

st.title('Poisson com aprendizado de máquina')
st.subheader('Jogos do dia')
urls = {
    "Inglaterra - Premiere League": [
        "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2122/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2021/E0.csv"
    ],
    "Escócia - Premiere League": [
        "https://www.football-data.co.uk/mmz4281/2425/SC0.csv",
        "https://www.football-data.co.uk/mmz4281/2324/SC0.csv",
        "https://www.football-data.co.uk/mmz4281/2223/SC0.csv",
        "https://www.football-data.co.uk/mmz4281/2122/SC0.csv",
        "https://www.football-data.co.uk/mmz4281/2021/SC0.csv"
    ],
    "Alemanha - Bundesliga 1": [
        "https://www.football-data.co.uk/mmz4281/2425/D1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/D1.csv",
        "https://www.football-data.co.uk/mmz4281/2223/D1.csv",
        "https://www.football-data.co.uk/mmz4281/2122/D1.csv",
        "https://www.football-data.co.uk/mmz4281/2021/D1.csv"
    ],
    "Itália - Serie A": [
        "https://www.football-data.co.uk/mmz4281/2425/I1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/I1.csv",
        "https://www.football-data.co.uk/mmz4281/2223/I1.csv",
        "https://www.football-data.co.uk/mmz4281/2122/I1.csv",
        "https://www.football-data.co.uk/mmz4281/2021/I1.csv"
    ],
    "Espanha - La Liga": [
        "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/SP1.csv",
        "https://www.football-data.co.uk/mmz4281/2223/SP1.csv",
        "https://www.football-data.co.uk/mmz4281/2122/SP1.csv",
        "https://www.football-data.co.uk/mmz4281/2021/SP1.csv"
    ],
    "França - Primeira divisão": [
        "https://www.football-data.co.uk/mmz4281/2425/F1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/F1.csv",
        "https://www.football-data.co.uk/mmz4281/2223/F1.csv",
        "https://www.football-data.co.uk/mmz4281/2122/F1.csv",
        "https://www.football-data.co.uk/mmz4281/2021/F1.csv"
    ],
    "Holanda": [
        "https://www.football-data.co.uk/mmz4281/2425/N1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/N1.csv",
        "https://www.football-data.co.uk/mmz4281/2223/N1.csv",
        "https://www.football-data.co.uk/mmz4281/2122/N1.csv",
        "https://www.football-data.co.uk/mmz4281/2021/N1.csv"
    ],
    "Bélgica": [
        "https://www.football-data.co.uk/mmz4281/2425/B1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/B1.csv",
        "https://www.football-data.co.uk/mmz4281/2223/B1.csv",
        "https://www.football-data.co.uk/mmz4281/2122/B1.csv",
        "https://www.football-data.co.uk/mmz4281/2021/B1.csv"
    ],
    "Portugal - Liga 1": [
        "https://www.football-data.co.uk/mmz4281/2425/P1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/P1.csv",
        "https://www.football-data.co.uk/mmz4281/2223/P1.csv",
        "https://www.football-data.co.uk/mmz4281/2122/P1.csv",
        "https://www.football-data.co.uk/mmz4281/2021/P1.csv"
    ],
    "Turquia - Liga 1": [
        "https://www.football-data.co.uk/mmz4281/2425/T1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/T1.csv",
        "https://www.football-data.co.uk/mmz4281/2223/T1.csv",
        "https://www.football-data.co.uk/mmz4281/2122/T1.csv",
        "https://www.football-data.co.uk/mmz4281/2021/T1.csv"
    ],
    "Grecia": [
        "https://www.football-data.co.uk/mmz4281/2425/G1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/G1.csv",
        "https://www.football-data.co.uk/mmz4281/2223/G1.csv",
        "https://www.football-data.co.uk/mmz4281/2122/G1.csv",
        "https://www.football-data.co.uk/mmz4281/2021/G1.csv"
    ],
    "Argentina": ["https://www.football-data.co.uk/new/ARG.csv"],
    "Austria": ["https://www.football-data.co.uk/new/AUT.csv"],
    "Brasil": ["https://www.football-data.co.uk/new/BRA.csv"],
    "China": ["https://www.football-data.co.uk/new/CHN.csv"],
    "Dinamarca": ["https://www.football-data.co.uk/new/DNK.csv"],
    "Finlândia": ["https://www.football-data.co.uk/new/FIN.csv"],
    "Irlanda": ["https://www.football-data.co.uk/new/IRL.csv"],
    "Japao": ["https://www.football-data.co.uk/new/JPN.csv"],
    "México": ["https://www.football-data.co.uk/new/MEX.csv"],
    "Noruega": ["https://www.football-data.co.uk/new/NOR.csv"],
    "Polonia": ["https://www.football-data.co.uk/new/POL.csv"],
    "Romenia": ["https://www.football-data.co.uk/new/ROU.csv"],
    "Russia": ["https://www.football-data.co.uk/new/RUS.csv"],
    "Suecia": ["https://www.football-data.co.uk/new/SWE.csv"],
    "Suíça": ["https://www.football-data.co.uk/new/SWZ.csv"],
    "EUA": ["https://www.football-data.co.uk/new/USA.csv"]
}

# Pasta onde os arquivos das ligas estão salvos
output_folder = "league_data"
os.makedirs(output_folder, exist_ok=True)

# Função para baixar dados de uma URL
def download_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text))
    else:
        print(f"Erro ao baixar {url}")
        return pd.DataFrame()

# Função para padronizar as colunas
def standardize_columns(df, league_name):
    # Renomear colunas para o padrão
    if 'HG' in df.columns and 'AG' in df.columns:
        df.rename(columns={'HG': 'FTHG', 'AG': 'FTAG'}, inplace=True)
    if 'Home' in df.columns and 'Away' in df.columns:
        df.rename(columns={'Home': 'HomeTeam', 'Away': 'AwayTeam'}, inplace=True)
    
    # Adicionar coluna da liga
    df['League'] = league_name
    
    # Selecionar apenas as colunas necessárias
    required_columns = ['League', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
    for col in required_columns:
        if col not in df.columns:
            df[col] = None  # Adicionar coluna vazia se não existir
    
    return df[required_columns]

# Baixar e compilar os dados de todas as ligas
all_leagues = []
for league_name, league_urls in urls.items():
    league_data = pd.DataFrame()
    for url in league_urls:
        df = download_data(url)
        if not df.empty:
            df = standardize_columns(df, league_name)
            league_data = pd.concat([league_data, df], ignore_index=True)
    
    if not league_data.empty:
        all_leagues.append(league_data)
        print(f"Dados da liga {league_name} processados.")

# Concatenar todos os dados em um único DataFrame
if all_leagues:
    compiled_data = pd.concat(all_leagues, ignore_index=True)
   
st.dataframe(compiled_data)
