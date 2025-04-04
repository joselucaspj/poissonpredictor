import gdown
import streamlit as st
import requests
from io import StringIO
import os
import pandas as pd
from pytz import timezone
import numpy as np
from scipy.stats import poisson
import joblib
import warnings

# Configuração inicial
st.set_page_config(layout="wide")
warnings.filterwarnings('ignore')
def calcular_diferenca_gols_para_jogos_do_dia(df_historico, df_today):
    """
    Calcula as diferenças de gols para os jogos do dia usando o histórico acumulado
    
    Parâmetros:
    df_historico: DataFrame com todos os jogos históricos
    df_today: DataFrame com os jogos do dia a serem calculados
    
    Retorna:
    DataFrame df_today com as colunas de diferença de gols adicionadas
    """
    # Primeiro processamos todo o histórico para obter os acumulados
    history = {
        'home': {},  # Estatísticas como mandante
        'away': {},  # Estatísticas como visitante
        'all': {}    # Estatísticas gerais (casa+fora)
    }
    
    # Processar o histórico completo
    for idx, row in df_historico.iterrows():
        home_team = row['Home_Team_ID']
        away_team = row['Away_Team_ID']
        
        # Atualizar histórico como mandante
        if home_team not in history['home']:
            history['home'][home_team] = {'gols_marcados': [], 'gols_sofridos': []}
        history['home'][home_team]['gols_marcados'].append(row['FTHG'])
        history['home'][home_team]['gols_sofridos'].append(row['FTAG'])
        
        # Atualizar histórico como visitante
        if away_team not in history['away']:
            history['away'][away_team] = {'gols_marcados': [], 'gols_sofridos': []}
        history['away'][away_team]['gols_marcados'].append(row['FTAG'])
        history['away'][away_team]['gols_sofridos'].append(row['FTHG'])
        
        # Atualizar histórico geral
        if home_team not in history['all']:
            history['all'][home_team] = {'gols_marcados': [], 'gols_sofridos': []}
        history['all'][home_team]['gols_marcados'].append(row['FTHG'])
        history['all'][home_team]['gols_sofridos'].append(row['FTAG'])
        
        if away_team not in history['all']:
            history['all'][away_team] = {'gols_marcados': [], 'gols_sofridos': []}
        history['all'][away_team]['gols_marcados'].append(row['FTAG'])
        history['all'][away_team]['gols_sofridos'].append(row['FTHG'])
    
    # Função para calcular as diferenças
    def get_diffs(team, team_type):
        stats = history[team_type].get(team, {'gols_marcados': [], 'gols_sofridos': []})
        
        all_gm = sum(stats['gols_marcados'])
        all_gs = sum(stats['gols_sofridos'])
        
        last10_gm = stats['gols_marcados'][-10:]
        last10_gs = stats['gols_sofridos'][-10:]
        
        last5_gm = stats['gols_marcados'][-5:]
        last5_gs = stats['gols_sofridos'][-5:]
        
        return {
            'all': all_gm - all_gs,
            'last10': sum(last10_gm) - sum(last10_gs),
            'last5': sum(last5_gm) - sum(last5_gs)
        }
    
    # Agora calculamos para os jogos do dia
    for idx, row in df_today.iterrows():
        home_team = row['Home_Team_ID']
        away_team = row['Away_Team_ID']
        
        # Calcular estatísticas (não atualizamos o histórico para os jogos do dia)
        diffs_home_as_home = get_diffs(home_team, 'home')
        diffs_away_as_away = get_diffs(away_team, 'away')
        diffs_home_all = get_diffs(home_team, 'all')
        diffs_away_all = get_diffs(away_team, 'all')
        
        # Atribuir aos DataFrames
        df_today.at[idx, 'Home_Team_As_Home_Diff_All'] = diffs_home_as_home['all']
        df_today.at[idx, 'Home_Team_As_Home_Diff_Last10'] = diffs_home_as_home['last10']
        df_today.at[idx, 'Home_Team_As_Home_Diff_Last5'] = diffs_home_as_home['last5']
        
        df_today.at[idx, 'Away_Team_As_Away_Diff_All'] = diffs_away_as_away['all']
        df_today.at[idx, 'Away_Team_As_Away_Diff_Last10'] = diffs_away_as_away['last10']
        df_today.at[idx, 'Away_Team_As_Away_Diff_Last5'] = diffs_away_as_away['last5']
        
        df_today.at[idx, 'Home_Team_Diff_All'] = diffs_home_all['all']
        df_today.at[idx, 'Home_Team_Diff_Last10'] = diffs_home_all['last10']
        df_today.at[idx, 'Home_Team_Diff_Last5'] = diffs_home_all['last5']
        
        df_today.at[idx, 'Away_Team_Diff_All'] = diffs_away_all['all']
        df_today.at[idx, 'Away_Team_Diff_Last10'] = diffs_away_all['last10']
        df_today.at[idx, 'Away_Team_Diff_Last5'] = diffs_away_all['last5']
    
    return df_today
def calcular_diferenca_gols_anteriores_novo(df):
    """
    Versão que considera APENAS jogos ANTERIORES ao jogo atual para cálculo:
    - Mantém todas as funcionalidades originais
    - Elimina vazamento de dados (data leakage)
    - Ordem de complexidade O(n)
    """
   #df = df.sort_values(['Home_Team_ID', 'Date']).copy()
    
    # Dicionários para armazenar históricos
    history = {
        'home': {},  # Para estatísticas como mandante
        'away': {},  # Para estatísticas como visitante
        'all': {}    # Para estatísticas gerais (casa+fora)
    }
    
    # Listas para armazenar resultados
    results = {
        'home': {'all': [], 'last10': [], 'last5': []},
        'away': {'all': [], 'last10': [], 'last5': []},
        'home_as_home': {'all': [], 'last10': [], 'last5': []},
        'away_as_away': {'all': [], 'last10': [], 'last5': []}
    }
    
    for idx, row in df.iterrows():
        home_team = row['Home_Team_ID']
        away_team = row['Away_Team_ID']
        
        # Função para calcular diferenças SEM incluir o jogo atual
        def get_diffs(team, team_type):
            stats = history[team_type].get(team, {'gols_marcados': [], 'gols_sofridos': []})
            
            all_gm = sum(stats['gols_marcados'])
            all_gs = sum(stats['gols_sofridos'])
            
            last10_gm = stats['gols_marcados'][-10:]
            last10_gs = stats['gols_sofridos'][-10:]
            
            last5_gm = stats['gols_marcados'][-5:]
            last5_gs = stats['gols_sofridos'][-5:]
            
            return {
                'all': all_gm - all_gs,
                'last10': sum(last10_gm) - sum(last10_gs),
                'last5': sum(last5_gm) - sum(last5_gs)
            }
        
        # Calcular estatísticas ANTES do jogo atual
        # Time mandante (como mandante)
        diffs_home_as_home = get_diffs(home_team, 'home')
        # Time visitante (como visitante)
        diffs_away_as_away = get_diffs(away_team, 'away')
        # Time mandante (todos jogos)
        diffs_home_all = get_diffs(home_team, 'all')
        # Time visitante (todos jogos)
        diffs_away_all = get_diffs(away_team, 'all')
        
        # Armazenar resultados
        for k in ['all', 'last10', 'last5']:
            results['home_as_home'][k].append(diffs_home_as_home[k])
            results['away_as_away'][k].append(diffs_away_as_away[k])
            results['home'][k].append(diffs_home_all[k])
            results['away'][k].append(diffs_away_all[k])
        
        # Atualizar históricos (APÓS o cálculo)
        # Atualizar histórico como mandante
        if home_team not in history['home']:
            history['home'][home_team] = {'gols_marcados': [], 'gols_sofridos': []}
        history['home'][home_team]['gols_marcados'].append(row['FTHG'])
        history['home'][home_team]['gols_sofridos'].append(row['FTAG'])
        
        # Atualizar histórico como visitante
        if away_team not in history['away']:
            history['away'][away_team] = {'gols_marcados': [], 'gols_sofridos': []}
        history['away'][away_team]['gols_marcados'].append(row['FTAG'])
        history['away'][away_team]['gols_sofridos'].append(row['FTHG'])
        
        # Atualizar histórico geral
        if home_team not in history['all']:
            history['all'][home_team] = {'gols_marcados': [], 'gols_sofridos': []}
        history['all'][home_team]['gols_marcados'].append(row['FTHG'])
        history['all'][home_team]['gols_sofridos'].append(row['FTAG'])
        
        if away_team not in history['all']:
            history['all'][away_team] = {'gols_marcados': [], 'gols_sofridos': []}
        history['all'][away_team]['gols_marcados'].append(row['FTAG'])
        history['all'][away_team]['gols_sofridos'].append(row['FTHG'])
    
    # Adicionar colunas ao DataFrame
    df['Home_Team_As_Home_Diff_All'] = results['home_as_home']['all']
    df['Home_Team_As_Home_Diff_Last10'] = results['home_as_home']['last10']
    df['Home_Team_As_Home_Diff_Last5'] = results['home_as_home']['last5']
    
    df['Away_Team_As_Away_Diff_All'] = results['away_as_away']['all']
    df['Away_Team_As_Away_Diff_Last10'] = results['away_as_away']['last10']
    df['Away_Team_As_Away_Diff_Last5'] = results['away_as_away']['last5']
    
    df['Home_Team_Diff_All'] = results['home']['all']
    df['Home_Team_Diff_Last10'] = results['home']['last10']
    df['Home_Team_Diff_Last5'] = results['home']['last5']
    
    df['Away_Team_Diff_All'] = results['away']['all']
    df['Away_Team_Diff_Last10'] = results['away']['last10']
    df['Away_Team_Diff_Last5'] = results['away']['last5']
    
    return df
def contar_resultados(dataframe):
    vitoria_mandante = 0
    soma_probabilidade_placar_mandante = 0
    vitoria_visitante = 0
    soma_probabilidade_placar_visitante = 0
    empate = 0
    soma_probabilidade_placar_empate = 0

    for indice, linha in dataframe.iterrows():
        if linha['Home_Goals'] > linha['Away_Goals']:
            vitoria_mandante += 1
            soma_probabilidade_placar_mandante = soma_probabilidade_placar_mandante + linha['Probability']
        elif linha['Home_Goals'] < linha['Away_Goals']:
            vitoria_visitante += 1
            soma_probabilidade_placar_visitante = soma_probabilidade_placar_visitante + linha['Probability']
        else:
            empate += 1
            soma_probabilidade_placar_empate = soma_probabilidade_placar_empate + linha['Probability']

    return {'Vitória do Mandante': vitoria_mandante, 'Vitória do Visitante': vitoria_visitante, 'Empate': empate, 'Probabilidade placares mandante': soma_probabilidade_placar_mandante, 'Probabilidade placares visitante': soma_probabilidade_placar_visitante, 'Probabilidade placares empate': soma_probabilidade_placar_empate }

def drop_reset_index(df):
    df = df.dropna()
    df = df.reset_index(drop=True)
    df.index += 1
    return df

def simulate_match(home_goals_for, home_goals_against, away_goals_for, away_goals_against, num_simulations=10000, random_seed=42):
    np.random.seed(random_seed)
    estimated_home_goals = (home_goals_for + away_goals_against) / 2
    estimated_away_goals = (away_goals_for + home_goals_against) / 2

    home_goals = poisson(estimated_home_goals).rvs(num_simulations)
    away_goals = poisson(estimated_away_goals).rvs(num_simulations)

    results = pd.DataFrame({
        'Home_Goals': home_goals,
        'Away_Goals': away_goals
    })

    return results
def simulate_match_predict(home_goals_for, away_goals_for, num_simulations=10000, random_seed=42):
    np.random.seed(random_seed)

    home_goals = poisson(home_goals_for).rvs(num_simulations)
    away_goals = poisson(away_goals_for).rvs(num_simulations)

    results = pd.DataFrame({
        'Home_Goals': home_goals,
        'Away_Goals': away_goals
    })

    return results
def top_results_df(simulated_results, top_n):

    result_counts = simulated_results.value_counts().head(top_n).reset_index()
    result_counts.columns = ['Home_Goals', 'Away_Goals', 'Count']

    sum_top_counts = result_counts['Count'].sum()
    result_counts['Probability'] = result_counts['Count'] / sum_top_counts

    return result_counts

def media_gols_marcados_HA_ultimos_n_jogos(df, equipe,liga,data_filtro,gols, n):

    df_equipe = df[((df['Home_Team_ID'] == equipe) | (df['Away_Team_ID'] == equipe)) & (df['League_ID'] == liga) & (df['Date'] < data_filtro)].tail(n)
    #display(df_equipe)
    if df_equipe.shape[0] == 0:
      resultado_media_gm_h_ha = gols
    else:
      resultado_media_gm_h_ha = round((df_equipe[(df_equipe['Home_Team_ID'] == equipe)]['FTHG'].sum() + df_equipe[(df_equipe['Away_Team_ID'] == equipe)]['FTAG'].sum()) / df_equipe.shape[0], 2)
    #print(resultado_media_gm_h_ha)
    return resultado_media_gm_h_ha

def media_gols_sofridos_HA_ultimos_n_jogos(df, equipe,liga,data_filtro,gols, n):
    df_equipe = df[((df['Home_Team_ID'] == equipe) | (df['Away_Team_ID'] == equipe)) & (df['League_ID'] == liga) & (df['Date'] < data_filtro)].tail(n)
    #display(df_equipe)
    if df_equipe.shape[0] == 0:
      resultado_media_gs_h_ha = gols
    else:
      resultado_media_gs_h_ha = round((df_equipe[(df_equipe['Home_Team_ID'] == equipe)]['FTAG'].sum() + df_equipe[(df_equipe['Away_Team_ID'] == equipe)]['FTHG'].sum()) / df_equipe.shape[0], 2)
    #print(resultado_media_gs_h_ha)
    return resultado_media_gs_h_ha

def media_gols_Marcados_vs_media_gols_Sofridos(df, equipe,liga,data_filtro,media_gols, n):
    df_equipe = df[((df['Home'] == equipe) | (df['Away'] == equipe)) & (df['League'] == liga) & (df['Date'] < data_filtro)]
    if df_equipe.shape[0] == 0:
      resultado_media_gs_h_ha = 0
    else:
      resultado_media_gs_h_ha = round((df_equipe[(df_equipe['Home'] == equipe)]['Goals_A'].sum() + df_equipe[(df_equipe['Away'] == equipe)]['Goals_H'].sum()) / df_equipe.shape[0], 2)
    #print(resultado_media_gs_h_ha)
    return resultado_media_gs_h_ha
# 1. Funções de Carregamento com Cache
@st.cache_data(ttl=3600)  # Cache por 1 hora
def load_models():
    MODEL_GOLS_URL = 'https://drive.google.com/uc?id=1XpKUMdD05ZZ70gLDsFaC2wzATm_FCdz7'
    MODEL_WINNER_URL = 'https://drive.google.com/uc?id=1b_uaLyGSBjxN8oLJMY0-rlXVbMlFu42R'
    MODEL_WINNER_HOME = 'https://drive.google.com/uc?id=19INu-irZFD3u0NfTnisHQPDkjxlWlTdo'
    MODEL_WINNER_AWAY='https://drive.google.com/uc?id=1miAciy7g46RzrQ3lu5PTMJD5AJdEzD6P'
    os.makedirs("models", exist_ok=True)
    gols_path = "models/modelo_predict_gols.pkl"
    winner_path = "models/modelo_predict_winner.pkl"
    winner_home_path = "models/modelo_winner_home.pkl"
    winner_away_path = "models/modelo_winner_away.pkl"
    
    if not os.path.exists(gols_path):
        gdown.download(MODEL_GOLS_URL, gols_path, quiet=True)
    if not os.path.exists(winner_path):
        gdown.download(MODEL_WINNER_URL, winner_path, quiet=True)
    if not os.path.exists(winner_home_path):
        gdown.download(MODEL_WINNER_HOME, winner_home_path, quiet=True)
    if not os.path.exists(winner_away_path):
        gdown.download(MODEL_WINNER_AWAY, winner_away_path, quiet=True)    
        
    return joblib.load(gols_path), joblib.load(winner_path)

@st.cache_data(ttl=3600)
def load_league_mapping():
    return pd.read_csv('https://raw.githubusercontent.com/joselucaspj/poissonpredictor/refs/heads/main/App/assets/ligas_dicionario.csv')

@st.cache_data(ttl=3600)
def load_team_mapping():
    return pd.read_csv('https://github.com/joselucaspj/poissonpredictor/raw/refs/heads/main/App/assets/times_dicionario.csv')
def get_compiled_data():
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
    return compiled_data

def get_todays_matches():
    url = "https://www.football-data.co.uk/fixtures.csv"
    url_new_league = "https://www.football-data.co.uk/new_league_fixtures.csv"
    league_mapping = {
    'E0': 'Inglaterra - Premiere League',
    'SC0': 'Escócia - Premiere League',
    'D1': 'Alemanha - Bundesliga 1',
    'I1': 'Itália - Serie A',
    'SP1': 'Espanha - La Liga',
    'F1': 'França - Primeira divisão',
    'N1': 'Holanda',
    'B1': 'Bélgica',
    'P1': 'Portugal - Liga 1',
    'T1': 'Turquia - Liga 1',
    'G1': 'Grecia',
    'ARG': 'Argentina',
    'AUT': 'Austria',
    'BRA': 'Brasil',
    'CHN': 'China',
    'FIN': 'Finlândia',
    'IRL': 'Irlanda',
    'JPN': 'Japao',
    'MEX': 'México',
    'NOR': 'Noruega',
    'POL': 'Polonia',
    'ROU': 'Romenia',
    'RUS': 'Russia',
    'SWE': 'Suecia',
    'SWZ': 'Suíça',
    'USA': 'EUA',
    'China' : 'China',
    'Japan' : 'Japao',
    'Switzerland' : 'Suíça',
    'Argentina' : 'Argentina',
    'Austria' : 'Austria',
    'Brazil' : 'Brasil',
    'Denmark' : 'Dinamarca',
    'Finland' : 'Finlândia',
    'Ireland' : 'Irlanda',
    'Mexico' : 'Mexico',
    'Norway' : 'Noruega',
    'Poland' : 'Polonia',
    'Romania' : 'Romenia',
    'Russia' : 'Russia',
    'Sweden' : 'Suecia'
}
    
    def download_todays_matches():
        try:
            # Baixar os dados
            response = requests.get(url)
            response.raise_for_status()
            
            # Remover o BOM (ï»¿) manualmente se existir
            content = response.text
            if content.startswith('\ufeff'):
                content = content[1:]
            
            # Ler o CSV
            df = pd.read_csv(StringIO(content))
            
            # Corrigir o nome da coluna Div (removendo possíveis caracteres especiais)
            df.columns = df.columns.str.replace('\ufeff', '')  # Remove BOM se ainda existir
            df.columns = df.columns.str.strip()  # Remove espaços em branco
            
            # Verificar se a coluna existe (pode ser 'Div' ou 'ï»¿Div')
            div_col = [col for col in df.columns if 'Div' in col]
            if not div_col:
                raise ValueError("Coluna 'Div' não encontrada no arquivo CSV")
            
            # Selecionar e renomear colunas
            df = df.rename(columns={div_col[0]: 'League'})
            df = df[['League', 'Date', 'Time', 'HomeTeam', 'AwayTeam']]
            df.columns = ['League', 'Date', 'TIME', 'HomeTeam', 'AwayTeam']
            
            # Mapear os códigos das ligas para nomes completos
            df['League'] = df['League'].map(league_mapping)
            
            # Remover linhas com valores nulos
            df = df.dropna(subset=['League'])
            
            # Converter e ordenar por data e hora
            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['TIME'], dayfirst=True)
            origin_tz = timezone('Europe/London')  # Considera horário de verão automaticamente
            df['datetime'] = df['datetime'].dt.tz_localize(origin_tz, ambiguous='NaT')
            df['TIME_BRASIL'] = df['datetime'].dt.tz_convert(timezone('America/Sao_Paulo')).dt.strftime('%H:%M')
            df['TIME'] =  df['TIME_BRASIL']
            df = df.sort_values('datetime')
            
            # Resetar índice
            df = df.reset_index(drop=True)
            
            return df
        
        except Exception as e:
            print(f"Erro ao processar os dados: {str(e)}")
            return pd.DataFrame()
    def download_todays_matches_new():
        try:
            # Baixar os dados
            response = requests.get(url_new_league)
            response.raise_for_status()
            
            # Remover o BOM (ï»¿) manualmente se existir
            content = response.text
            if content.startswith('\ufeff'):
                content = content[1:]
            
            # Ler o CSV
            df = pd.read_csv(StringIO(content))
            # Corrigir o nome da coluna Div (removendo possíveis caracteres especiais)
            df.columns = df.columns.str.replace('\ufeff', '')  # Remove BOM se ainda existir
            df.columns = df.columns.str.strip()  # Remove espaços em branco
            df.drop(['League'], axis=1, inplace=True)
            # Verificar se a coluna existe (pode ser 'Div' ou 'ï»¿Div')
            div_col = [col for col in df.columns if 'Country' in col]
            if not div_col:
                raise ValueError("Coluna 'Div' não encontrada no arquivo CSV")
            
            # Selecionar e renomear colunas
            df = df.rename(columns={div_col[0]: 'League'})
            df = df[['League', 'Date', 'Time', 'Home', 'Away']]
            df.columns = ['League', 'Date', 'TIME', 'HomeTeam', 'AwayTeam']
            
            # Mapear os códigos das ligas para nomes completos
            df['League'] = df['League'].map(league_mapping)
            
            # Remover linhas com valores nulos
            df = df.dropna(subset=['League'])
            
            # Converter e ordenar por data e hora
            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['TIME'], dayfirst=True)
           
            origin_tz = timezone('Europe/London')  # Considera horário de verão automaticamente
            df['datetime'] = df['datetime'].dt.tz_localize(origin_tz, ambiguous='NaT')
            df['TIME_BRASIL'] = df['datetime'].dt.tz_convert(timezone('America/Sao_Paulo')).dt.strftime('%H:%M')
            df['TIME'] =  df['TIME_BRASIL']
            # Resetar índice
            df = df.reset_index(drop=True)
            
            return df
        
        except Exception as e:
            print(f"Erro ao processar os dados: {str(e)}")
            return pd.DataFrame()
    
    # Executar e salvar
    todays_matches = download_todays_matches()
    todays_matches_new = download_todays_matches_new()
    todays_matches = pd.concat([todays_matches, todays_matches_new], ignore_index=True)
    return todays_matches
# 2. Funções de Processamento Principal
@st.cache_data(ttl=600)  # Cache por 10 minutos
def process_main_data(_model_gols, _model_winner, model_winner_home, model_winner_away, ligas_dicionario, times_dicionario):
    # Todas suas operações de processamento aqui
    # (Mantenha a mesma lógica, mas dentro desta função)
    
    # Exemplo:
    compiled_data = get_compiled_data()  # Você precisará criar esta função
    todays_matches = get_todays_matches()  # E esta também
    jogos_do_dia = todays_matches
    base = compiled_data
    model_gols, model_winner, model_winner_home, model_winner_away = load_models()
    base['Date'] = pd.to_datetime(base['Date'], dayfirst=True)
    base = base[(base['Date'].dt.year >= 2020)]
    base.sort_values('Date', ascending=True, inplace = True)
    
    jogos_do_dia['League_ID'] = jogos_do_dia['League'].map(ligas_dicionario.set_index('League')['League_ID'])
    jogos_do_dia['Home_Team_ID'] = jogos_do_dia['HomeTeam'].map(times_dicionario.set_index('Team')['Team_ID'])
    jogos_do_dia['Away_Team_ID'] = jogos_do_dia['AwayTeam'].map(times_dicionario.set_index('Team')['Team_ID'])
    base['League_ID'] = base['League'].map(ligas_dicionario.set_index('League')['League_ID'])
    base['Home_Team_ID'] = base['HomeTeam'].map(times_dicionario.set_index('Team')['Team_ID'])
    base['Away_Team_ID'] = base['AwayTeam'].map(times_dicionario.set_index('Team')['Team_ID'])
    
    jogos_do_dia = drop_reset_index(jogos_do_dia)
    base = drop_reset_index(base)
    jogos_do_dia['Media_GM_H_HA'] = jogos_do_dia.apply(lambda row: media_gols_marcados_HA_ultimos_n_jogos(base, row['Home_Team_ID'], row['League_ID'],row['Date'],0, 5), axis=1)
    jogos_do_dia['Media_GS_H_HA'] = jogos_do_dia.apply(lambda row: media_gols_sofridos_HA_ultimos_n_jogos(base, row['Home_Team_ID'], row['League_ID'],row['Date'],0, 5), axis=1)
    jogos_do_dia['Media_GM_A_HA'] = jogos_do_dia.apply(lambda row: media_gols_marcados_HA_ultimos_n_jogos(base, row['Away_Team_ID'], row['League_ID'],row['Date'],0, 5), axis=1)
    jogos_do_dia['Media_GS_A_HA'] = jogos_do_dia.apply(lambda row: media_gols_sofridos_HA_ultimos_n_jogos(base, row['Away_Team_ID'], row['League_ID'],row['Date'],0, 5), axis=1)
    jogos_do_dia = drop_reset_index(jogos_do_dia)
    jogos_do_dia['Probabilidade_placar_mandante_media_HA']=None
    jogos_do_dia['Probabilidade_placar_visitante_media_AH']=None
    jogos_do_dia['Probabilidade_placar_empate_media_HA']=None
    for indice, linha in jogos_do_dia.iterrows():
      simulated_results = simulate_match(linha['Media_GM_H_HA'], linha['Media_GS_H_HA'], linha['Media_GM_A_HA'], linha['Media_GS_A_HA'])
      simulated_results = drop_reset_index(simulated_results)
    
      results = top_results_df(simulated_results,100)
      results = drop_reset_index(results)
      resultados = contar_resultados(results)
      jogos_do_dia.at[indice,'Probabilidade_placar_mandante_media_HA'] = resultados['Probabilidade placares mandante']
      jogos_do_dia.at[indice,'Probabilidade_placar_visitante_media_AH'] = resultados['Probabilidade placares visitante']
      jogos_do_dia.at[indice,'Probabilidade_placar_empate_media_HA'] = resultados['Probabilidade placares empate']
    
    jogos_do_dia['0.0x0.0_Probability_HA'] = 0
    jogos_do_dia['1.0x0.0_Probability_HA'] = 0
    jogos_do_dia['2.0x0.0_Probability_HA'] = 0
    jogos_do_dia['3.0x0.0_Probability_HA'] = 0
    jogos_do_dia['0.0x1.0_Probability_HA'] = 0
    jogos_do_dia['0.0x2.0_Probability_HA'] = 0
    jogos_do_dia['0.0x3.0_Probability_HA'] = 0
    jogos_do_dia['1.0x1.0_Probability_HA'] = 0
    jogos_do_dia['2.0x1.0_Probability_HA'] = 0
    jogos_do_dia['3.0x1.0_Probability_HA'] = 0
    jogos_do_dia['1.0x2.0_Probability_HA'] = 0
    jogos_do_dia['1.0x3.0_Probability_HA'] = 0
    jogos_do_dia['2.0x2.0_Probability_HA'] = 0
    jogos_do_dia['3.0x2.0_Probability_HA'] = 0
    jogos_do_dia['2.0x3.0_Probability_HA'] = 0
    jogos_do_dia['3.0x3.0_Probability_HA'] = 0
    jogos_do_dia['Goleada_Home_Probability_HA'] = 0
    jogos_do_dia['Goleada_Away_Probability_HA'] = 0
    jogos_do_dia['Goleada_Empate_Probability_HA'] = 0
    for index, row in jogos_do_dia.iterrows():
        home_goals_for = row['Media_GM_H_HA']
        home_goals_against = row['Media_GS_H_HA']
        away_goals_for = row['Media_GM_A_HA']
        away_goals_against = row['Media_GS_A_HA']
    
        # Simular o resultado da partida
        simulated_results = simulate_match(home_goals_for, home_goals_against, away_goals_for, away_goals_against)
    
        # Gerar dataframe com os resultados mais frequentes
        top_results = top_results_df(simulated_results, 100)
    
        # Adicionar as informações ao dataframe original
        for i, result in top_results.iterrows():
            home_goals = result['Home_Goals']
            away_goals = result['Away_Goals']
            probability = result['Probability']
    
    
            if home_goals >= 4 and home_goals > away_goals:
              jogos_do_dia.at[index,'Goleada_Home_Probability_HA'] = jogos_do_dia.at[index,'Goleada_Home_Probability_HA'] + probability
            elif away_goals >= 4 and home_goals < away_goals:
              jogos_do_dia.at[index,'Goleada_Away_Probability_HA'] = jogos_do_dia.at[index,'Goleada_Away_Probability_HA'] + probability
            elif away_goals >= 4 and away_goals >= 4:
              jogos_do_dia.at[index,'Goleada_Empate_Probability_HA'] = jogos_do_dia.at[index,'Goleada_Empate_Probability_HA'] + probability
            else:
              coluna_placar = f'{home_goals}x{away_goals}_Probability_HA'
              jogos_do_dia.at[index, coluna_placar] = probability
    
    features = [    'Media_GM_H_HA', 'Media_GS_H_HA', 'Media_GM_A_HA', 'Media_GS_A_HA','Probabilidade_placar_mandante_media_HA',
                    'Probabilidade_placar_visitante_media_AH', 'Probabilidade_placar_empate_media_HA',
                    '0.0x0.0_Probability_HA','1.0x0.0_Probability_HA','2.0x0.0_Probability_HA','3.0x0.0_Probability_HA',
                    '0.0x1.0_Probability_HA','0.0x2.0_Probability_HA','0.0x3.0_Probability_HA','1.0x1.0_Probability_HA',
                    '2.0x1.0_Probability_HA','3.0x1.0_Probability_HA','1.0x2.0_Probability_HA','1.0x3.0_Probability_HA',
                    '2.0x2.0_Probability_HA','3.0x2.0_Probability_HA','2.0x3.0_Probability_HA','3.0x3.0_Probability_HA',
                    'Goleada_Home_Probability_HA','Goleada_Away_Probability_HA','Goleada_Empate_Probability_HA','League_ID','Home_Team_ID','Away_Team_ID']
    
    model= model_gols
    # Previsões para os dados de teste
    X_test = jogos_do_dia[features]
    predicted = model.predict(X_test)
    jogos_do_dia['Predicted_Goals_H'] = predicted[:, 0]
    jogos_do_dia['Predicted_Goals_A'] = predicted[:, 1]
    
    num_simulations = 10000
    for indice, row in jogos_do_dia.iterrows():
            home_goals = row['Predicted_Goals_H']
            away_goals = row['Predicted_Goals_A']
            # Realizando a simulação com o número de simulações ajustado
            simulated_results = simulate_match_predict(home_goals, away_goals, num_simulations=num_simulations)
    
            # Obtenção dos 8 placares mais prováveis
            top_results = top_results_df(simulated_results, top_n=8)
    
            for i, result in top_results.iterrows():
                home_goals = int(result['Home_Goals'])
                away_goals = int(result['Away_Goals'])
                probability = result['Probability']
                coluna_placar_H = f'Placar {i}_H'
                coluna_placar_A = f'Placar {i}_A'
                coluna_probabilidade = f'Probabilidade Placar {i}'
                jogos_do_dia.at[indice, coluna_placar_H] = home_goals
                jogos_do_dia.at[indice, coluna_placar_A] = away_goals
                jogos_do_dia.at[indice, coluna_probabilidade] = probability 
    #jogos_do_dia.to_csv("teste_aprendizado_de_maquina.csv", index=False)
    base.sort_values('Date', ascending=True, inplace = True)
    jogos_do_dia = calcular_diferenca_gols_para_jogos_do_dia(base, jogos_do_dia)
    jogos_do_dia['Placar_0_Diff'] = jogos_do_dia['Placar 0_H'] - jogos_do_dia['Placar 0_A']
    jogos_do_dia['Placar_1_Diff'] = jogos_do_dia['Placar 1_H'] - jogos_do_dia['Placar 1_A']
    jogos_do_dia['Placar_2_Diff'] = jogos_do_dia['Placar 2_H'] - jogos_do_dia['Placar 2_A']
    jogos_do_dia['Placar_3_Diff'] = jogos_do_dia['Placar 3_H'] - jogos_do_dia['Placar 3_A']
    jogos_do_dia['Placar_4_Diff'] = jogos_do_dia['Placar 4_H'] - jogos_do_dia['Placar 4_A']
    jogos_do_dia['Placar_5_Diff'] = jogos_do_dia['Placar 5_H'] - jogos_do_dia['Placar 5_A']
    jogos_do_dia['Placar_6_Diff'] = jogos_do_dia['Placar 6_H'] - jogos_do_dia['Placar 6_A']
    jogos_do_dia['Placar_7_Diff'] = jogos_do_dia['Placar 7_H'] - jogos_do_dia['Placar 7_A']
    
    classification_features = features + ['Placar 0_H', 'Placar 0_A','Probabilidade Placar 0', 'Placar 1_H', 'Placar 1_A','Probabilidade Placar 1', 'Placar 2_H', 'Placar 2_A',
                           'Probabilidade Placar 2', 'Placar 3_H', 'Placar 3_A','Probabilidade Placar 3', 'Placar 4_H', 'Probabilidade Placar 4','Placar 5_H', 'Placar 5_A', 'Probabilidade Placar 5', 'Placar 6_H',
                           'Placar 6_A', 'Probabilidade Placar 6', 'Placar 7_H', 'Placar 7_A','Probabilidade Placar 7','Placar_0_Diff','Placar_1_Diff','Placar_2_Diff',
                           'Placar_3_Diff','Placar_4_Diff','Placar_5_Diff','Placar_6_Diff','Placar_7_Diff','Home_Team_As_Home_Diff_All','Home_Team_As_Home_Diff_Last10',
                                      'Home_Team_As_Home_Diff_Last5','Away_Team_As_Away_Diff_All','Away_Team_As_Away_Diff_Last10','Away_Team_As_Away_Diff_Last5','Home_Team_Diff_All',
                                     'Home_Team_Diff_Last10','Home_Team_Diff_Last5','Away_Team_Diff_All','Away_Team_Diff_Last10','Away_Team_Diff_Last5']
    modelo_winner = model_winner
    modelo_winner_home = model_winner_home
    modelo_winner_away = model_winner_away
    pridict_winner_home = modelo_winner_home.predict(jogos_do_dia[classification_features])
    probabilities_best_home = modelo_winner_home.predict_proba(jogos_do_dia[classification_features])
    pridict_winner_away = modelo_winner_away.predict(jogos_do_dia[classification_features])
    probabilities_best_away = modelo_winner_away.predict_proba(jogos_do_dia[classification_features])
    jogos_do_dia['Predict_winner_home']= pridict_winner_home
    jogos_do_dia['probabilities_best_home'] = np.max(probabilities_best_home, axis=1)
    jogos_do_dia['Predict_winner_away']= pridict_winner_away
    jogos_do_dia['probabilities_best_away'] = np.max(probabilities_best_away, axis=1)
    jogos_do_dia_home = jogos_do_dia[(jogos_do_dia['Predict_winner_home'] == 0) & (jogos_do_dia['Predict_winner_away'] == 1)].copy()
    jogos_do_dia_home['Predict_winner_home'] = "Home"
    jogos_do_dia_away = jogos_do_dia[(jogos_do_dia['Predict_winner_away'] == 0) & (jogos_do_dia['Predict_winner_home'] == 1)].copy()
    jogos_do_dia_away['Predict_winner_away'] = "Away"
    jogos_do_dia_final_home = jogos_do_dia_home[['League', 'Date', 'TIME', 'HomeTeam', 'AwayTeam','Predict_winner_home','probabilities_best_away']]
    jogos_do_dia_final_home = jogos_do_dia_final_home.rename(
        columns={'Predict_winner_home': 'Predict_winner', 'probabilities_best_away': 'prediction_confidence'}
    )
    jogos_do_dia_final_away = jogos_do_dia_away[['League', 'Date', 'TIME', 'HomeTeam', 'AwayTeam','Predict_winner_away','probabilities_best_home']]
    jogos_do_dia_final_away = jogos_do_dia_final_away.rename(
        columns={'Predict_winner_away': 'Predict_winner', 'probabilities_best_home': 'prediction_confidence'}
    )
    
    jogos_do_dia = pd.concat([jogos_do_dia_final_home, jogos_do_dia_final_away], ignore_index=True)
    # Converter e ordenar por data e hora
    jogos_do_dia['datetime'] = pd.to_datetime(jogos_do_dia['Date'] + ' ' + jogos_do_dia['TIME'], dayfirst=True)
    jogos_do_dia = jogos_do_dia.sort_values('datetime').drop('datetime', axis=1)
    return jogos_do_dia  # Retorne o DataFrame final

# 3. Interface do Usuário
def main():
    st.title('Poisson com aprendizado de máquina')
    st.subheader('Jogos do dia')
    
    # Carrega modelos e dados uma vez
    model_gols, model_winner, model_winner_home, model_winner_away = load_models()
    ligas_dicionario = load_league_mapping()
    times_dicionario = load_team_mapping()
    
    # Processa os dados principais
    df = process_main_data(model_gols, model_winner, model_winner_home, model_winner_away, ligas_dicionario, times_dicionario)
    df['Date'] = pd.to_datetime(df['Date']).dt.date  # Convertendo para date (sem hora)
    # Filtros na sidebar
    with st.sidebar:
        st.header('⚙️ Filtros')
        ligas = st.multiselect('Ligas', options=df['League'].unique(), default=df['League'].unique())
        
        date_col = pd.to_datetime(df['Date'])
        min_date, max_date = date_col.min(), date_col.max()
        
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
            (df['Date'] >= date_range[0]) &
            (df['Date'] <= date_range[1]) &
            (df['prediction_confidence'] >= conf_range)
        )
        filtered_df = df[mask]
    else:
        st.warning("Selecione um intervalo de datas válido")
        filtered_df = df
    
    # Exibir resultados
    st.dataframe(filtered_df)

if __name__ == "__main__":
    main()
