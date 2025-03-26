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
st.subTitle('Jogos do dia')
