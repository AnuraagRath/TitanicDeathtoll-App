import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("""
TitanicDeathToll App
by Anuraag Rath

""")

st.sidebar.header('Filter Preferences')

def openData():
    url = "titanic.csv"
    csvopen = pd.read_csv(url)
    return csvopen
csvopen = openData()

uniPclass = sorted(csvopen.Pclass.unique())
selectedPClass = st.sidebar.multiselect('Pclass', uniPclass, uniPclass)

sorted_unique_sex = sorted(csvopen.Sex.unique())
selected_sex = st.sidebar.multiselect('Sex', sorted_unique_sex, sorted_unique_sex)

unique_pos = ['1', '0']
selected_pos = st.sidebar.multiselect('Survived', unique_pos, unique_pos)

selectedPeople = csvopen[(csvopen.Pclass.isin(selectedPClass)) & 
                         (csvopen.Sex.isin(selected_sex)) & 
                         (csvopen.Survived.isin(selected_pos))]

st.header('Titanic Data')
st.write('DataFrameDimension: ' + str(selectedPeople.shape[0]) + ' rows and ' + str(selectedPeople.shape[1]) + ' columns.')
st.dataframe(selectedPeople)


if st.button('Heatmap'):
    st.header('Intercorrelation Matrix')
    df_selected_team = pd.read_csv('titanic.csv')

    corr = df_selected_team.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot(f, ax)
