'''
Created on 

Course work: 

@author: Tact Team

'''

# Import necessary modules
import warnings

from kats.consts import TimeSeriesData
from kats.models.holtwinters import HoltWintersParams, HoltWintersModel
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(
    page_title              = 'Learning Entries Prediction 2.0', 
    # page_icon = favicon, 
    layout                  = 'wide', 
    initial_sidebar_state   = 'auto'
)

CSV_FILEPATH = 'data-migrated.csv'

warnings.simplefilter(action = 'ignore')

def get_base_data():

  df = pd.read_csv(CSV_FILEPATH, 
    usecols = [
        'date', 
        'Migrated-entities-count'
    ]
  )

  return df

base_df = get_base_data()

def get_predicted_df(days):

    indi_new = base_df

    if(len(indi_new) == 0):
        empty_df = pd.DataFrame()
        return empty_df

    indi_new.plot()

    indi_new_graph = indi_new.groupby(['date']).sum()
    indi_new_graph.reset_index(level = 0, inplace = True)

    indi_new_graph.columns  = ['time', 'value']
    indi_new_graph          = TimeSeriesData(indi_new_graph)

    params = HoltWintersParams(
                trend             = "add",
                damped            = False,
                seasonal          = "add",
                seasonal_periods  = 12,
            )

    model = HoltWintersModel(
        data    = indi_new_graph, 
        params  = params
    )

    model.fit()

    fcst = model.predict(steps = days, alpha = 0.1)

    fcst = fcst.drop(['fcst_lower', 'fcst_upper'], axis = 1)
    fcst = fcst.rename(columns = {"fcst" : "value"})

    indi_new_graph = indi_new_graph.to_dataframe()
    data = indi_new_graph.append(fcst)

    return data


def tact_start():

    st.title('Data Migration Prediction 1.0')

    if st.button('View Current Data'):
        
        df = base_df

        basic_chart = alt.Chart(df).mark_line().encode(
            x   = 'date',
            y   = 'Migrated-entities-count'
        )

        st.altair_chart(basic_chart)

    # if st.button("Predict by Days"):

    days = st.slider("Enter Number of Days")

    # 

    if st.button('View Forecast for Given days'):

        df = get_predicted_df(days)

        total = df['value'].sum()

        st.info(f"Total Predicted data migrated is {total} in the next {days} days")

        basic_chart = alt.Chart(df).mark_line().encode(
            x   = 'time',
            y   = 'value'
        )

        st.altair_chart(basic_chart)

    target_data_migrated = int(st.text_input("Enter Target data Migrated"))

    if st.button('View Forecast for a given target amount of data'):

        df = get_predicted_df(100)

        total = 0

        for i in range(10000):
            total += df['value'][i]
            if total > target_data_migrated:
                count = i
                df = df [:len(base_df)+i]
                break

        st.info(f"Total Predicted data migrated is {total} in the next {count} days")

        basic_chart = alt.Chart(df).mark_line().encode(
            x   = 'time',
            y   = 'value'
        )

        st.altair_chart(basic_chart)



if __name__ == '__main__':
    tact_start()
