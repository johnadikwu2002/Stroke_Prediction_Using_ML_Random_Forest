import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from sklearn.preprocessing import LabelEncoder


def app():
    st.title(" Predicting Stroke likelihood :orange[Dashboard] :bar_chart:")
    st.write("You are in 'dashboard' page.")

    df = pd.read_csv(r"healthcare-dataset-stroke-data.csv")  # read the csv file using pandas and binds the heading names above
    # st.write(df)

    # Replace missing data in bmi with the mean
    df['bmi'].fillna(df['bmi'].mean(), inplace=True)

    # Explicitly specify column names for categorical variables
    cat_cols = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'stroke']

    # Explicitly specify column names for continuous variables
    cont_cols = ['age', 'avg_glucose_level', 'bmi']

    # Select only the categorical variables from the DataFrame
    cat_cols_correlation = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
                            'Residence_type', 'smoking_status', 'stroke', 'age', 'avg_glucose_level', 'bmi']
    df_cat = df[cat_cols]

    # Select only the continuous variables from the DataFrame
    df_cont = df[cont_cols]

    # stroke pie chart
    # Define custom colors
    colors = ['#FFA500', '#6AC0F8']
    table2 = px.pie(
        df,
        names='stroke',
        color='stroke',
        color_discrete_sequence=colors,  # Set custom colors
        template='ggplot2',
        title="Stroke Percentage"
    )

    # Layout (Sidebar)
    st.sidebar.markdown("## Target Variable")
    # Show pie chart on Streamlit app
    st.sidebar.plotly_chart(table2, use_container_width=True)
    st.sidebar.markdown("## Settings")
    cat_selected = st.sidebar.selectbox('Categorical Variables', cat_cols)
    cont_selected = st.sidebar.selectbox('Continuous Variables', cont_cols)
    cont_multi_selected = st.sidebar.multiselect('Correlation Matrix', cat_cols_correlation,
        default=cat_cols_correlation)

    # Categorical Variable Bar Chart in Content
    # Group the data by the selected categorical variable and stroke, and count the number of samples in each group
    df_cat_counts = df_cat.groupby([cat_selected, 'stroke']).size().reset_index(name='count')

    # Split the data into two DataFrames, one for stroke=0 and one for stroke=1
    df_cat_counts_0 = df_cat_counts[df_cat_counts['stroke'] == 0]
    df_cat_counts_1 = df_cat_counts[df_cat_counts['stroke'] == 1]

    # Create a bar chart using Plotly
    fig1 = go.Figure(data=[
        go.Bar(name='stroke=0', x=df_cat_counts_0[cat_selected], y=df_cat_counts_0['count']),
        go.Bar(name='stroke=1', x=df_cat_counts_1[cat_selected], y=df_cat_counts_1['count'])
    ])

    # Customize the layout of the chart
    fig1.update_layout(
        height=400,
        width=500,
        margin={'l': 50, 'r': 50, 't': 50, 'b': 50},
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        barmode='stack',
        xaxis_title='Category',
        yaxis_title='Count'
    )

    # Continuous Variable Distribution in Content
    li_cont0 = df[df['stroke'] == 0][cont_selected].values.tolist()
    li_cont1 = df[df['stroke'] == 1][cont_selected].values.tolist()

    cont_data = [li_cont0, li_cont1]
    group_labels = ['stroke=0', 'stroke=1']

    fig2 = ff.create_distplot(cont_data, group_labels,
        show_hist=False,
        show_rug=False)
    fig2.update_layout(height=400,
        width=500,
        margin={'l': 20, 'r': 20, 't': 0, 'b': 0},
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99),
        xaxis_title='Distribution',
        yaxis_title='Measure'
    )

    # Correlation Matrix in Content
    class MultiColumnLabelEncoder:
        def __init__(self, columns=None):
            self.columns = columns  # array of column names to encode

        def fit(self, X, y=None):
            self.encoders = {}
            columns = X.columns if self.columns is None else self.columns
            for col in columns:
                self.encoders[col] = LabelEncoder().fit(X[col])
            return self

        def transform(self, X):
            output = X.copy()
            columns = X.columns if self.columns is None else self.columns
            for col in columns:
                output[col] = self.encoders[col].transform(X[col])
            return output

        def fit_transform(self, X, y=None):
            return self.fit(X, y).transform(X)

        def inverse_transform(self, X):
            output = X.copy()
            columns = X.columns if self.columns is None else self.columns
            for col in columns:
                output[col] = self.encoders[col].inverse_transform(X[col])
            return output

    multi = MultiColumnLabelEncoder(columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
    df = multi.fit_transform(df)

    # Compute correlation matrix
    corr = df[cont_multi_selected].corr().round(3)

    # Correlation Matrix in Content
    df_corr = df[cont_multi_selected].corr()
    fig_corr = go.Figure([go.Heatmap(z=df_corr.values,
        x=df_corr.index.values,
        y=df_corr.columns.values)])
    fig_corr.update_layout(height=300,
        width=1200,
        margin={'l': 20, 'r': 20, 't': 0, 'b': 0})

    # Inverse transform the encoded dataframe
    original_df = multi.inverse_transform(df)

    # Layout (Content)
    left_column, right_column = st.columns(2)
    left_column.subheader('Categorical Variable Distribution: ' + cat_selected)
    right_column.subheader('Continuous Variable Distribution: ' + cont_selected)
    left_column.plotly_chart(fig1)
    right_column.plotly_chart(fig2)
    st.subheader('Correlation Matrix')
    st.plotly_chart(fig_corr)

    st.markdown("### Dataset Statistics")
    # Display summary statistics with full width
    st.write(df.describe(), width=1000)

    df = pd.read_csv(r"healthcare-dataset-stroke-data.csv")  # read the csv file using pandas and binds the heading names above
