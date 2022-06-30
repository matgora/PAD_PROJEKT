#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
from dash import Dash, html, dcc, Input, Output, dash_table
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


app = Dash(__name__)
df = pd.read_csv('preprocessed_data.tsv', sep='\t')
numeric = df.select_dtypes(include=np.number).columns.tolist()
outfield = df[df['Position']!='Goalkeeper']
goalkeepers = df[df['Position']=='Goalkeeper']
outfield = outfield.drop(columns=['Goals conceded', 'Clean sheets'])
goalkeepers = goalkeepers.drop(columns=['Own goals', 'Penalty goals', 'Minutes per goal'])
numeric_outfield = outfield.select_dtypes(include=np.number).columns.tolist()
numeric_goalkeepers = goalkeepers.select_dtypes(include=np.number).columns.tolist()


def get_descrption(data):
    df_d = pd.DataFrame()
    df_d['mean'] = data.mean(skipna=True, numeric_only=True)
    df_d['median'] = data.median(skipna=True, numeric_only=True)
    df_d['min'] = data.min(skipna=True, numeric_only=True)
    df_d['max'] = data.max(skipna=True, numeric_only=True)
    df_d['skew'] = data.skew(skipna=True, numeric_only=True)
    df_d['std'] = data.std(skipna=True, numeric_only=True)
    df_d = df_d.reset_index()
    df_d = df_d.rename(columns={'index': 'Atrybut'})
    return df_d


app.layout = html.Div([
    html.H1('Analiza wartosci pilkarzy na podstawie danych z transfermarkt.com'),
    html.Br(),
    html.H2(children='Analiza eksploracyjna'),
    html.H3(['Wykres punktowy dla dwoch wybranych atrybutow']),
    html.H4((
        "Z wykresow punktowych mozemy przede wszystkim zobaczyc liniowa zaleznosc "
        "miedzy atrybutami 'Value before' i 'Value after', czyli wartosciami "
        "pilkarzy przed i po sezonie 2020/21. Widoczna jest rowniez "
        "duza liczba pilkarzy z niska wartoscia, co potwierdzaja dalej histogramy. "
        "Wartosciami odstajacymi sa topowi pilkarze z najlepszych lig europejskich. ",
        "Przodują oni w statystykach takich jak Gole i Asysty (oraz oczywiscie wartosc)."
        )),
    html.Div([
        html.Div([
            html.Label('Wybor X'),
            dcc.Dropdown(
                numeric,
                'Value before',
                id='xaxis-column-scat'
            ),
            dcc.RadioItems(
                ['Linear', 'Log'],
                'Linear',
                id='xaxis-type-scat',
                inline=True
            )
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            html.Label('Wybor Y'),
            dcc.Dropdown(
                numeric,
                'Value after',
                id='yaxis-column-scat'
            ),
            dcc.RadioItems(
                ['Linear', 'Log'],
                'Linear',
                id='yaxis-type-scat',
                inline=True
            )
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),
    dcc.Graph(id='scatter'),
    html.H3(['Histogram dla wybranego atrybutu']),
    html.H4((
        "Histogramy potwierdzaja skosnosc atrybutow 'Value before' i 'Value after'. "
        "Dodatkowo mozemy zauwazyc skosnosc w atrybutach takich jak 'Goals' czy 'Assists'. "
        "Takie dysproporcje moga wynikac z roznych obowiazkow pilakrzy na boisku, np. "
        "obroncy nie sa odpowiedzialni za zdobywanie bramek."
        "Atrybuty takie jak 'Age' czy 'Height' przypominaja swoim ksztaltem rozklad normalny."
        )),
    dcc.Dropdown(
        numeric,
        'Value after',
        id='xaxis-column-hist'
    ),
    dcc.Graph(id='hist'),
    html.H3('Macierz korelacji'),
    html.H4((
        'Problem podzielilem na dwa podproblemy - bramkarzy i graczy z pola. '
        'Wynika to glownie z roznicy w najwazniejszych statystykach do oceny jakosci sezonu.'
        'Dla bramkarzy licza sie czyste konta i wpuszczone bramki, co nie jest istotne dla zawodnikow z pola. '
        'Niestety strona transfermarkt nie udostepnia statystyk waznych dla obroncow, '
        'takze tutaj musimy tak samo oceniac graczy ze wszystkich pozycji. '
        'Na macierzy korelacji widzimy przede wszystkim duza korelacje miedzy ocena przed i po sezonie. '
        "Skorelowane tez sa ze soba atrybuty 'Appearances' i 'Squad' czy 'Appearances' i 'Minutes played'."
    )),
    html.Div(
        children=[
            dcc.RadioItems(
                ['Goalkeepers', 'Outfield players'],
                'Outfield players',
                id='corr-players',
                inline=True
            ),
            dcc.Dropdown(
                id='corr-cols',
                multi=True,
            ),
            html.Button("SELECT ALL", id="select-all", n_clicks=0),
        ]
    ),
    dcc.Graph(id='corr'),
    html.H3('Statystyki'),
    html.H4((
        "Ponizej widoczna jest tabela ze statystykami pozycyjnymi. Mozemy stad "
        "odczytac przede wszystkim jakie atrybuty sa skosne, co moze wplynac na dzialanie "
        "modelu regresji liniowej."
        )),
    html.Div(
        children=[
            dcc.RadioItems(
                ['Goalkeepers', 'Outfield players'],
                'Outfield players',
                id='stats-players',
                inline=True
            ),
            dash_table.DataTable(id='stats')
        ]
    ),
    html.H2('Modelowanie'),
    html.H4((
        "Atrybutem decyzyjnym jest 'Value after'. Do wyboru daje przewidywanie wartosci "
        "bramkarzy lub graczy z pola jednym z dwoch modeli: Regresora opartego na lasach losowych "
        "oraz regresji liniowej. Dodatkowo mozna wybrac aby dane byly standaryzowane (dla ulatwienia interpretacji "
        "wynikow) lub zlogarytmizowane zostaly atrybuty 'Value after', 'Value before', 'Goals' i 'Assists', "
        "poniewaz te atrybuty wykazuja sie wysoka skosnoscia oraz przewiduje ze moga miec duze znaczenie."
        " Jak pokazuja wyniki, daje to efekt glownie w przypadku "
        "modelu regresji liniowej, gdzie R-squared rosnie o okolo 0.02 dla obu problemow."
    )),
    html.Div(
        children=[
            dcc.RadioItems(
                ['Goalkeepers', 'Outfield players'],
                'Outfield players',
                id='regr-players',
                inline=True
            ),
            dcc.RadioItems(
                ['Random Forest Regressor', 'Linear Regression'],
                'Random Forest Regressor',
                id='regr-model',
                inline=True
            ),
            dcc.Checklist(
                ['Standardize', 'Logarithmize'],
                id='checklist',
                inline=True
            ),
            dcc.Dropdown(
                id='regr-cols',
                multi=True,
            ),
        ]
    ),
    html.Div(
        children=[
            html.H3(id='regr-score'),
            html.H3(id='regr-mse'),
            html.H3(id='regr-rmse'),
            ]
    ),
    dcc.Graph('regr'),
    ])


@app.callback(
    Output('scatter', 'figure'),
    Input('xaxis-column-scat', 'value'),
    Input('yaxis-column-scat', 'value'),
    Input('xaxis-type-scat', 'value'),
    Input('yaxis-type-scat', 'value'),)
def update_scatter(xaxis_column_name, yaxis_column_name,
                   xaxis_type, yaxis_type,):
    fig = px.scatter(df, x=xaxis_column_name, y=yaxis_column_name,
                     hover_name='Name', hover_data=['Current club', 'League Country'])
    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    fig.update_xaxes(title=xaxis_column_name,
                     type='linear' if xaxis_type == 'Linear' else 'log')

    fig.update_yaxes(title=yaxis_column_name,
                     type='linear' if yaxis_type == 'Linear' else 'log')

    return fig


@app.callback(
    Output('hist', 'figure'),
    Input('xaxis-column-hist', 'value'))
def update_hist(xaxis_column_name):
    fig = px.histogram(df, x=xaxis_column_name)
    return fig


@app.callback(
    Output('corr', 'figure'),
    Input('corr-cols', 'value'),
    Input('corr-players', 'value'))
def get_corr(corr_cols, corr_players):
    if corr_players == 'Goalkeepers':
        dff = goalkeepers[corr_cols]
    else:
        dff = outfield[corr_cols]
    fig = px.imshow(dff.corr(), text_auto=True, aspect='auto')
    return fig


@app.callback(Output('corr-cols', 'options'), Input('corr-players', 'value'))
def get_corr_players(corr_players):
    if corr_players == 'Goalkeepers':
        return [{'label': x, 'value': x} for x in numeric_goalkeepers]
    else:
        return [{'label': x, 'value': x} for x in numeric_outfield]


@app.callback(Output("corr-cols", "value"),
              Input("select-all", "n_clicks"),
              Input('corr-players', 'value'))
def select_all(n_clicks, corr_players):
    if corr_players == 'Goalkeepers':
        return numeric_goalkeepers
    else:
        return numeric_outfield


@app.callback(Output('stats', 'data'), Input('stats-players', 'value'))
def get_stats(stats_players):
    if stats_players == 'Goalkeepers':
        return get_descrption(goalkeepers).to_dict('records')
    else:
        return get_descrption(outfield).to_dict('records')


@app.callback(Output('regr-cols', 'options'), Input('regr-players', 'value'))
def get_regr_cols(regr_players):
    if regr_players == 'Goalkeepers':
        return [{'label': x, 'value': x} for x in numeric_goalkeepers]
    else:
        return [{'label': x, 'value': x} for x in numeric_outfield]


@app.callback(Output('regr-cols', 'value'), Input('regr-players', 'value'))
def get_regr_cols(regr_players):
    if regr_players == 'Goalkeepers':
        return ['Value before', 'ranking', 'Goals conceded', 'Clean sheets', 'Age', 'PPG',]
    else:
        return ['Value before', 'ranking', 'Minutes played', 'Goals', 'Assists', 'Age', 'PPG',]


@app.callback(
        Output('regr', 'figure'),
        Output('regr-score', 'children'),
        Output('regr-mse', 'children'),
        Output('regr-rmse', 'children'),
        Input('regr-players', 'value'),
        Input('regr-cols', 'value'),
        Input('regr-model', 'value'),
        Input('checklist', 'value'),)
def get_regression(regr_players, regr_cols, regr_model, checklist):
    if regr_players == 'Goalkeepers':
        dff = goalkeepers.copy()
    else:
        dff = outfield.copy()

    if checklist and 'Logarithmize' in checklist:
        dff[['Value after', 'Value before']] = dff[['Value after', 'Value before']].apply(lambda x: np.log(x + 10**4))
        dff[['Goals', 'Assists']] = dff[['Goals', 'Assists']].apply(lambda x: np.log(x + 1))

    X = dff[regr_cols]
    y = dff['Value after']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if 'Height' in regr_cols:
        median = X_train['Height'].median()
        X_train['Height'] = X_train['Height'].fillna(median)
        X_test['Height'] = X_test['Height'].fillna(median)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    if checklist and 'Standardize' in checklist:
        for col in X_train.columns:
            mean = X_train[col].mean()
            std = X_train[col].std()
            X_train[col] = X_train[col].apply(lambda x: (x-mean)/std)
            X_test[col] = X_test[col].apply(lambda x: (x-mean)/std)

        mean = y_train.mean()
        std = y_train.std()
        y_train = y_train.apply(lambda x: (x-mean)/std)
        y_test = y_test.apply(lambda x: (x-mean)/std)

    if regr_model == 'Linear Regression':
        model = LinearRegression()
    else:
        model = RandomForestRegressor(max_depth=None, random_state=42)

    return do_regression(X_train, X_test, y_train, y_test, model)


def do_regression(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    score = model.score(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    fig = px.scatter(x=y_test, y=y_pred, title='Wykres zaleznosci wartosci atrybutu decyzyjnego i wartosci przewidywanej')
    fig.add_shape(
        type="line", line=dict(dash='dash'),
        x0=y_test.min(), y0=y_test.min(),
        x1=y_test.max(), y1=y_test.max()
    )
    fig.update_xaxes(title='Ground truth')
    fig.update_yaxes(title='Prediction')
    return fig, f"R-squared: {score}", f"MSE: {mse}", f"RMSE: {mse**(1/2)}"

if __name__ == '__main__':
    app.run_server(debug=True)
