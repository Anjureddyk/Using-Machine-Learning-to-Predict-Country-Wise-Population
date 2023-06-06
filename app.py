import pandas as pd
import pickle
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Load the preprocessed data
data = pd.read_csv("pre_processed_data.csv")

# Load the trained model
with open('random_forest.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Load the country dictionary
with open('country_dict.pkl', 'rb') as file:
    country_dict = pickle.load(file)

# Create the Dash app
app = dash.Dash(__name__)

# Set the app layout
app.layout = html.Div(
    children=[
        html.H1('Population Prediction App'),
        html.Label('Country (or dependency)'),
        dcc.Dropdown(
            id='input-country',
            options=[{'label': f"{country} ({number})", 'value': number} for country, number in country_dict.items()],
            placeholder='Select a country'
        ),
        html.Label('Yearly Change'),
        dcc.Input(id='input-yearly-change', type='number', placeholder='Enter yearly change'),
        html.Label('Net Change'),
        dcc.Input(id='input-net-change', type='number', placeholder='Enter net change'),
        html.Label('Density (P/Km²)'),
        dcc.Input(id='input-density', type='number', placeholder='Enter density'),
        html.Label('Land Area (Km²)'),
        dcc.Input(id='input-land-area', type='number', placeholder='Enter land area'),
        html.Label('Migrants (net)'),
        dcc.Input(id='input-migrants', type='number', placeholder='Enter migrants'),
        html.Label('Fertility Rate'),
        dcc.Input(id='input-fertility-rate', type='number', placeholder='Enter fertility rate'),
        html.Label('Median Age'),
        dcc.Input(id='input-median-age', type='number', placeholder='Enter median age'),
        html.Label('Urban Pop %'),
        dcc.Input(id='input-urban-pop', type='number', placeholder='Enter urban population percentage'),
        html.Label('World Share'),
        dcc.Input(id='input-world-share', type='number', placeholder='Enter world share'),
        html.Button('Predict', id='button-predict', n_clicks=0),
        html.Div(id='output-prediction')
    ]
)

# Define the callback function to handle prediction
@app.callback(
    Output('output-prediction', 'children'),
    [Input('button-predict', 'n_clicks')],
    [dash.dependencies.State('input-country', 'value'),
     dash.dependencies.State('input-yearly-change', 'value'),
     dash.dependencies.State('input-net-change', 'value'),
     dash.dependencies.State('input-density', 'value'),
     dash.dependencies.State('input-land-area', 'value'),
     dash.dependencies.State('input-migrants', 'value'),
     dash.dependencies.State('input-fertility-rate', 'value'),
     dash.dependencies.State('input-median-age', 'value'),
     dash.dependencies.State('input-urban-pop', 'value'),
     dash.dependencies.State('input-world-share', 'value')]
)
def predict_population(n_clicks, country, yearly_change, net_change, density, land_area, migrants, fertility_rate,
                       median_age, urban_pop, world_share):
    if n_clicks > 0:
        # Perform the prediction using the loaded model
        encoded_country = label_encoder.transform([country])[0]
        prediction = model.predict([[encoded_country, yearly_change, net_change, density, land_area, migrants,
                                     fertility_rate, median_age, urban_pop, world_share]])

        # Display the predicted population
        return f'Predicted Population for {country}: {prediction[0]}'

    return ''

    return ''


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
