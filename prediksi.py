import numpy as np
import pandas as pd
import websocket
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import threading
import time
from dash import Dash, dcc, html, Input, Output
import dash

# Define maximum data points (bisa diubah sesuai kebutuhan)
MAX_DATA_POINTS = 100  # Ubah jumlah data yang ingin disimpan

# Load trained model
model = load_model("bitcoin_model.keras")

# Load and fit scaler
file_path = "bitcoin_gabungan.xlsx"
df = pd.read_excel(file_path)
df_clean = df.drop_duplicates(subset=["timestamp"])
df_clean.loc[:, 'timestamp'] = pd.to_datetime(df_clean['timestamp'])  # Fix warning
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df_clean['close'].values.reshape(-1, 1))

# WebSocket setup
socket = "wss://stream.binance.com:9443/ws/btcusdt@trade"
real_time_prices = []
seq_length = 50
actual_prices = []
predicted_prices = []
time_intervals = []  # Store time in minutes
start_time = time.time()
ws_running = True  # Flag to stop WebSocket

def on_message(ws, message):
    global real_time_prices, time_intervals, actual_prices, predicted_prices, ws_running
    
    if not ws_running:
        ws.close()
        return
    
    data = json.loads(message)
    price = float(data['p'])  # Extract price
    real_time_prices.append(price)
    actual_prices.append(price)
    elapsed_time = (time.time() - start_time) / 60  # Convert to minutes
    time_intervals.append(elapsed_time)
    print(f"Actual Bitcoin Price: {price}")

    # Batasi jumlah data berdasarkan MAX_DATA_POINTS
    if len(actual_prices) >= MAX_DATA_POINTS:
        ws_running = False  # Stop WebSocket
        ws.close()
        print("Maximum data points reached. Stopping WebSocket.")
        return
    
    if len(real_time_prices) >= seq_length:
        input_data = np.array(real_time_prices[-seq_length:]).reshape(-1, 1)
        input_scaled = scaler.transform(input_data)
        input_seq = np.reshape(input_scaled, (1, seq_length, 1))

        prediction = model.predict(input_seq)
        predicted_price = scaler.inverse_transform(prediction)[0][0]
        predicted_prices.append(predicted_price)

        print(f"Predicted Bitcoin Price: {predicted_price}")

def start_websocket():
    ws = websocket.WebSocketApp(socket, on_message=on_message)
    ws.run_forever()

# Start WebSocket in a separate thread
ws_thread = threading.Thread(target=start_websocket)
ws_thread.daemon = True
ws_thread.start()

# Dash App for Interactive Visualization
app = Dash(__name__)

UPDATE_INTERVAL = 5  # dalam detik

app.layout = html.Div([
    html.H1("Real-Time Bitcoin Price Prediction", style={'textAlign': 'center'}),
    dcc.Graph(id='live-graph'),
    dcc.Interval(
        id='interval-component',
        interval=UPDATE_INTERVAL * 1000,  # Konversi ke milidetik
        n_intervals=0
    )
])

@app.callback(
    Output('live-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    if len(actual_prices) > 10 and len(predicted_prices) > 10:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_intervals[-MAX_DATA_POINTS:], y=actual_prices[-MAX_DATA_POINTS:], 
                                 mode='markers+lines', name='Actual Prices', marker=dict(color='blue')))
        fig.add_trace(go.Scatter(x=time_intervals[-MAX_DATA_POINTS:], y=predicted_prices[-MAX_DATA_POINTS:], 
                                 mode='markers+lines', name='Predicted Prices', marker=dict(color='red')))
        fig.update_layout(title='Real-Time Bitcoin Price Prediction (Scatter Plot)',
                          xaxis_title='Time (Minutes)', yaxis_title='Price')
        return fig
    return go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True)
