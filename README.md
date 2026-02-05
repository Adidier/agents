LSTM PV prediction

Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train (example):

```bash
python src/models/train_lstm_pv.py --data examples/LSTM3miso/completeDataF.csv --epochs 30
```

3. Predict next value:

```bash
python src/models/predict_lstm_pv.py --model models/lstm_pv.h5 --scaler models/scaler.pkl --data examples/LSTM3miso/completeDataF.csv
```

Notes

- The scripts follow the preprocessing from `examples/LSTM3miso/misopredict.ipynb` (default uses columns 2:8 and time_steps=24). Adjust arguments if your CSV differs.
- Model and scaler are saved under `models/` by default.



# agents

## Weather Agent con Mock Server

El Weather Agent puede funcionar con o sin conexión a internet:

```bash
# Modo normal (intenta NASA API, fallback a mock)
python src/agents/weather.py

# Modo simulación (siempre usa mock)
./launch_mock_weather.sh  # Terminal 1: Servidor mock
python src/agents/weather.py --mock  # Terminal 2: Agente
```

Ver [documentación completa del Mock Server](docs/MOCK_WEATHER_SERVER.md)

## Otros Agentes

```bash
python src/agents/solar.py
python src/agents/weather.py --mock 
python src/agents/dashboard.py 
python tools/mock_weather_server.py 
``` 



pip install flask