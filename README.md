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


chmod +x launch_agents.sh


./launch_agents.sh
