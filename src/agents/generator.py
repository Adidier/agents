"""
Solar Agent

This agent provides solar information and forecasts with automatic supervision.
Includes traffic light classification system for PV performance monitoring.
"""

import os
import sys
import argparse
import random

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a2a.serverOllama import run_server
from algorithms.lstm.lstm import A2ALSTM
from algorithms.lstm.pv_simulator import PVSimulator


def main():
    """Run the solar agent server."""
    
    skills = [
        {
            "id": "solar_forecast",
            "name": "Solar Forecast",
            "description": "Provides solar forecasts and current conditions"
        },
        {
            "id": "solar_analysis",
            "name": "Solar Analysis",
            "description": "Analyzes solar patterns and trends"
        }
    ]
    
    parser = argparse.ArgumentParser(description="Run Solar Agent")
    parser.add_argument("--model", type=str, default="solar-expert-system", help="The model type to use")
    parser.add_argument("--port", type=int, default=8002, help="The port to run the server on")
    parser.add_argument("--ollama-host", type=str, default="http://localhost:11434", help="The Ollama host URL")
    parser.add_argument("--auto-simulate", action="store_true", default=True, help="Auto-simulate real power for supervision")
    parser.add_argument("--no-simulate", dest="auto_simulate", action="store_false", help="Disable auto-simulation")
    parser.add_argument("--orchestrator-url", type=str, default="http://localhost:8001", help="Orchestrator registry URL for auto-registration")
    
    args = parser.parse_args()
    
    # Create the solar expert system agent
    solar_agent = A2ALSTM(
        model=args.model,
        name="Solar Generator Agent",
        skills=skills,
        description="An A2A agent that specializes in providing solar information and forecasts",
        host=args.ollama_host,
        endpoint=args.ollama_host,
        auto_simulate=args.auto_simulate,
    )
    
    # Attempt to load a pre-trained LSTM model (optional).
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_model_path = os.path.join(repo_root, "models", "lstm_pv.h5")
    default_scaler_path = os.path.join(repo_root, "models", "scaler.pkl")
    if os.path.exists(default_model_path) and os.path.exists(default_scaler_path):
        try:
            solar_agent.load_lstm(default_model_path, default_scaler_path)
            print(f"Loaded LSTM model from {default_model_path}")
        except Exception as e:
            print(f"Failed to load LSTM model: {e}")
    else:
        print("No pre-trained LSTM model found (models/lstm_pv.h5). To train, run src/models/train_lstm_pv.py")

    # Start the A2A server with the Solar Agent
    run_server(
        port=args.port,
        iaAlgorithm=solar_agent,
        orchestrator_url=args.orchestrator_url
    )


if __name__ == "__main__":
    main() 


