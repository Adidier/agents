"""
Solar Agent

This agent provides solar information and forecasts.
"""

import os
import sys
import argparse

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a2a.serverOllama import run_server
from a2a.algorithms.a2a_lstm import A2ALSTM


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
    
    args = parser.parse_args()
    
    # Create the solar expert system agent
    solar_agent = A2ALSTM(
        model=args.model,
        name="Solar Agent",
        skills=skills,
        description="An A2A agent that specializes in providing solar information and forecasts",
        host=args.ollama_host,
        endpoint=args.ollama_host,
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
        iaAlgorithm=solar_agent
    )


if __name__ == "__main__":
    main() 



# """
# Weather Agent

# This agent provides a weather conclusions.
# """

# import os
# import sys
# import argparse

# # Add the parent directory to sys.path BEFORE importing a2a
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from a2a.serverOllama import run_server
# from a2a.core.a2a_ollama import A2AOllama

# def main():
#     """Run the weather agent server."""
#     parser = argparse.ArgumentParser(description="Run weather Agent")
#     parser.add_argument("--model", type=str, default="llama3.2:latest", help="The Ollama model to use")
#     parser.add_argument("--port", type=int, default=8001, help="The port to run the server on")
#     parser.add_argument("--ollama-host", type=str, default="http://localhost:11434", help="The Ollama host URL")
    
#     args = parser.parse_args()
    
#     # Define the agent's skills
#     skills = [
#         {
#             "id": "research",
#             "name": "Research",
#             "description": "Provides weather information on local"
#         },
#         {
#             "id": "fact_check",
#             "name": "Fact Checking",
#             "description": "Verifies claims against known facts"
#         }
#     ]
    
#     # Create a system prompt to guide the model behavior
#     system_prompt = """
#     You are a specialized weather Agent that focuses on providing local information.
#     Your responses should be:
#     - Based on weather information
#     - Well-structured with clear sections
#     - Comprehensive yet concise
#     - Focused on verifiable data and statistics
#     - Neutral in tone

#     As a weather Agent, your goal is to provide accurate information without speculation or opinion.
#     """
    
#     serverOllama = A2AOllama(
#             model=args.model,
#             name="Solar Wheather",
#             skills=skills,
#             description="An A2A agent that specializes in generating solar resume",
#             host=args.ollama_host,
#             endpoint=args.ollama_host,
#         )
    
    
#     # Start the A2A server with the Reasoning Agent
#     run_server(
#         port=args.port,
#         iaAlgorithm=serverOllama
#     )


# if __name__ == "__main__":
#     main() 