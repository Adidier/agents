"""
Weather Agent

This agent provides weather information and forecasts.
"""

import os
import sys
import argparse

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a2a.serverOllama import run_server
from a2a.algorithms.a2a_weather_expert_system import A2AWeatherExpertSystem


def main():
    """Run the weather agent server."""
    
    skills = [
        {
            "id": "weather_forecast",
            "name": "Weather Forecast",
            "description": "Provides weather forecasts and current conditions"
        },
        {
            "id": "weather_analysis",
            "name": "Weather Analysis",
            "description": "Analyzes weather patterns and trends"
        }
    ]
    
    parser = argparse.ArgumentParser(description="Run Weather Agent")
    parser.add_argument("--model", type=str, default="weather-expert-system", help="The model type to use")
    parser.add_argument("--port", type=int, default=8003, help="The port to run the server on")
    parser.add_argument("--ollama-host", type=str, default="http://localhost:11434", help="The Ollama host URL")
    
    args = parser.parse_args()
    
    # Create the weather expert system agent
    weather_agent = A2AWeatherExpertSystem(
        model=args.model,
        name="Weather Agent",
        skills=skills,
        description="An A2A agent that specializes in providing weather information and forecasts",
        host=args.ollama_host,
        endpoint=args.ollama_host,
    )
    
    # Start the A2A server with the Weather Agent
    run_server(
        port=args.port,
        iaAlgorithm=weather_agent
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