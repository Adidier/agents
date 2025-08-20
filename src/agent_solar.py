"""
Solar Agent

This agent  
"""

import os
import sys
import argparse

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from a2a.server import run_server


def main():
    """Run the creative agent server."""
    parser = argparse.ArgumentParser(description="Run Solar Agent")
    parser.add_argument("--model", type=str, default="llama3.2:latest", help="The Ollama model to use")
    parser.add_argument("--port", type=int, default=8003, help="The port to run the server on")
    parser.add_argument("--ollama-host", type=str, default="http://localhost:11434", help="The Ollama host URL")
    
    args = parser.parse_args()
    
    # Define the agent's skills
    skills = [
        {
            "id": "content_generation",
            "name": "Content Generation",
            "description": "Generates anwers for solar agent"
        },
        {
            "id": "reports",
            "name": "Resports",
            "description": "Creates reports of the status of the solar panels"
        },
        {
            "id": "expression",
            "name": "Expressive Writing",
            "description": "Communicates ideas in tables with data"
        }
    ]
    
    # Create a system prompt to guide the model behavior
    system_prompt = """
    You are a specialized Solar Agent that focuses on generating status of my panels.
    Your responses should be:
    - Short
    - Data with conclusion
    As a Solar Agent, your goal is comunicate the esttus of the solar panels.
    """
    
    # Start the A2A server with the Creative Agent
    run_server(
        model=args.model,
        name="Solar Agent",
        description="An A2A agent that specializes in generating solar resume",
        skills=skills,
        port=args.port,
        ollama_host=args.ollama_host
    )


if __name__ == "__main__":
    main() 