"""
Solar Agent

This agent  
"""

import os
import sys
import argparse

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from a2a.serverOllama import run_server
from a2a.core.a2a_expert_system import A2AExpertSystem

def main():
    
    skills = [
        {
            "id": "analysis",
            "name": "Analysis",
            "description": "Analyzes information for air condicion AC for human house"
        }
    ]
    """Run the creative agent server."""
    parser = argparse.ArgumentParser(description="Run Solar Agent")
    parser.add_argument("--model", type=str, default="llama3.2:latest", help="The Ollama model to use")
    parser.add_argument("--port", type=int, default=8003, help="The port to run the server on")
    parser.add_argument("--ollama-host", type=str, default="http://localhost:11434", help="The Ollama host URL")
    
    args = parser.parse_args()
    
    
    serverOllama = A2AExpertSystem(
            model=args.model,
            name="Solar Agent",
            skills=skills,
            description="An A2A agent that specializes in generating solar resume",
            host=args.ollama_host,
            endpoint=args.ollama_host,
        )
    
    # Start the A2A server with the Creative Agent
    run_server(
        port=args.port,
        iaAlgorithm=serverOllama
    )


if __name__ == "__main__":
    main() 