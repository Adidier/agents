"""
AC Agent

This agent manage local AC.
"""

import os
import sys
import argparse

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from a2a.serverOllama import run_server
from a2a.core.a2a_ollama import A2AOllama

def main():
    """Run the AC agent server."""
    parser = argparse.ArgumentParser(description="Run AC Agent")
    parser.add_argument("--model", type=str, default="llama3.2:latest", help="The Ollama model to use")
    parser.add_argument("--port", type=int, default=8002, help="The port to run the server on")
    parser.add_argument("--ollama-host", type=str, default="http://localhost:11434", help="The Ollama host URL")
    
    args = parser.parse_args()
    
    # Define the agent's skills
    skills = [
        {
            "id": "analysis",
            "name": "Analysis",
            "description": "Analyzes information for air condicion AC for human house"
        }
    ]
    
    # Create a system prompt to guide the model behavior
    system_prompt = """
    You are a specialized air conditioner Agent that focuses manage correct temperature for humans
    Your responses should:
    - Analyze given information for patterns and implications
    - Identify cause-and-effect relationships
    - Draw logical inferences
    - Explore multiple perspectives
    - Consider implications and consequences
    - Organize thoughts with clear structure and reasoning

    As a AC Agent, your goal is to provide decisition to save energy and mantain a correct temperature for humans.
    """
    
    serverOllama = A2AOllama(
            model=args.model,
            name="Solar AC",
            skills=skills,
            description="An A2A agent that specializes in generating solar resume",
            host=args.ollama_host,
            endpoint=args.ollama_host,
        )
    
    
    # Start the A2A server with the Reasoning Agent
    run_server(
        port=args.port,
        iaAlgorithm=serverOllama
    )


if __name__ == "__main__":
    main() 