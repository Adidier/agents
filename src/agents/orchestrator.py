"""
Multi-Agent Example - Orchestrator

This module coordinates multiple specialized A2A agents to complete complex tasks.
Includes continuous monitoring with traffic light supervision system.
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
from typing import Dict, Any, List

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a2a.client import A2AClient


class AgentOrchestrator:
    """
    Orchestrator for coordinating multiple A2A agents.
    """
    
    def __init__(self, output_file: str = "orchestrator_data.json", **endpoints):
        """
        Initialize the orchestrator with dynamic endpoints.
        
        Args:
            output_file: Path to the JSON file where data will be saved
            **endpoints: Variable keyword arguments for agent endpoints
                        (e.g., ac_endpoint="http://localhost:8001")
        """
        self.clients = {}
        self.output_file = output_file
        self.history = []
        
        # Create clients dynamically for each endpoint
        for name, endpoint in endpoints.items():
            agent_name = name.replace("_endpoint", "")
            self.clients[agent_name] = A2AClient(endpoint)
        
        # Verify connections and discover capabilities
        print("Connecting to agents...")
        self._verify_connections()
    
    def _verify_connections(self):
        """Verify connections to all agents and print their capabilities."""
        try:
            for agent_name, client in self.clients.items():
                card = client.discover_agent()
                print(f"✓ Connected to {agent_name.capitalize()} Agent: {card['name']}")
                print(f"  Skills: {', '.join(skill['name'] for skill in card['skills'])}")
            
            print("\nAll agents connected successfully.\n")
        except Exception as e:
            print(f"Error connecting to agents: {e}")
            print("Please ensure all agent servers are running.")
            sys.exit(1)
    
    def _save_to_json(self):
        """Save the collected history to a JSON file."""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "total_iterations": len(self.history),
                        "agents": list(self.clients.keys()),
                        "last_updated": datetime.now().isoformat()
                    },
                    "history": self.history
                }, f, indent=2, ensure_ascii=False)
            print(f"  💾 Saved to {self.output_file}")
        except Exception as e:
            print(f"❌ Error saving data to JSON: {e}")
    
    def process_topic(self, topic: str) -> Dict[str, Any]:
        """Process a topic by querying all connected agents dynamically.
        
        Returns:
            Dict with raw responses for each agent
        """
        
        # Define prompts for each agent dynamically
        prompts = {
            "ac": "Provide current temperature of the AC. Include important data points, historical context, and current state.",
            "solar": "Provide current status of the Solar panels. Include important data points.",
            "weather": "Provide current status of the Weather. Include important data points."
        }
        
        raw_responses = {}
        
        # Send requests to all agents dynamically
        for agent_name, client in self.clients.items():
            if agent_name in prompts:
                prompt = prompts[agent_name]
            else:
                prompt = f"Provide information about {agent_name}."
            
            print(f"  ✓ Querying {agent_name.capitalize()} Agent...", end=" ")
            try:
                agent_response = client.chat(prompt)
                raw_responses[agent_name] = agent_response
                print("OK")
            except Exception as e:
                print(f"ERROR: {e}")
                raw_responses[agent_name] = {"error": str(e)}
        
        return {"raw": raw_responses}


def main():
    parser = argparse.ArgumentParser(description="A2A Multi-Agent Orchestrator")
    parser.add_argument("--solar-endpoint", type=str, default="http://localhost:8002", help="Solar Agent endpoint")
    parser.add_argument("--weather-endpoint", type=str, default="http://localhost:8004", help="Weather Agent endpoint")
    parser.add_argument("--output", type=str, default="orchestrator_data.json", help="Output JSON file for monitoring data")
    
    args = parser.parse_args()
    
    # Dynamically create endpoint arguments
    endpoints = {}
    for key, value in vars(args).items():
        if key.endswith("_endpoint"):
            endpoints[key] = value
    
    # Create the orchestrator with dynamic endpoints and output file
    orchestrator = AgentOrchestrator(output_file=args.output, **endpoints)
    
    # Continuous monitoring loop
    print("\n" + "="*80)
    print("🚀 Orchestrator monitoring started")
    print("📊 Data will be saved to:", args.output)
    print("🔄 Refresh rate: 10 seconds")
    print("💡 Tip: Run dashboard.py to visualize real-time data")
    print("Press Ctrl+C to stop")
    print("="*80)
    
    try:
        iteration = 0
        while True:
            iteration += 1
            timestamp = datetime.now().isoformat()
            print(f"\n[{time.strftime('%H:%M:%S')}] Iteration #{iteration}")
            
            # Process and get status from all agents
            responses = orchestrator.process_topic("")
            
            # Store iteration data
            iteration_data = {
                "iteration": iteration,
                "timestamp": timestamp,
                "agents": responses["raw"]
            }
            orchestrator.history.append(iteration_data)
            
            # Save to JSON every iteration
            orchestrator._save_to_json()
            
            # Wait 10 seconds before next iteration
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("Monitoring stopped by user")
        print("="*80)
        # Save final data before exiting
        orchestrator._save_to_json()
        print(f"\n✅ Final data saved. Total iterations: {len(orchestrator.history)}")


if __name__ == "__main__":
    main()
