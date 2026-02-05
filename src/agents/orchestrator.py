"""
Multi-Agent Example - Orchestrator

This module coordinates multiple specialized A2A agents to complete complex tasks.
"""

import os
import sys
import argparse
import time
from typing import Dict, Any, List

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a2a.client import A2AClient


class AgentOrchestrator:
    """
    Orchestrator for coordinating multiple A2A agents.
    """
    
    def __init__(self, **endpoints):
        """
        Initialize the orchestrator with dynamic endpoints.
        
        Args:
            **endpoints: Variable keyword arguments for agent endpoints
                        (e.g., ac_endpoint="http://localhost:8001")
        """
        self.clients = {}
        
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
    
    def _extract_content(self, response: Dict[str, Any]) -> str:
        """Extract text content from an agent response."""
        if not response:
            return "No content available"
        
        if "message" in response:
            if response["message"] and "parts" in response["message"]:
                for part in response["message"]["parts"]:
                    if part.get("type") == "text" and part.get("content"):
                        return part["content"]
        
        return str(response)  # Fallback
    
    def process_topic(self, topic: str) -> Dict[str, str]:
        """Process a topic by querying all connected agents dynamically."""
        
        # Define prompts for each agent dynamically
        prompts = {
            "ac": "Provide current temperature of the AC. Include important data points, historical context, and current state.",
            "solar": "Provide current status of the Solar panels. Include important data points.",
            "weather": "Provide current status of the Weather. Include important data points."
        }
        
        responses = {}
        
        # Send requests to all agents dynamically
        for i, (agent_name, client) in enumerate(self.clients.items(), 1):
            if agent_name in prompts:
                prompt = prompts[agent_name]
            else:
                prompt = f"Provide information about {agent_name}."
            
            print(f"\n{i}. Gathering information from {agent_name.capitalize()} Agent...")
            agent_response = client.chat(prompt)
            
            # Validate that agent_response is not None or empty
            if not agent_response:
                print(f"Warning: Empty response from {agent_name.capitalize()} Agent")
                content = f"No response from {agent_name.capitalize()} Agent"
            else:
                content = self._extract_content(agent_response)
            
            responses[agent_name] = content
            print(content)
        
        return responses
    
    def generate_report(self, topic: str, responses: Dict[str, str]) -> str:
        """
        Generate a comprehensive report based on agent responses.
        
        Args:
            topic: The original topic
            responses: Responses from each agent
            
        Returns:
            Formatted report
        """
        # Send the combined insights to the creative agent for final coherent presentation
        synthesis_prompt = f"""
        Create a comprehensive, well-structured article about {topic} using the following three components:
        
        1. INTRODUCTION:
        {responses['creative']}
        
        2. FACTS AND CONTEXT:
        {responses['knowledge']}
        
        3. ANALYSIS AND IMPLICATIONS:
        {responses['reasoning']}
        
        Synthesize these into a cohesive, engaging article with appropriate sections and a conclusion.
        Make the transitions between sections smooth and natural.
        """
        
        print("\n4. Synthesizing final report...")
        synthesis_response = self.creative_client.chat(synthesis_prompt)
        synthesis_content = self._extract_content(synthesis_response)
        
        return f"# Comprehensive Analysis: {topic.title()}\n\n{synthesis_content}"


def main():

    parser = argparse.ArgumentParser(description="A2A Multi-Agent Orchestrator")
    # parser.add_argument("--ac-endpoint", type=str, default="http://localhost:8001", help="AC Agent endpoint")
    parser.add_argument("--solar-endpoint", type=str, default="http://localhost:8002", help="Solar Agent endpoint")
    

    args = parser.parse_args()
    
    # Dynamically create endpoint arguments
    endpoints = {}
    for key, value in vars(args).items():
        if key.endswith("_endpoint"):
            endpoints[key] = value
    
    # Create the orchestrator with dynamic endpoints
    orchestrator = AgentOrchestrator(**endpoints)
    
    # Process the topic
    responses = orchestrator.process_topic("")
    
    # # Generate and print the final report
    # report = orchestrator.generate_report(args.topic, responses)
    
    # print("\n" + "="*80)
    # print("\nFINAL REPORT:\n")
    # print(report)
    # print("\n" + "="*80)
    
    # # Save the report to a file
    # filename = f"{args.topic.replace(' ', '_').lower()}_report.md"
    # with open(filename, "w") as f:
    #     f.write(report)
    
    # print(f"\nReport saved to {filename}")


if __name__ == "__main__":
    main()
