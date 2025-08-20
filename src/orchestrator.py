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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from a2a.client import A2AClient


class AgentOrchestrator:
    """
    Orchestrator for coordinating multiple A2A agents.
    """
    
    def __init__(
        self,
        ac_endpoint: str = "http://localhost:8001",
        solar_endpoint: str = "http://localhost:8002",
        weather_endpoint: str = "http://localhost:8003",
  
        # creative_endpoint: str = "http://192.168.31.178:8003"
    ):
        """
        Initialize the orchestrator.
        
        Args:
            ac_endpoint: Endpoint for the AC Agent
            wheather_endpoint: Endpoint for the Wheater Agent
            solar_endpoint: Endpoint for the Solar Agent
        """
        self.ac_client = A2AClient(ac_endpoint)
        self.solar_client = A2AClient(solar_endpoint)
        self.weather_client = A2AClient(weather_endpoint)
        
        # Verify connections and discover capabilities
        print("Connecting to agents...")
        self._verify_connections()
    
    def _verify_connections(self):
        """Verify connections to all agents and print their capabilities."""
        try:
            ac_card = self.ac_client.discover_agent()
            print(f"✓ Connected to AC Agent: {ac_card['name']}")
            print(f"  Skills: {', '.join(skill['name'] for skill in ac_card['skills'])}")
            
            solar_card = self.solar_client.discover_agent()
            print(f"✓ Connected to Solar Agent: {solar_card['name']}")
            print(f"  Skills: {', '.join(skill['name'] for skill in solar_card['skills'])}")
            
            weather_card = self.weather_client.discover_agent()
            print(f"✓ Connected to Weather Agent: {weather_card['name']}")
            print(f"  Skills: {', '.join(skill['name'] for skill in weather_card['skills'])}")
            
            print("\nAll agents connected successfully.\n")
        except Exception as e:
            print(f"Error connecting to agents: {e}")
            print("Please ensure all agent servers are running.")
            sys.exit(1)
    
    def _extract_content(self, response: Dict[str, Any]) -> str:
        """Extract text content from an agent response."""
        if "message" in response:
            for part in response["message"]["parts"]:
                if part["type"] == "text":
                    return part["content"]
        
        return str(response)  # Fallback
    
    def process_topic(self, topic: str) -> Dict[str, str]:

        # print(f"Processing topic: '{topic}'")
        
        # Define prompts for each agent
        ac_prompt = f"Provide current temperature of the AC. Include important data points, historical context, and current state."
        solar_prompt = f"Provide current status of the Solar pannels Include important data points,"
        weather_prompt = f"Provide current status of the Wheater Include important data points"
        
        # Send requests to all agents
        print("\n1. Gathering factual information from AC Agent...")
        ac_response = self.ac_client.chat(ac_prompt)
        ac_content = self._extract_content(ac_response)
        print(ac_content)

        print("2. Analyzing implications with Solar Agent...")
        solar_response = self.solar_client.chat(solar_prompt)
        solar_content = self._extract_content(solar_response)
        print(solar_content)

        print("3. Creating engaging narrative with Wheater Agent...")
        weather_response = self.weather_client.chat(weather_prompt)
        weather_content = self._extract_content(weather_response)
        print(weather_content)

        # Return all responses
        return {
            "ac": ac_content,
            "solar": solar_content,
            "whater": weather_content
        }
    
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
    # parser.add_argument("--topic", type=str, required=True, help="The topic to analyze")
    parser.add_argument("--ac-endpoint", type=str, default="http://localhost:8001", help="ac Agent endpoint")
    parser.add_argument("--solar-endpoint", type=str, default="http://localhost:8002", help="solar Agent endpoint")
    parser.add_argument("--weather-endpoint", type=str, default="http://localhost:8003", help="wheater Agent endpoint")
    
    args = parser.parse_args()
    
    # Create the orchestrator
    orchestrator = AgentOrchestrator(
        ac_endpoint=args.ac_endpoint,
        solar_endpoint=args.solar_endpoint,
        weather_endpoint=args.weather_endpoint
    )
    
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
