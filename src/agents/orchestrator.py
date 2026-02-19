"""
Multi-Agent Example - Orchestrator

This module coordinates multiple specialized A2A agents to complete complex tasks.
Includes continuous monitoring with traffic light supervision system.
Stores data in MongoDB directly via pymongo.
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a2a.client import A2AClient

# Try to import pymongo for MongoDB support
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, OperationFailure
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    print("⚠️  pymongo not installed. MongoDB support disabled.")


class AgentOrchestrator:
    """
    Orchestrator for coordinating multiple A2A agents.
    Stores data in local JSON and optionally in MongoDB.
    """
    
    def __init__(
        self, 
        mongodb_uri: Optional[str] = None,
        db_name: str = "solar_energy",
        collection_name: str = "agent_data",
        **endpoints
    ):
        """
        Initialize the orchestrator with dynamic endpoints.
        
        Args:
            mongodb_uri: MongoDB connection string (e.g., mongodb://localhost:27017 or mongodb+srv://...)
            db_name: MongoDB database name
            collection_name: MongoDB collection name
            **endpoints: Variable keyword arguments for agent endpoints
                        (e.g., solar_endpoint="http://localhost:8002")
        """
        self.clients = {}
        self.history = []
        self.mongo_client = None
        self.mongo_collection = None
        self.mongodb_uri = mongodb_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.mongodb_enabled = False
        
        # Create clients dynamically for each endpoint
        for name, endpoint in endpoints.items():
            agent_name = name.replace("_endpoint", "")
            self.clients[agent_name] = A2AClient(endpoint)
        
        # Verify connections and discover capabilities
        print("Connecting to agents...")
        self._verify_connections()
        
        # Initialize MongoDB connection if URI provided
        if self.mongodb_uri and PYMONGO_AVAILABLE:
            print(f"\n🔌 Connecting to MongoDB...")
            try:
                self._init_mongodb()
                self.mongodb_enabled = True
                print(f"✅ MongoDB connection established")
                print(f"   Database: {self.db_name}")
                print(f"   Collection: {self.collection_name}")
            except Exception as e:
                print(f"⚠️  Could not connect to MongoDB: {e}")
                self.mongodb_enabled = False
        elif self.mongodb_uri and not PYMONGO_AVAILABLE:
            print(f"\n⚠️  MongoDB URI provided but pymongo not installed")
            print(f"   Install with: pip install pymongo")
        else:
            print(f"\n⚠️  No MongoDB URI provided. Data will not be persisted.")
            print(f"   Use --mongodb-uri to enable MongoDB storage")
    
    def _init_mongodb(self):
        """Initialize MongoDB connection."""
        self.mongo_client = MongoClient(
            self.mongodb_uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000
        )
        
        # Test connection
        self.mongo_client.admin.command('ping')
        
        # Get database and collection
        db = self.mongo_client[self.db_name]
        self.mongo_collection = db[self.collection_name]
    
    def _save_to_mongodb(self, data: Dict[str, Any]) -> bool:
        """
        Save data to MongoDB.
        
        Args:
            data: Data to save to MongoDB
            
        Returns:
            True if successful, False otherwise
        """
        if not self.mongodb_enabled or self.mongo_collection is None:
            return False
        
        try:
            # Prepare document for MongoDB
            document = {
                "timestamp": datetime.now(),
                "iteration": data.get("iteration"),
                "agents": data.get("agents", {}),
                "metadata": {
                    "db_name": self.db_name,
                    "collection": self.collection_name
                }
            }
            
            # Insert document
            result = self.mongo_collection.insert_one(document)
            
            print(f"  💾 Saved to MongoDB (ID: {result.inserted_id})")
            return True
            
        except Exception as e:
            print(f"  ⚠️  Error saving to MongoDB: {e}")
            return False
    
    def _verify_connections(self):
        """Verify connections to all agents and print their capabilities."""
        try:
            for agent_name, client in self.clients.items():
                card = client.discover_agent()
                print(f"✓ Connected to {agent_name.capitalize()} Agent: {card['name']}")
                print(f"  Skills: {', '.join(skill['name'] for skill in card['skills'])}")
        except Exception as e:
            print(f"⚠️  Error verifying connections: {e}")
    
    def process_topic(self, topic: str) -> Dict[str, Any]:
        """Process a topic by querying all connected agents dynamically.
        
        Returns:
            Dict with raw responses for each agent
        """
        
        # Define prompts for each agent dynamically
        prompts = {
            "generator": "Provide current status of the Solar/Generator panels. Include power generation, predictions, and current state.",
            "solar": "Provide current status of the Solar panels. Include power generation, predictions, and current state.",
            "weather": "Provide current weather status. Include temperature, radiation, and forecast.",
            "battery": "Provide battery status. Include SoC, voltage, current, and charge/discharge state.",
            "load": "Provide load consumption status. Include current demand, load types, and forecast.",
            "energy_price": "Provide electricity price predictions and current market conditions with recommendations.",
            "cenace": "Provide electricity price predictions and current market conditions with recommendations."
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
    parser = argparse.ArgumentParser(description="A2A Multi-Agent Orchestrator with MongoDB")
    parser.add_argument("--generator-endpoint", type=str, default="http://localhost:8002", help="Generator Agent endpoint")
    parser.add_argument("--solar-endpoint", type=str, default="http://localhost:8002", help="Solar Agent endpoint")
    parser.add_argument("--weather-endpoint", type=str, default="http://localhost:8004", help="Weather Agent endpoint")
    parser.add_argument("--battery-endpoint", type=str, default="http://localhost:8005", help="Battery Agent endpoint")
    parser.add_argument("--load-endpoint", type=str, default="http://localhost:8006", help="Load Agent endpoint")
    parser.add_argument("--energy-price-endpoint", type=str, default="http://localhost:8007", help="Energy Price Predictor Agent endpoint")
    parser.add_argument("--cenace-endpoint", type=str, default=None, help="(Deprecated: use --energy-price-endpoint) CENACE Agent endpoint")
    parser.add_argument("--mongodb-uri", type=str, default=None, help="MongoDB connection string (e.g., mongodb://localhost:27017 or mongodb+srv://...)")
    parser.add_argument("--db-name", type=str, default="solar_energy", help="MongoDB database name")
    parser.add_argument("--collection", type=str, default="agent_data", help="MongoDB collection name")
    
    args = parser.parse_args()
    
    # Handle backwards compatibility for cenace-endpoint
    if args.cenace_endpoint and not hasattr(args, 'energy_price_endpoint'):
        args.energy_price_endpoint = args.cenace_endpoint
    elif args.cenace_endpoint:
        print("⚠️  Warning: --cenace-endpoint is deprecated, use --energy-price-endpoint")
        if not args.energy_price_endpoint:
            args.energy_price_endpoint = args.cenace_endpoint
    
    # Dynamically create endpoint arguments
    endpoints = {}
    for key, value in vars(args).items():
        if key.endswith("_endpoint") and value:
            endpoints[key] = value
    
    # Create the orchestrator with dynamic endpoints
    orchestrator = AgentOrchestrator(
        mongodb_uri=args.mongodb_uri,
        db_name=args.db_name,
        collection_name=args.collection,
        **endpoints
    )
    
    # Continuous monitoring loop
    print("\n" + "="*80)
    print("🚀 Orchestrator monitoring started")
    if orchestrator.mongodb_enabled:
        print("🗄️  MongoDB:", f"{args.db_name}.{args.collection}")
    else:
        print("⚠️  MongoDB not configured - data will not be persisted")
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
            
            # Save to MongoDB only
            if orchestrator.mongodb_enabled:
                orchestrator._save_to_mongodb(iteration_data)
            else:
                print(f"  ⚠️  MongoDB not configured. Use --mongodb-uri to enable data storage.")
            
            # Wait 10 seconds before next iteration
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("Monitoring stopped by user")
        print("="*80)
        if orchestrator.mongodb_enabled:
            print(f"\n✅ Data saved to MongoDB. Total iterations: {len(orchestrator.history)}")
        else:
            print(f"\n⚠️  No data saved (MongoDB not configured). Total iterations: {len(orchestrator.history)}")


if __name__ == "__main__":
    main()
