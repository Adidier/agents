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
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from flask import Flask, request, jsonify

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
    Implements dynamic agent discovery similar to JADE's Directory Facilitator.
    Agents register themselves automatically via REST API.
    """
    
    def __init__(
        self, 
        mongodb_uri: Optional[str] = None,
        db_name: str = "solar_energy",
        collection_name: str = "agent_data",
        registry_port: int = 8001,
        use_dynamic_registry: bool = True
    ):
        """
        Initialize the orchestrator with dynamic agent registry.
        
        Args:
            mongodb_uri: MongoDB connection string
            db_name: MongoDB database name
            collection_name: MongoDB collection name
            registry_port: Port for the registry REST API
            use_dynamic_registry: Use dynamic registry or fixed endpoints
        """
        self.clients = {}
        self.registered_agents = {}  # {agent_id: {endpoint, name, skills, last_heartbeat}}
        self.history = []
        self.mongo_client = None
        self.mongo_collection = None
        self.mongodb_uri = mongodb_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.mongodb_enabled = False
        self.registry_port = registry_port
        self.use_dynamic_registry = use_dynamic_registry
        self.app = Flask(__name__) if use_dynamic_registry else None
        self.registry_thread = None
        
        if use_dynamic_registry:
            self._setup_registry_routes()
            self._start_registry_server()
            print(f"\n📋 Agent Registry Server started on port {registry_port}")
            print(f"   Registration endpoint: http://localhost:{registry_port}/register")
            print(f"   Agents should POST to this endpoint to register\n")
        
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
    
    def _save_registry_to_mongodb(self) -> bool:
        """
        Save current agent registry state to MongoDB.
        Stores a snapshot of all registered agents.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.mongodb_enabled or self.mongo_client is None:
            return False
        
        try:
            # Get registry collection
            db = self.mongo_client[self.db_name]
            registry_collection = db['agent_registry']
            
            # Prepare registry snapshot document
            agents_snapshot = []
            # Use list() to avoid RuntimeError if dict changes during iteration
            for agent_id, info in list(self.registered_agents.items()):
                agents_snapshot.append({
                    'agent_id': agent_id,
                    'name': info['name'],
                    'endpoint': info['endpoint'],
                    'skills': info['skills'],
                    'registered_at': info['registered_at'],
                    'last_heartbeat': info['last_heartbeat']
                })
            
            document = {
                "timestamp": datetime.now(),
                "agents": agents_snapshot,
                "total_agents": len(self.registered_agents),
                "registry_port": self.registry_port
            }
            
            # Insert registry snapshot
            result = registry_collection.insert_one(document)
            
            print(f"  💾 Registry snapshot saved to MongoDB (ID: {result.inserted_id})")
            return True
            
        except Exception as e:
            print(f"  ⚠️  Error saving registry to MongoDB: {e}")
            return False
    
    def _setup_registry_routes(self):
        """Setup Flask routes for agent registry."""
        @self.app.route('/register', methods=['POST'])
        def register_agent():
            """Endpoint for agents to register themselves."""
            try:
                data = request.json
                agent_id = data.get('agent_id')
                agent_name = data.get('name')
                endpoint = data.get('endpoint')
                skills = data.get('skills', [])
                
                if not all([agent_id, agent_name, endpoint]):
                    return jsonify({"error": "Missing required fields: agent_id, name, endpoint"}), 400
                
                self.registered_agents[agent_id] = {
                    'name': agent_name,
                    'endpoint': endpoint,
                    'skills': skills,
                    'registered_at': datetime.now(),
                    'last_heartbeat': datetime.now()
                }
                
                # Create A2A client for this agent
                agent_key = agent_name.lower().replace(' agent', '').replace(' ', '_')
                self.clients[agent_key] = A2AClient(endpoint)
                
                # Save registry to MongoDB
                self._save_registry_to_mongodb()
                
                print(f"✅ Agent registered: {agent_name} ({endpoint})")
                return jsonify({
                    "status": "registered",
                    "agent_id": agent_id,
                    "message": f"Agent {agent_name} successfully registered"
                }), 200
                
            except Exception as e:
                print(f"❌ Error registering agent: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/deregister', methods=['POST'])
        def deregister_agent():
            """Endpoint for agents to deregister themselves."""
            try:
                data = request.json
                agent_id = data.get('agent_id')
                
                if agent_id in self.registered_agents:
                    agent_info = self.registered_agents[agent_id]
                    agent_key = agent_info['name'].lower().replace(' agent', '').replace(' ', '_')
                    
                    del self.registered_agents[agent_id]
                    if agent_key in self.clients:
                        del self.clients[agent_key]
                    
                    # Update registry in MongoDB
                    self._save_registry_to_mongodb()
                    
                    print(f"🔴 Agent deregistered: {agent_info['name']}")
                    return jsonify({"status": "deregistered"}), 200
                else:
                    return jsonify({"error": "Agent not found"}), 404
                    
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/agents', methods=['GET'])
        def list_agents():
            """List all registered agents."""
            # Use list() to avoid RuntimeError if dict changes during iteration
            agents_list = [
                {
                    'agent_id': agent_id,
                    'name': info['name'],
                    'endpoint': info['endpoint'],
                    'skills': info['skills']
                }
                for agent_id, info in list(self.registered_agents.items())
            ]
            return jsonify({"agents": agents_list, "count": len(agents_list)}), 200
        
        @self.app.route('/heartbeat', methods=['POST'])
        def heartbeat():
            """Endpoint for agents to send heartbeat."""
            try:
                data = request.json
                agent_id = data.get('agent_id')
                
                if agent_id in self.registered_agents:
                    self.registered_agents[agent_id]['last_heartbeat'] = datetime.now()
                    return jsonify({"status": "ok"}), 200
                else:
                    return jsonify({"error": "Agent not registered"}), 404
            except Exception as e:
                return jsonify({"error": str(e)}), 500
    
    def _start_registry_server(self):
        """Start the Flask registry server in a separate thread."""
        def run_server():
            self.app.run(host='0.0.0.0', port=self.registry_port, threaded=True, use_reloader=False)
        
        self.registry_thread = threading.Thread(target=run_server, daemon=True)
        self.registry_thread.start()
        time.sleep(1)  # Give server time to start
    
    def cleanup_stale_agents(self, timeout_minutes: int = 5):
        """Remove agents that haven't sent heartbeat recently."""
        threshold = datetime.now() - timedelta(minutes=timeout_minutes)
        stale_agents = []
        
        # Use list() to avoid RuntimeError if dict changes during iteration
        for agent_id, info in list(self.registered_agents.items()):
            if info['last_heartbeat'] < threshold:
                stale_agents.append(agent_id)
        
        for agent_id in stale_agents:
            agent_info = self.registered_agents[agent_id]
            print(f"⚠️  Removing stale agent: {agent_info['name']}")
            agent_key = agent_info['name'].lower().replace(' agent', '').replace(' ', '_')
            del self.registered_agents[agent_id]
            if agent_key in self.clients:
                del self.clients[agent_key]
    
    def _verify_connections(self):
        """Verify connections to all agents and print their capabilities."""
        if not self.clients:
            print("⏳ No agents registered yet. Waiting for agents to register...")
            return
            
        try:
            # Use list() to avoid RuntimeError if dict changes during iteration
            for agent_name, client in list(self.clients.items()):
                try:
                    card = client.discover_agent()
                    print(f"✓ Connected to {agent_name.capitalize()} Agent: {card['name']}")
                    print(f"  Skills: {', '.join(skill['name'] for skill in card['skills'])}")
                except Exception as e:
                    print(f"⚠️  Could not verify {agent_name}: {e}")
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
        # Use list() to create a copy and avoid RuntimeError if dict changes during iteration
        for agent_name, client in list(self.clients.items()):
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
    parser = argparse.ArgumentParser(description="A2A Multi-Agent Orchestrator with Dynamic Agent Registry")
    parser.add_argument("--registry-port", type=int, default=8001, help="Port for agent registry server")
    parser.add_argument("--mongodb-uri", type=str, default=None, help="MongoDB connection string (e.g., mongodb://localhost:27017 or mongodb+srv://...)")
    parser.add_argument("--db-name", type=str, default="solar_energy", help="MongoDB database name")
    parser.add_argument("--collection", type=str, default="agent_data", help="MongoDB collection name")
    
    args = parser.parse_args()
    
    # Create the orchestrator with dynamic registry
    orchestrator = AgentOrchestrator(
        mongodb_uri=args.mongodb_uri,
        db_name=args.db_name,
        collection_name=args.collection,
        registry_port=args.registry_port,
        use_dynamic_registry=True
    )
    
    # Continuous monitoring loop
    print("\n" + "="*80)
    print("🚀 Orchestrator monitoring started (Dynamic Agent Registry)")
    print(f"📋 Registry API: http://localhost:{args.registry_port}")
    print(f"   Agents will auto-register at: http://localhost:{args.registry_port}/register")
    if orchestrator.mongodb_enabled:
        print("🗄️  MongoDB:", f"{args.db_name}.{args.collection}")
    else:
        print("⚠️  MongoDB not configured - data will not be persisted")
    print("🔄 Refresh rate: 10 seconds")
    print("💡 Tip: Run dashboard.py to visualize real-time data")
    print("Press Ctrl+C to stop")
    print("="*80)
    
    # Wait a bit for agents to register
    print("\n⏳ Waiting 5 seconds for agents to register...")
    time.sleep(5)
    
    try:
        iteration = 0
        while True:
            iteration += 1
            timestamp = datetime.now().isoformat()
            print(f"\n[{time.strftime('%H:%M:%S')}] Iteration #{iteration} - Active agents: {len(orchestrator.clients)}")
            
            # Cleanup stale agents every 10 iterations
            if iteration % 10 == 0:
                orchestrator.cleanup_stale_agents()
            
            # Skip if no agents registered
            if not orchestrator.clients:
                print("  ⏳ No agents registered yet, waiting...")
                time.sleep(10)
                continue
            
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
