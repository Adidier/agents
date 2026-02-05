"""
Multi-Agent Example - Orchestrator

This module coordinates multiple specialized A2A agents to complete complex tasks.
Includes continuous monitoring with traffic light supervision system.
"""

import os
import sys
import argparse
import time
import random
from typing import Dict, Any, List

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a2a.client import A2AClient
from a2a.algorithms.a2a_pv_simulator import PVSimulator


class AgentOrchestrator:
    """
    Orchestrator for coordinating multiple A2A agents.
    """
    
    def __init__(self, simulate_real_data: bool = True, **endpoints):
        """
        Initialize the orchestrator with dynamic endpoints.
        
        Args:
            simulate_real_data: Si True, simula valores reales para el sistema de supervisión
            **endpoints: Variable keyword arguments for agent endpoints
                        (e.g., ac_endpoint="http://localhost:8001")
        """
        self.clients = {}
        self.simulate_real_data = simulate_real_data
        self.simulator = PVSimulator(seed=None) if simulate_real_data else None  # Sin semilla para más aleatoriedad
        
        # Escenarios rotativos para simulación - más variados
        self.scenario_cycle = [
            "normal",      # 🟢 Verde
            "normal",      # 🟢 Verde
            "degraded",    # 🟡 Amarillo
            "normal",      # 🟢 Verde
            "fault",       # 🔴 Rojo
            "degraded",    # 🟡 Amarillo
            "fault",       # 🔴 Rojo
            "normal",      # 🟢 Verde
        ]
        self.scenario_index = 0
        
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
        """Extract text content from an agent response, including supervision status."""
        if not response:
            return "No content available"
        
        # Check if it's a task response with result
        if "result" in response and isinstance(response["result"], dict):
            result = response["result"]
            content_parts = []
            
            # Check if it's a classification result (with real power comparison)
            if result.get("type") == "pv_classification":
                pred = result.get("predicted_power", 0)
                real = result.get("real_power", 0)
                deviation = result.get("deviation_percent", 0)
                deviation_instant = result.get("deviation_instant", 0)
                supervision = result.get("supervision", {})
                metrics = result.get("metrics", {})
                
                emoji = supervision.get("light_emoji", "⚪")
                message = supervision.get("message", "")
                
                content_parts.append(f"📊 Predicción: {pred:.2f} kW")
                content_parts.append(f"\n📈 Real: {real:.2f} kW")
                content_parts.append(f"\n{emoji} {message}")
                content_parts.append(f"\n   Desviación: {deviation:.2f}% (Instantánea: {deviation_instant:.2f}%)")
                
                if metrics:
                    mae = metrics.get("MAE", 0)
                    rmse = metrics.get("RMSE", 0)
                    content_parts.append(f"\n   MAE: {mae:.4f} | RMSE: {rmse:.4f}")
                
                return "".join(content_parts)
            
            # Regular prediction result
            # Extract formatted value
            if "formatted" in result:
                content_parts.append(f"📊 {result['formatted']}")
            elif "value" in result:
                unit = result.get("unit", "")
                content_parts.append(f"📊 {result['value']:.2f} {unit}")
            
            # Extract supervision info if available
            if "supervision" in result:
                supervision = result["supervision"]
                light = supervision.get("light_status", "unknown")
                emoji = supervision.get("light_emoji", "⚪")
                
                if light == "green":
                    content_parts.append(f"\n{emoji} Estado: NORMAL - Sistema operando correctamente")
                elif light == "yellow":
                    content_parts.append(f"\n{emoji} Estado: ADVERTENCIA - Desviación moderada detectada")
                elif light == "red":
                    content_parts.append(f"\n{emoji} Estado: FALLA CRÍTICA - Intervención requerida")
                else:
                    content_parts.append(f"\n{emoji} Estado: {light.upper()}")
                
                # Add stats if available
                if "recent_stats" in supervision and supervision.get("history_size", 0) > 0:
                    stats = supervision["recent_stats"]
                    total = stats.get("total", 0)
                    if total > 0:
                        content_parts.append(
                            f"\n   Últimas {total} mediciones: "
                            f"🟢{stats.get('green', 0)} | "
                            f"🟡{stats.get('yellow', 0)} | "
                            f"🔴{stats.get('red', 0)}"
                        )
            
            return "".join(content_parts) if content_parts else str(result)
        
        # Fallback to original message extraction
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
                # Si es el agente solar y tenemos simulador, enviar valor real
                if agent_name == "solar" and self.simulate_real_data and "result" in agent_response:
                    predicted_value = self._extract_prediction_value(agent_response)
                    if predicted_value is not None:
                        # Simular valor real con escenarios rotativos
                        scenario = self.scenario_cycle[self.scenario_index % len(self.scenario_cycle)]
                        self.scenario_index += 1
                        
                        real_power = self.simulator.simulate_real_power(predicted_value, scenario=scenario)
                        
                        # Mostrar escenario usado
                        scenario_emoji = {"normal": "🟢", "degraded": "🟡", "fault": "🔴"}.get(scenario, "⚪")
                        print(f"   {scenario_emoji} Escenario: {scenario.upper()} (Real: {real_power:.2f} kW)")
                        
                        # Enviar valor real al agente
                        real_msg = f"REAL_POWER:{real_power:.2f}"
                        updated_response = client.chat(real_msg)
                        
                        # Usar respuesta actualizada si está disponible
                        if updated_response:
                            agent_response = updated_response
                
                content = self._extract_content(agent_response)
            
            responses[agent_name] = content
            print(content)
        
        return responses
    
    def _extract_prediction_value(self, response: Dict[str, Any]) -> float:
        """Extrae el valor numérico de predicción de una respuesta."""
        try:
            if "result" in response and isinstance(response["result"], dict):
                return float(response["result"].get("value", 0))
        except (ValueError, TypeError):
            pass
        return None


def main():
    parser = argparse.ArgumentParser(description="A2A Multi-Agent Orchestrator")
    parser.add_argument("--solar-endpoint", type=str, default="http://localhost:8002", help="Solar Agent endpoint")
    
    args = parser.parse_args()
    
    # Dynamically create endpoint arguments
    endpoints = {}
    for key, value in vars(args).items():
        if key.endswith("_endpoint"):
            endpoints[key] = value
    
    # Create the orchestrator with dynamic endpoints
    orchestrator = AgentOrchestrator(**endpoints)
    
    # Continuous monitoring loop
    print("\n" + "="*80)
    print("Starting continuous monitoring (polling every 10 seconds)")
    print("Press Ctrl+C to stop")
    print("="*80)
    
    try:
        iteration = 0
        while True:
            iteration += 1
            print(f"\n{'='*80}")
            print(f"Iteration #{iteration} - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}")
            
            # Process and get status from all agents
            responses = orchestrator.process_topic("")
            
            # Wait 10 seconds before next iteration
            print(f"\n⏳ Waiting 10 seconds before next check...")
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("Monitoring stopped by user")
        print("="*80)


if __name__ == "__main__":
    main()
