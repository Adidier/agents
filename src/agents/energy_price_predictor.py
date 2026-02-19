"""
Energy Price Predictor Agent - Mexican Electric Market

This agent predicts and analyzes electricity prices using real-time data from 
CENACE (Centro Nacional de Control de Energía) API. Provides price forecasting, 
market analysis, and consumption recommendations through the A2A protocol.
"""

import os
import sys
import argparse
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Iterator

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a2a.server import A2AServer
from a2a.core.a2a_ia_algorithm_interface import IA2AIAAlgorithm
from a2a.core.agent_card import AgentCard
from a2a.core.task_manager import TaskManager
from a2a.core.message_handler import MessageHandler


class EnergyPricePredictorAgent(IA2AIAAlgorithm):
    """
    Energy Price Predictor Agent that retrieves and analyzes electric market data from CENACE.
    Provides real-time pricing, forecasts, and intelligent recommendations for the Mexican electricity market.
    """
    
    def __init__(
        self,
        sistema: str = "SIN",
        mercado: str = "MDA",
        nodo: str = "06MTY-115",
        port: int = 8007,
        endpoint: str = "http://localhost:8007"
    ):
        """
        Initialize the Energy Price Predictor Agent.
        
        Args:
            sistema: Sistema (SIN=Sistema Interconectado Nacional, BCA=Baja California, BCS=Baja California Sur)
            mercado: Market type (MDA=Mercado del Día en Adelanto, MTR=Mercado Tiempo Real)
            nodo: Price node ID (e.g., 06MTY-115 for Monterrey)
            port: Port for the A2A server
            endpoint: Endpoint URL for agent card
        """
        self.sistema = sistema
        self.mercado = mercado
        self.nodo = nodo
        self.port = port
        
        # Initialize A2A components
        self.agent_card = AgentCard(
            name="Energy Price Predictor Agent",
            description="Predicts and analyzes electricity prices using real-time data from CENACE (Mexican Electric Market)",
            endpoint=endpoint,
            skills=[
                {
                    "name": "get_current_price",
                    "description": "Get current electricity market price",
                    "parameters": []
                },
                {
                    "name": "get_price_statistics",
                    "description": "Get price statistics for last 24 hours",
                    "parameters": []
                },
                {
                    "name": "analyze_market",
                    "description": "Analyze current market conditions",
                    "parameters": []
                },
                {
                    "name": "forecast_prices",
                    "description": "Forecast prices for next 6 hours",
                    "parameters": []
                }
            ]
        )
        
        self.task_manager = TaskManager()
        self.message_handler = MessageHandler()
        self.mcp_client = None
        
        # Data storage
        self.current_data = None
        self.last_update = None
        self.price_history = []
        
        # Download initial data
        print(f"🔌 Initializing Energy Price Predictor Agent")
        print(f"   Sistema: {self.sistema}")
        print(f"   Mercado: {self.mercado}")
        print(f"   Nodo: {self.nodo}")
        self._update_data()
    
    def _build_url(self, start_date: datetime, end_date: datetime) -> str:
        """
        Build CENACE API URL.
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            API URL string
        """
        return (
            f"https://ws01.cenace.gob.mx:8082/SWPML/SIM/"
            f"{self.sistema}/{self.mercado}/{self.nodo}/"
            f"{start_date.year}/{start_date.month:02d}/{start_date.day:02d}/"
            f"{end_date.year}/{end_date.month:02d}/{end_date.day:02d}/JSON"
        )
    
    def _update_data(self):
        """Update data from CENACE API."""
        try:
            # Get data for last 24 hours
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)
            
            url = self._build_url(start_date, end_date)
            
            print(f"\n📡 Fetching data from CENACE...")
            print(f"   URL: {url[:80]}...")
            
            # Read JSON data from CENACE
            response = requests.get(url, timeout=30, verify=False)
            response.raise_for_status()
            
            data = response.json()
            
            if "Resultados" in data and len(data["Resultados"]) > 0:
                valores = data["Resultados"][0]["Valores"]
                df = pd.DataFrame(valores)
                
                # Store current data
                self.current_data = df
                self.last_update = datetime.now()
                
                # Update price history (keep last 100 records)
                if not df.empty:
                    latest_prices = df.tail(10).to_dict('records')
                    self.price_history.extend(latest_prices)
                    self.price_history = self.price_history[-100:]  # Keep last 100
                
                print(f"✅ Data updated: {len(df)} records retrieved")
                print(f"   Latest price: ${float(df.iloc[-1]['pml']):.2f} MXN/MWh")
                print(f"   Timestamp: {df.iloc[-1]['fecha']}")
            else:
                print("⚠️  No data available from CENACE")
                
        except requests.exceptions.SSLError:
            print("⚠️  SSL verification error. Running without verification...")
            # This is expected with CENACE API
        except requests.exceptions.Timeout:
            print("❌ Timeout connecting to CENACE API")
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching CENACE data: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    
    def get_current_price(self) -> Dict[str, Any]:
        """
        Get the current electricity price.
        
        Returns:
            Dictionary with current price information
        """
        if self.current_data is None or self.current_data.empty:
            return {
                "price": 0.0,
                "currency": "MXN/MWh",
                "status": "no_data",
                "message": "No data available from CENACE"
            }
        
        try:
            latest = self.current_data.iloc[-1]
            
            return {
                "price": float(latest['pml']),
                "currency": "MXN/MWh",
                "timestamp": str(latest['fecha']),
                "node": self.nodo,
                "system": self.sistema,
                "market": self.mercado,
                "status": "active"
            }
        except Exception as e:
            return {
                "price": 0.0,
                "currency": "MXN/MWh",
                "status": "error",
                "message": str(e)
            }
    
    def get_price_statistics(self) -> Dict[str, Any]:
        """
        Calculate price statistics from current data.
        
        Returns:
            Dictionary with price statistics
        """
        if self.current_data is None or self.current_data.empty:
            return {
                "status": "no_data",
                "message": "No data available"
            }
        
        try:
            prices = self.current_data['pml']
            
            return {
                "current": float(prices.iloc[-1]),
                "average_24h": float(prices.mean()),
                "min_24h": float(prices.min()),
                "max_24h": float(prices.max()),
                "std_dev": float(prices.std()),
                "currency": "MXN/MWh",
                "samples": len(prices)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_price_forecast(self, hours: int = 6) -> Dict[str, Any]:
        """
        Generate a simple price forecast based on historical patterns.
        
        Args:
            hours: Number of hours to forecast
            
        Returns:
            Dictionary with forecast information
        """
        if self.current_data is None or self.current_data.empty:
            return {
                "status": "no_data",
                "message": "Not enough data for forecast"
            }
        
        try:
            prices = self.current_data['pml'].values
            
            # Simple moving average forecast
            window = min(6, len(prices))
            recent_avg = float(prices[-window:].mean())
            overall_avg = float(prices.mean())
            trend = recent_avg - overall_avg
            
            # Generate forecast
            forecast = []
            current_time = datetime.now()
            
            for hour in range(1, hours + 1):
                forecast_time = current_time + timedelta(hours=hour)
                forecast_price = recent_avg + (trend * hour * 0.1)  # Simple linear extrapolation
                
                forecast.append({
                    "hour": hour,
                    "timestamp": forecast_time.isoformat(),
                    "predicted_price": round(forecast_price, 2),
                    "confidence": "low"  # Simple model = low confidence
                })
            
            return {
                "status": "success",
                "method": "moving_average",
                "base_price": round(recent_avg, 2),
                "trend": round(trend, 2),
                "forecast": forecast,
                "currency": "MXN/MWh"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def analyze_market_conditions(self) -> Dict[str, Any]:
        """
        Analyze current market conditions using expert system.
        
        Returns:
            Dictionary with market analysis
        """
        current_price_info = self.get_current_price()
        stats = self.get_price_statistics()
        
        if current_price_info["status"] != "active":
            return {
                "status": "no_data",
                "recommendation": "Cannot analyze without data"
            }
        
        current_price = current_price_info["price"]
        avg_price = stats.get("average_24h", current_price)
        
        # Classify market conditions
        price_ratio = current_price / avg_price if avg_price > 0 else 1.0
        
        if price_ratio < 0.8:
            condition = "low_price"
            recommendation = "Favorable for consumption. Consider increasing load."
            color = "green"
        elif price_ratio > 1.2:
            condition = "high_price"
            recommendation = "High prices. Consider reducing consumption or using stored energy."
            color = "red"
        else:
            condition = "normal"
            recommendation = "Normal market conditions. Operate as planned."
            color = "yellow"
        
        return {
            "status": "active",
            "condition": condition,
            "current_price": round(current_price, 2),
            "average_price": round(avg_price, 2),
            "price_ratio": round(price_ratio, 2),
            "recommendation": recommendation,
            "light_status": color,
            "currency": "MXN/MWh"
        }
    
    def chat(self, prompt: str) -> Dict[str, Any]:
        """
        Chat interface with CENACE market data.
        
        Args:
            prompt: User prompt/question
            
        Returns:
            Response dictionary with market information
        """
        response_text = self.handle_chat(prompt)
        
        return {
            "message": {
                "parts": [
                    {
                        "type": "text",
                        "content": response_text
                    }
                ]
            }
        }
    
    def handle_chat(self, message: str) -> str:
        """
        Handle chat messages using expert system.
        
        Args:
            message: User message
            
        Returns:
            Response string
        """
        message_lower = message.lower()
        
        # Update data if requested
        if "actualizar" in message_lower or "refresh" in message_lower or "update" in message_lower:
            self._update_data()
            return "Datos actualizados desde CENACE."
        
        # Get current price
        if "precio" in message_lower or "price" in message_lower:
            price_info = self.get_current_price()
            if price_info["status"] == "active":
                return f"Precio actual: ${price_info['price']:.2f} {price_info['currency']} (Nodo: {price_info['node']})"
            else:
                return f"No hay datos disponibles: {price_info.get('message', 'Unknown error')}"
        
        # Get statistics
        if "estadistica" in message_lower or "stats" in message_lower or "statistics" in message_lower:
            stats = self.get_price_statistics()
            if stats.get("status") == "no_data":
                return "No hay suficientes datos para estadísticas."
            
            return (
                f"Estadísticas de Precio (24h):\n"
                f"Actual: ${stats['current']:.2f}\n"
                f"Promedio: ${stats['average_24h']:.2f}\n"
                f"Mínimo: ${stats['min_24h']:.2f}\n"
                f"Máximo: ${stats['max_24h']:.2f}\n"
                f"Desviación: ${stats['std_dev']:.2f}"
            )
        
        # Get forecast
        if "pronostico" in message_lower or "prediccion" in message_lower or "forecast" in message_lower:
            forecast = self.get_price_forecast()
            if forecast["status"] == "success":
                return f"Pronóstico de precio (próximas 6 horas) - Precio base: ${forecast['base_price']:.2f}, Tendencia: {forecast['trend']:.2f}"
            else:
                return f"No se puede generar pronóstico: {forecast.get('message', 'Unknown error')}"
        
        # Default response with market analysis
        analysis = self.analyze_market_conditions()
        current = self.get_current_price()
        
        if analysis["status"] == "active":
            return (
                f"📊 CENACE Market Data\n"
                f"Precio actual: ${current['price']:.2f} MXN/MWh\n"
                f"Condición: {analysis['condition']}\n"
                f"Recomendación: {analysis['recommendation']}\n"
                f"Última actualización: {self.last_update.strftime('%Y-%m-%d %H:%M:%S') if self.last_update else 'N/A'}"
            )
        else:
            return "No hay datos disponibles del CENACE. Use 'actualizar' para obtener datos."
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an A2A request.
        
        Args:
            request: Request dictionary with message content
            
        Returns:
            Response dictionary
        """
        try:
            message_content = request.get("message", {}).get("content", "")
            response_content = self.handle_chat(message_content)
            
            return {
                "message": {
                    "parts": [
                        {
                            "type": "text",
                            "content": response_content
                        }
                    ]
                }
            }
        except Exception as e:
            return {
                "message": {
                    "parts": [
                        {
                            "type": "text",
                            "content": f"Error processing request: {str(e)}"
                        }
                    ]
                }
            }
    
    def _process_task(self, task_id: str) -> Dict[str, Any]:
        """
        Process a task: fetch CENACE data and analyze market.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task result with market data
        """
        try:
            # Update data from CENACE
            self._update_data()
            
            # Get analysis
            current_price = self.get_current_price()
            statistics = self.get_price_statistics()
            analysis = self.analyze_market_conditions()
            
            if current_price["status"] != "active":
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": "No data available from CENACE"
                }
            
            # Build comprehensive result
            result = {
                "task_id": task_id,
                "status": "completed",
                "result": {
                    "type": "cenace_market_data",
                    "current_price": current_price,
                    "statistics": statistics,
                    "analysis": analysis,
                    "formatted": (
                        f"Precio: ${current_price['price']:.2f} MXN/MWh | "
                        f"Condición: {analysis.get('condition', 'unknown')} | "
                        f"Estado: {analysis.get('light_status', 'unknown')}"
                    )
                }
            }
            
            return result
            
        except Exception as e:
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(e)
            }
    
    def _process_task_stream(self, task_id: str) -> Iterator[Dict[str, Any]]:
        """Mock implementation of streaming tasks."""
        return iter([])
    
    def _get_ollama_messages(self, task_id: str) -> List[Dict[str, Any]]:
        """Mock implementation - CENACE doesn't use Ollama."""
        return []
    
    def configure_mcp_client(self, mcp_client: Any) -> None:
        """Configure MCP client if needed."""
        self.mcp_client = mcp_client
    
    def discover_agent(self) -> Dict[str, Any]:
        """Return agent card information."""
        return self.get_agent_card()
    
    def get_agent_card(self) -> Dict[str, Any]:
        """
        Return the agent card with capabilities.
        
        Returns:
            Agent card dictionary
        """
        return {
            "name": "Energy Price Predictor Agent",
            "description": "Predicts and analyzes electricity prices using real-time data from CENACE (Mexican Electric Market)",
            "version": "1.0.0",
            "capabilities": [
                "Real-time electricity price monitoring",
                "Price statistics and trend analysis",
                "Market condition assessment",
                "Smart price forecasting",
                "Energy consumption recommendations"
            ],
            "skills": [
                {
                    "name": "get_current_price",
                    "description": "Get current electricity market price",
                    "parameters": []
                },
                {
                    "name": "get_price_statistics",
                    "description": "Get price statistics for last 24 hours",
                    "parameters": []
                },
                {
                    "name": "analyze_market",
                    "description": "Analyze current market conditions",
                    "parameters": []
                },
                {
                    "name": "forecast_prices",
                    "description": "Forecast prices for next 6 hours",
                    "parameters": []
                }
            ],
            "config": {
                "sistema": self.sistema,
                "mercado": self.mercado,
                "nodo": self.nodo,
                "last_update": self.last_update.isoformat() if self.last_update else None
            }
        }


def main():
    """Main function to run the Energy Price Predictor agent."""
    parser = argparse.ArgumentParser(description="Energy Price Predictor Agent - CENACE Market Data")
    parser.add_argument("--port", type=int, default=8007, help="Port for A2A server")
    parser.add_argument("--sistema", type=str, default="SIN", help="Sistema (SIN/BCA/BCS)")
    parser.add_argument("--mercado", type=str, default="MDA", help="Mercado (MDA/MTR)")
    parser.add_argument("--nodo", type=str, default="06MTY-115", help="Nodo de precio")
    parser.add_argument("--orchestrator-url", type=str, default="http://localhost:8001", help="Orchestrator registry URL for auto-registration")
    
    args = parser.parse_args()
    
    # Create Energy Price Predictor agent with proper endpoint
    endpoint = f"http://localhost:{args.port}"
    price_predictor_agent = EnergyPricePredictorAgent(
        sistema=args.sistema,
        mercado=args.mercado,
        nodo=args.nodo,
        port=args.port,
        endpoint=endpoint
    )
    
    # Create A2A server with the Energy Price Predictor agent as iaAlgorithm
    server = A2AServer(port=args.port, iaAlgorithm=price_predictor_agent, orchestrator_url=args.orchestrator_url)
    
    # Start server
    print(f"\n{'='*80}")
    print(f"🚀 Energy Price Predictor Agent started on port {args.port}")
    print(f"📡 Sistema: {args.sistema} | Mercado: {args.mercado} | Nodo: {args.nodo}")
    print(f"{'='*80}\n")
    
    server.run()


if __name__ == "__main__":
    main()
