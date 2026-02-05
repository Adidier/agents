"""
A2A Weather Expert System

A specialized expert system that simulates a weather agent with hardcoded responses.
"""

from typing import Dict, Any, List, Iterator
from a2a.core.a2a_ia_algorithm_interface import IA2AIAAlgorithm
from a2a.core.agent_card import AgentCard
from a2a.core.task_manager import TaskManager
from a2a.core.message_handler import MessageHandler


class A2AWeatherExpertSystem(IA2AIAAlgorithm):
    """
    Weather Expert System - A specialized agent for weather forecasting
    with hardcoded responses for simulation purposes.
    """
    
    def __init__(
        self,
        model: str,
        name: str,
        description: str,
        skills: List[Dict[str, Any]],
        host: str = "http://localhost:11434",
        endpoint: str = "http://localhost:8000",
    ):
        self.model = model
        self.agent_card = AgentCard(
            name=name,
            description=description,
            endpoint=endpoint,
            skills=skills,
        )
        self.task_manager = TaskManager()
        self.message_handler = MessageHandler()
        self.mcp_client = None
        
        # Hardcoded weather responses
        self.weather_responses = {
            "temperature": "Current temperature: 22°C (72°F). Comfortable conditions for outdoor activities.",
            "conditions": "Weather conditions: Partly cloudy with 40% chance of rain. Wind speed: 15 km/h from the east.",
            "forecast": "7-Day Forecast:\n- Today: 22°C, Partly Cloudy\n- Tomorrow: 20°C, Rainy\n- Wednesday: 18°C, Rainy\n- Thursday: 21°C, Cloudy\n- Friday: 24°C, Sunny\n- Saturday: 25°C, Sunny\n- Sunday: 23°C, Partly Cloudy",
            "humidity": "Humidity levels: 65% - Moderate. Dew point: 14°C.",
            "uv_index": "UV Index: 5 (Moderate). Recommend SPF 30+ sunscreen if outdoors.",
            "pressure": "Atmospheric pressure: 1013 hPa (Normal). Stable weather expected.",
            "wind": "Wind speed: 15 km/h from the east. Gusts up to 25 km/h.",
            "default": "Weather information not available for the requested query. Please try asking about temperature, conditions, forecast, humidity, UV index, wind, or pressure."
        }

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a weather request and return hardcoded response.
        
        Args:
            request: Request dictionary with message content
            
        Returns:
            Response dictionary with message parts
        """
        try:
            # Extract message content
            message_content = request.get("message", {}).get("content", "")
            
            # Get weather response
            response_content = self._get_weather_response(message_content)
            
            # Return formatted response
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
                            "content": f"Error processing weather request: {str(e)}"
                        }
                    ]
                }
            }

    def chat(self, prompt: str) -> Dict[str, Any]:
        """
        Process a chat request and return weather response.
        
        Args:
            prompt: User prompt/query about weather
            
        Returns:
            Response dictionary with message parts
        """
        response_content = self._get_weather_response(prompt)
        
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
    
    def _get_weather_response(self, prompt: str) -> str:
        """
        Select appropriate weather response based on prompt keywords.
        
        Args:
            prompt: User prompt/query
            
        Returns:
            Weather response string
        """
        # Handle empty or None prompt
        if not prompt or not isinstance(prompt, str):
            return self.weather_responses["temperature"]
        
        prompt_lower = prompt.lower().strip()
        
        # If prompt is empty, return temperature by default
        if not prompt_lower:
            return self.weather_responses["temperature"]
        
        # Keyword mapping for responses
        keywords = {
            "temperature": "temperature",
            "temp": "temperature",
            "hot": "temperature",
            "cold": "temperature",
            "degrees": "temperature",
            "conditions": "conditions",
            "weather": "conditions",
            "status": "conditions",
            "forecast": "forecast",
            "week": "forecast",
            "days": "forecast",
            "humidity": "humidity",
            "humid": "humidity",
            "moisture": "humidity",
            "uv": "uv_index",
            "sun": "uv_index",
            "sunburn": "uv_index",
            "pressure": "pressure",
            "atmospheric": "pressure",
            "wind": "wind",
            "breeze": "wind",
        }
        
        # Find matching keyword
        for keyword, response_key in keywords.items():
            if keyword in prompt_lower:
                return self.weather_responses.get(response_key, self.weather_responses["default"])
        
        # Default response returns temperature
        return self.weather_responses["temperature"]

    def _process_task(self, task_id: str) -> Dict[str, Any]:
        """
        Process a weather task and return weather forecast.
        
        Args:
            task_id: The task identifier
            
        Returns:
            Task result with weather information
        """
        # Get task details from task manager
        task = self.task_manager.get_task(task_id)
        
        if not task:
            return {
                "task_id": task_id,
                "status": "failed",
                "error": "Task not found"
            }
        
        # Extract query from task
        query = task.get("query", "wind")
        
        # Get weather response based on query
        response_content = self._get_weather_response(query)
        
        # Return processed task result
        return {
            "task_id": task_id,
            "status": "completed",
            "result": {
                "type": "weather_forecast",
                "content": response_content,
                "timestamp": task.get("timestamp", "")
            }
        }

    def _process_task_stream(self, task_id: str) -> Iterator[Dict[str, Any]]:
        pass

    def _get_ollama_messages(self, task_id: str) -> List[Dict[str, Any]]:
        pass

    def configure_mcp_client(self, mcp_client: Any) -> None:
        pass

    def discover_agent(self) -> Dict[str, Any]:
        pass
