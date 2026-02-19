"""
Weather Agent - NASA POWER API

Este agente se conecta a la API POWER de la NASA para obtener datos meteorológicos
de cualquier punto del planeta especificado por coordenadas geográficas.

NASA POWER API: https://power.larc.nasa.gov/docs/services/api/
"""

import os
import sys
import re
import argparse
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Iterator

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a2a.serverOllama import run_server
from a2a.core.a2a_ia_algorithm_interface import IA2AIAAlgorithm
from a2a.core.agent_card import AgentCard
from a2a.core.task_manager import TaskManager
from a2a.core.message_handler import MessageHandler


class WeatherAgent(IA2AIAAlgorithm):
    """
    Agente meteorológico que obtiene datos de la API POWER de la NASA.
    Proporciona información climática para sistemas fotovoltaicos.
    Con soporte de servidor mock para simulación sin conexión a internet.
    """
    
    NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
    MOCK_SERVER_URL = "http://localhost:8005/api/temporal/daily/point"
    
    # Parámetros relevantes para energía solar
    SOLAR_PARAMS = [
        "ALLSKY_SFC_SW_DWN",      # Irradiancia solar (kWh/m²/day)
        "CLRSKY_SFC_SW_DWN",      # Irradiancia cielo despejado
        "T2M",                     # Temperatura a 2m (°C)
        "T2M_MAX",                 # Temperatura máxima
        "T2M_MIN",                 # Temperatura mínima
        "RH2M",                    # Humedad relativa (%)
        "PRECTOTCORR",            # Precipitación (mm/day)
        "WS2M",                    # Velocidad del viento a 2m (m/s)
        "PS",                      # Presión atmosférica (kPa)
    ]
    
    def __init__(
        self,
        name: str,
        description: str,
        skills: List[Dict[str, Any]],
        endpoint: str = "http://localhost:8004",
        default_lat: float = 20.1011,  # Pachuca, Hidalgo, México
        default_lon: float = -98.7625,
        mock_server_url: Optional[str] = None,
        use_mock: bool = False,
    ):
        self.agent_card = AgentCard(
            name=name,
            description=description,
            endpoint=endpoint,
            skills=skills,
        )
        self.task_manager = TaskManager()
        self.message_handler = MessageHandler()
        self.mcp_client = None
        
        self.default_lat = default_lat
        self.default_lon = default_lon
        
        # Configuración del servidor mock
        self.mock_server_url = mock_server_url or self.MOCK_SERVER_URL
        self.use_mock = use_mock
        self.using_mock_mode = False
        
        # Cache para evitar solicitudes repetidas
        self.cache = {}
    
    def get_weather_data(
        self,
        latitude: float,
        longitude: float,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        parameters: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Obtiene datos meteorológicos de la API POWER de la NASA.
        
        Args:
            latitude: Latitud (-90 a 90)
            longitude: Longitud (-180 a 180)
            start_date: Fecha inicio (YYYYMMDD), por defecto últimos 7 días
            end_date: Fecha fin (YYYYMMDD), por defecto hoy
            parameters: Lista de parámetros, por defecto SOLAR_PARAMS
            
        Returns:
            Diccionario con los datos meteorológicos
        """
        # Validar coordenadas
        if not -90 <= latitude <= 90:
            return {"error": f"Latitud inválida: {latitude}. Debe estar entre -90 y 90"}
        if not -180 <= longitude <= 180:
            return {"error": f"Longitud inválida: {longitude}. Debe estar entre -180 y 180"}
        
        # Fechas por defecto
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
        
        # Parámetros por defecto
        if parameters is None:
            parameters = self.SOLAR_PARAMS
        
        # Crear clave de cache
        cache_key = f"{latitude}_{longitude}_{start_date}_{end_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Preparar solicitud
        params = {
            "parameters": ",".join(parameters),
            "community": "RE",  # Renewable Energy
            "longitude": longitude,
            "latitude": latitude,
            "start": start_date,
            "end": end_date,
            "format": "JSON"
        }
        
        # Determinar qué URL usar
        url = self.mock_server_url if self.use_mock else self.NASA_POWER_URL
        
        try:
            # Intentar con la API principal (o mock si está configurado)
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Marcar si estamos usando mock
            if self.use_mock or url == self.mock_server_url:
                self.using_mock_mode = True
                data["_mock_mode"] = True
            
            # Guardar en cache
            self.cache[cache_key] = data
            
            return data
            
        except requests.exceptions.RequestException as e:
            # Si falla la API de NASA, intentar con el servidor mock
            if not self.use_mock and url == self.NASA_POWER_URL:
                print(f"⚠️  API de NASA no disponible. Intentando servidor mock...")
                try:
                    response = requests.get(self.mock_server_url, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    data["_mock_mode"] = True
                    self.using_mock_mode = True
                    self.cache[cache_key] = data
                    print("✅ Usando datos simulados del servidor mock")
                    return data
                except:
                    return {"error": f"Error: Sin conexión a NASA API ni servidor mock disponible. {str(e)}"}
            else:
                return {"error": f"Error al conectar con servidor: {str(e)}"}
        except Exception as e:
            return {"error": f"Error procesando datos: {str(e)}"}
    
    def format_weather_response(self, data: Dict[str, Any]) -> str:
        """Formatea la respuesta de la API en texto legible."""
        if "error" in data:
            return f"❌ {data['error']}"
        
        try:
            params = data.get("properties", {}).get("parameter", {})
            
            if not params:
                return "❌ No se encontraron datos meteorológicos"
            
            # Verificar si estamos en modo simulación
            is_mock = data.get("_mock_mode", False)
            
            # Obtener últimos valores disponibles
            if is_mock:
                response_parts = ["🌍 Datos Meteorológicos (MODO SIMULACIÓN)\n"]
                response_parts.append("⚠️  Usando datos simulados - Sin conexión a NASA API\n")
            else:
                response_parts = ["🌍 Datos Meteorológicos NASA POWER\n"]
            
            if "ALLSKY_SFC_SW_DWN" in params:
                values = list(params["ALLSKY_SFC_SW_DWN"].values())
                avg = sum(values) / len(values) if values else 0
                response_parts.append(f"☀️ Irradiancia Solar: {avg:.2f} kWh/m²/día")
            
            if "T2M" in params:
                values = list(params["T2M"].values())
                avg = sum(values) / len(values) if values else 0
                response_parts.append(f"🌡️ Temperatura: {avg:.1f}°C")
            
            if "T2M_MAX" in params and "T2M_MIN" in params:
                max_vals = list(params["T2M_MAX"].values())
                min_vals = list(params["T2M_MIN"].values())
                if max_vals and min_vals:
                    response_parts.append(f"   Máx: {max(max_vals):.1f}°C | Mín: {min(min_vals):.1f}°C")
            
            if "RH2M" in params:
                values = list(params["RH2M"].values())
                avg = sum(values) / len(values) if values else 0
                response_parts.append(f"💧 Humedad Relativa: {avg:.1f}%")
            
            if "WS2M" in params:
                values = list(params["WS2M"].values())
                avg = sum(values) / len(values) if values else 0
                response_parts.append(f"💨 Velocidad del Viento: {avg:.1f} m/s")
            
            if "PRECTOTCORR" in params:
                values = list(params["PRECTOTCORR"].values())
                total = sum(values) if values else 0
                response_parts.append(f"🌧️ Precipitación Total: {total:.1f} mm")
            
            if "PS" in params:
                values = list(params["PS"].values())
                avg = sum(values) / len(values) if values else 0
                response_parts.append(f"🔘 Presión Atmosférica: {avg:.1f} kPa")
            
            # Información de ubicación
            geometry = data.get("geometry", {})
            coords = geometry.get("coordinates", [])
            if len(coords) >= 2:
                response_parts.append(f"\n📍 Ubicación: ({coords[1]:.4f}, {coords[0]:.4f})")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            return f"❌ Error formateando respuesta: {str(e)}"
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa solicitudes del protocolo A2A."""
        try:
            message_content = request.get("message", {}).get("content", "")
            
            # Extraer coordenadas del mensaje (formato: "lat,lon" o usar default)
            lat, lon = self.default_lat, self.default_lon
            
            # Intentar parsear coordenadas del mensaje
            if "," in message_content:
                try:
                    parts = message_content.split(",")
                    if len(parts) >= 2:
                        lat = float(parts[0].strip())
                        lon = float(parts[1].strip())
                except:
                    pass
            
            # Obtener datos meteorológicos
            weather_data = self.get_weather_data(lat, lon)
            response_text = self.format_weather_response(weather_data)
            
            return {
                "message": {
                    "parts": [{
                        "type": "text",
                        "content": response_text
                    }]
                }
            }
            
        except Exception as e:
            return {
                "message": {
                    "parts": [{
                        "type": "text",
                        "content": f"❌ Error: {str(e)}"
                    }]
                }
            }
    
    def chat(self, prompt: str) -> Dict[str, Any]:
        """Interface de chat para el agente meteorológico."""
        # Extraer coordenadas del prompt
        lat, lon = self.default_lat, self.default_lon
        
        # Buscar patrones de coordenadas
        coord_match = re.search(r'(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)', prompt)
        if coord_match:
            lat = float(coord_match.group(1))
            lon = float(coord_match.group(2))
        
        # Obtener datos meteorológicos
        weather_data = self.get_weather_data(lat, lon)
        response_text = self.format_weather_response(weather_data)
        
        return {
            "message": {
                "parts": [{
                    "type": "text",
                    "content": response_text
                }]
            }
        }
    
    def _process_task(self, task_id: str) -> Dict[str, Any]:
        """Procesa una tarea. Implementación requerida por IA2AIAAlgorithm."""
        # Obtener la tarea del task manager
        task = self.task_manager.get_task(task_id)
        
        if not task:
            self.task_manager.update_task_status(task_id, "failed")
            return {
                "task_id": task_id,
                "status": "failed",
                "error": "Task not found"
            }
        
        # Obtener mensajes de la tarea
        messages = self.message_handler.get_messages(task_id)
        
        if not messages:
            self.task_manager.update_task_status(task_id, "completed")
            return {
                "task_id": task_id,
                "status": "completed",
                "result": {"content": "No messages to process"}
            }
        
        # Obtener el último mensaje del usuario
        last_message = messages[-1]
        content = ""
        
        if "parts" in last_message:
            for part in last_message["parts"]:
                if part.get("type") == "text":
                    content = part.get("content", "")
                    break
        
        # Extraer coordenadas del contenido
        lat, lon = self.default_lat, self.default_lon
        
        coord_match = re.search(r'(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)', content)
        if coord_match:
            lat = float(coord_match.group(1))
            lon = float(coord_match.group(2))
        
        # Obtener datos meteorológicos
        weather_data = self.get_weather_data(lat, lon)
        response_text = self.format_weather_response(weather_data)
        
        # Actualizar estado de la tarea
        self.task_manager.update_task_status(task_id, "completed")
        
        # Devolver resultado
        return {
            "task_id": task_id,
            "status": "completed",
            "result": {
                "type": "weather_data",
                "content": response_text,
                "latitude": lat,
                "longitude": lon
            }
        }
    
    def _process_task_stream(self, task_id: str) -> Iterator[Dict[str, Any]]:
        """Procesa tarea con streaming. Implementación requerida por IA2AIAAlgorithm."""
        yield {"error": "Streaming not implemented for Weather Agent"}
    
    def _get_ollama_messages(self, task_id: str) -> List[Dict[str, Any]]:
        """Obtiene mensajes para Ollama. Implementación requerida por IA2AIAAlgorithm."""
        # Este agente no usa Ollama, devuelve lista vacía
        return []
    
    def configure_mcp_client(self, mcp_client: Any) -> None:
        """Configura el cliente MCP. Implementación requerida por IA2AIAAlgorithm."""
        self.mcp_client = mcp_client
    
    def get_card(self) -> AgentCard:
        """Retorna la tarjeta del agente."""
        return self.agent_card


def main():
    """Ejecuta el servidor del agente meteorológico NASA."""
    
    skills = [
        {
            "id": "weather_data",
            "name": "Weather Data",
            "description": "Obtiene datos meteorológicos de cualquier punto del planeta desde NASA POWER API"
        },
        {
            "id": "solar_radiation",
            "name": "Solar Radiation",
            "description": "Proporciona datos de irradiancia solar para sistemas fotovoltaicos"
        },
        {
            "id": "climate_info",
            "name": "Climate Info",
            "description": "Información climática: temperatura, humedad, viento, precipitación"
        }
    ]
    
    parser = argparse.ArgumentParser(description="Run NASA Weather Agent")
    parser.add_argument("--port", type=int, default=8004, help="Puerto del servidor")
    parser.add_argument("--lat", type=float, default=20.1011, help="Latitud por defecto (Pachuca, Hidalgo)")
    parser.add_argument("--lon", type=float, default=-98.7625, help="Longitud por defecto (Pachuca, Hidalgo)")
    parser.add_argument("--mock", action="store_true", help="Usar servidor mock en lugar de NASA API")
    parser.add_argument("--mock-url", default="http://localhost:8005/api/temporal/daily/point", 
                        help="URL del servidor mock")
    parser.add_argument("--orchestrator-url", type=str, default="http://localhost:8001", help="Orchestrator registry URL for auto-registration")
    
    args = parser.parse_args()
    
    # Crear el agente meteorológico
    weather_agent = WeatherAgent(
        name="NASA Weather Agent",
        skills=skills,
        description="Agente que obtiene datos meteorológicos de la API POWER de la NASA",
        endpoint=f"http://localhost:{args.port}",
        default_lat=args.lat,
        default_lon=args.lon,
        mock_server_url=args.mock_url,
        use_mock=args.mock
    )
    
    print(f"🚀 Iniciando NASA Weather Agent en puerto {args.port}")
    print(f"🌍 Ubicación por defecto: ({args.lat}, {args.lon})")
    print(f"📡 Conectado a NASA POWER API")
    
    # Iniciar el servidor A2A
    run_server(
        port=args.port,
        iaAlgorithm=weather_agent,
        orchestrator_url=args.orchestrator_url
    )


if __name__ == "__main__":
    import sys
    import codecs
    
    # Forzar codificación UTF-8 para manejar emojis
    if sys.stdout.encoding != 'UTF-8':
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    
    main() 


