"""
Load Agent - Simulador de Consumo Eléctrico

Este agente simula el consumo de energía eléctrica de una vivienda o edificio.
Incluye perfiles de consumo variables según hora del día, tipo de carga y patrones de uso.
"""

import os
import sys
import re
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Iterator
import random
import math

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a2a.serverOllama import run_server
from a2a.core.a2a_ia_algorithm_interface import IA2AIAAlgorithm
from a2a.core.agent_card import AgentCard
from a2a.core.task_manager import TaskManager
from a2a.core.message_handler import MessageHandler


class LoadAgent(IA2AIAAlgorithm):
    """
    Agente de simulación de consumo eléctrico.
    Modela diferentes tipos de cargas y perfiles de uso.
    """
    
    # Tipos de perfil de consumo
    PROFILE_RESIDENTIAL = "residential"
    PROFILE_COMMERCIAL = "commercial"
    PROFILE_INDUSTRIAL = "industrial"
    PROFILE_CUSTOM = "custom"
    
    # Tipos de carga
    LOAD_LIGHTING = "lighting"
    LOAD_HVAC = "hvac"  # Heating, Ventilation, Air Conditioning
    LOAD_APPLIANCES = "appliances"
    LOAD_WATER_HEATER = "water_heater"
    LOAD_ELECTRONICS = "electronics"
    LOAD_OTHER = "other"
    
    # Consumo base por tipo de carga (kW)
    BASE_CONSUMPTION = {
        LOAD_LIGHTING: 0.3,
        LOAD_HVAC: 2.0,
        LOAD_APPLIANCES: 1.5,
        LOAD_WATER_HEATER: 2.5,
        LOAD_ELECTRONICS: 0.5,
        LOAD_OTHER: 0.2
    }
    
    def __init__(
        self,
        name: str,
        description: str,
        skills: List[Dict[str, Any]],
        endpoint: str = "http://localhost:8008",
        profile: str = PROFILE_RESIDENTIAL,
        base_load_kw: float = 1.0,
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
        
        # Configuración del consumo
        self.profile = profile
        self.base_load_kw = base_load_kw
        self.current_load_kw = base_load_kw
        self.peak_load_kw = 0.0
        self.total_energy_kwh = 0.0
        
        # Estado de cargas individuales
        self.loads = {
            self.LOAD_LIGHTING: {"enabled": True, "power_kw": self.BASE_CONSUMPTION[self.LOAD_LIGHTING]},
            self.LOAD_HVAC: {"enabled": True, "power_kw": self.BASE_CONSUMPTION[self.LOAD_HVAC]},
            self.LOAD_APPLIANCES: {"enabled": True, "power_kw": self.BASE_CONSUMPTION[self.LOAD_APPLIANCES]},
            self.LOAD_WATER_HEATER: {"enabled": False, "power_kw": self.BASE_CONSUMPTION[self.LOAD_WATER_HEATER]},
            self.LOAD_ELECTRONICS: {"enabled": True, "power_kw": self.BASE_CONSUMPTION[self.LOAD_ELECTRONICS]},
            self.LOAD_OTHER: {"enabled": True, "power_kw": self.BASE_CONSUMPTION[self.LOAD_OTHER]}
        }
        
        # Multiplicador de demanda (1.0 = normal, >1.0 = alta demanda)
        self.demand_multiplier = 1.0
        
        # Modo automático
        self.auto_mode = True
        
        # Historial
        self.history = []
        self.last_update = datetime.now()
        
        # Inicializar consumo
        self._update_load()
    
    def _get_time_of_day_factor(self) -> float:
        """Calcula el factor de consumo según la hora del día."""
        now = datetime.now()
        hour = now.hour
        
        if self.profile == self.PROFILE_RESIDENTIAL:
            # Perfil residencial: picos en mañana y noche
            if 6 <= hour < 9:
                return 1.5  # Mañana: desayuno, preparación
            elif 9 <= hour < 12:
                return 0.6  # Media mañana: baja actividad
            elif 12 <= hour < 14:
                return 1.2  # Almuerzo
            elif 14 <= hour < 18:
                return 0.7  # Tarde: actividad moderada
            elif 18 <= hour < 23:
                return 1.8  # Noche: pico máximo
            else:
                return 0.4  # Madrugada: mínimo consumo
        
        elif self.profile == self.PROFILE_COMMERCIAL:
            # Perfil comercial: alto durante horas laborales
            if 8 <= hour < 18:
                return 1.5  # Horario laboral
            elif 18 <= hour < 22:
                return 0.8  # Cierre
            else:
                return 0.3  # Cerrado
        
        elif self.profile == self.PROFILE_INDUSTRIAL:
            # Perfil industrial: constante 24/7
            if 6 <= hour < 22:
                return 1.3  # Turno principal
            else:
                return 0.8  # Turno nocturno reducido
        
        else:  # CUSTOM
            return 1.0
    
    def _get_seasonal_factor(self) -> float:
        """Calcula el factor de consumo según la temporada."""
        now = datetime.now()
        month = now.month
        
        # Verano (junio-agosto): más AC
        if 6 <= month <= 8:
            return 1.3
        # Invierno (diciembre-febrero): más calefacción
        elif month == 12 or month <= 2:
            return 1.2
        # Primavera/Otoño: consumo moderado
        else:
            return 1.0
    
    def _update_load(self) -> None:
        """Actualiza el consumo total basado en cargas activas y factores."""
        # Calcular consumo base de cargas activas
        base_consumption = sum(
            load["power_kw"] for load in self.loads.values() if load["enabled"]
        )
        
        # Aplicar factores
        time_factor = self._get_time_of_day_factor()
        seasonal_factor = self._get_seasonal_factor()
        
        # Agregar variación aleatoria (±10%)
        random_factor = random.uniform(0.9, 1.1)
        
        # Calcular consumo actual
        self.current_load_kw = (
            base_consumption * 
            time_factor * 
            seasonal_factor * 
            self.demand_multiplier * 
            random_factor
        )
        
        # Actualizar pico
        if self.current_load_kw > self.peak_load_kw:
            self.peak_load_kw = self.current_load_kw
        
        # Calcular energía consumida desde última actualización
        now = datetime.now()
        time_delta = (now - self.last_update).total_seconds() / 3600.0  # horas
        self.total_energy_kwh += self.current_load_kw * time_delta
        self.last_update = now
    
    def get_load_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual del consumo."""
        self._update_load()
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "profile": self.profile,
            "current_load_kw": round(self.current_load_kw, 3),
            "base_load_kw": round(self.base_load_kw, 3),
            "peak_load_kw": round(self.peak_load_kw, 3),
            "total_energy_kwh": round(self.total_energy_kwh, 3),
            "demand_multiplier": round(self.demand_multiplier, 2),
            "time_factor": round(self._get_time_of_day_factor(), 2),
            "seasonal_factor": round(self._get_seasonal_factor(), 2),
            "auto_mode": self.auto_mode,
            "loads": {}
        }
        
        # Agregar estado de cada carga
        for load_type, load_info in self.loads.items():
            status["loads"][load_type] = {
                "enabled": load_info["enabled"],
                "power_kw": round(load_info["power_kw"], 3),
                "consumption_kw": round(
                    load_info["power_kw"] if load_info["enabled"] else 0.0, 3
                )
            }
        
        # Agregar alertas
        alerts = []
        if self.current_load_kw > 5.0:
            alerts.append("⚠️ Consumo elevado")
        if self.peak_load_kw > 8.0:
            alerts.append("🔴 Pico de demanda muy alto")
        
        if alerts:
            status["alerts"] = alerts
        
        # Guardar en historial
        self.history.append(status.copy())
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        
        return status
    
    def set_load(self, load_type: str, enabled: bool) -> Dict[str, Any]:
        """Activa o desactiva un tipo de carga."""
        if load_type not in self.loads:
            return {
                "success": False,
                "message": f"❌ Tipo de carga inválido: {load_type}",
                "valid_types": list(self.loads.keys())
            }
        
        self.loads[load_type]["enabled"] = enabled
        action = "activada" if enabled else "desactivada"
        
        return {
            "success": True,
            "message": f"✅ Carga '{load_type}' {action}",
            "status": self.get_load_status()
        }
    
    def adjust_load_power(self, load_type: str, power_kw: float) -> Dict[str, Any]:
        """Ajusta la potencia de un tipo de carga."""
        if load_type not in self.loads:
            return {
                "success": False,
                "message": f"❌ Tipo de carga inválido: {load_type}"
            }
        
        if power_kw < 0:
            return {
                "success": False,
                "message": "❌ La potencia no puede ser negativa"
            }
        
        old_power = self.loads[load_type]["power_kw"]
        self.loads[load_type]["power_kw"] = power_kw
        
        return {
            "success": True,
            "message": f"✅ Potencia de '{load_type}' ajustada: {old_power:.2f} kW → {power_kw:.2f} kW",
            "status": self.get_load_status()
        }
    
    def set_demand_multiplier(self, multiplier: float) -> Dict[str, Any]:
        """Establece el multiplicador de demanda."""
        if multiplier < 0:
            return {
                "success": False,
                "message": "❌ El multiplicador no puede ser negativo"
            }
        
        old_multiplier = self.demand_multiplier
        self.demand_multiplier = multiplier
        
        return {
            "success": True,
            "message": f"✅ Multiplicador de demanda: {old_multiplier:.2f}x → {multiplier:.2f}x",
            "status": self.get_load_status()
        }
    
    def set_profile(self, profile: str) -> Dict[str, Any]:
        """Cambia el perfil de consumo."""
        valid_profiles = [
            self.PROFILE_RESIDENTIAL,
            self.PROFILE_COMMERCIAL,
            self.PROFILE_INDUSTRIAL,
            self.PROFILE_CUSTOM
        ]
        
        if profile not in valid_profiles:
            return {
                "success": False,
                "message": f"❌ Perfil inválido. Perfiles válidos: {', '.join(valid_profiles)}"
            }
        
        old_profile = self.profile
        self.profile = profile
        
        return {
            "success": True,
            "message": f"✅ Perfil cambiado: {old_profile} → {profile}",
            "status": self.get_load_status()
        }
    
    def get_consumption_forecast(self, hours: int = 24) -> Dict[str, Any]:
        """Genera un pronóstico de consumo para las próximas horas."""
        forecast = []
        current_time = datetime.now()
        
        for i in range(hours):
            future_time = current_time + timedelta(hours=i)
            hour = future_time.hour
            
            # Simular factor de tiempo futuro
            if self.profile == self.PROFILE_RESIDENTIAL:
                if 6 <= hour < 9:
                    time_factor = 1.5
                elif 9 <= hour < 12:
                    time_factor = 0.6
                elif 12 <= hour < 14:
                    time_factor = 1.2
                elif 14 <= hour < 18:
                    time_factor = 0.7
                elif 18 <= hour < 23:
                    time_factor = 1.8
                else:
                    time_factor = 0.4
            else:
                time_factor = self._get_time_of_day_factor()
            
            # Calcular consumo estimado
            base = sum(load["power_kw"] for load in self.loads.values() if load["enabled"])
            estimated_load = base * time_factor * self._get_seasonal_factor() * self.demand_multiplier
            
            forecast.append({
                "time": future_time.strftime("%H:%M"),
                "hour": hour,
                "estimated_load_kw": round(estimated_load, 2)
            })
        
        return {
            "success": True,
            "forecast_hours": hours,
            "forecast": forecast,
            "total_estimated_kwh": round(sum(f["estimated_load_kw"] for f in forecast), 2)
        }
    
    def reset_statistics(self) -> Dict[str, Any]:
        """Reinicia las estadísticas de consumo."""
        self.total_energy_kwh = 0.0
        self.peak_load_kw = 0.0
        self.history = []
        
        return {
            "success": True,
            "message": "✅ Estadísticas reiniciadas",
            "status": self.get_load_status()
        }
    
    def format_load_response(self, status: Dict[str, Any]) -> str:
        """Formatea el estado del consumo en texto legible."""
        try:
            profile_emoji = {
                self.PROFILE_RESIDENTIAL: "🏠",
                self.PROFILE_COMMERCIAL: "🏢",
                self.PROFILE_INDUSTRIAL: "🏭",
                self.PROFILE_CUSTOM: "⚙️"
            }
            
            emoji = profile_emoji.get(status["profile"], "⚡")
            profile_name = status["profile"].title()
            
            response_parts = [
                f"{emoji} Consumo Eléctrico - Perfil {profile_name}\n",
                f"⚡ Consumo Actual: {status['current_load_kw']} kW",
                f"📊 Consumo Base: {status['base_load_kw']} kW",
                f"📈 Pico de Demanda: {status['peak_load_kw']} kW",
                f"🔋 Energía Total Consumida: {status['total_energy_kwh']} kWh",
                f"📉 Multiplicador de Demanda: {status['demand_multiplier']}x",
                f"🕐 Factor Horario: {status['time_factor']}x",
                f"🌡️ Factor Estacional: {status['seasonal_factor']}x",
                "\n💡 Estado de Cargas:"
            ]
            
            load_names = {
                self.LOAD_LIGHTING: "Iluminación",
                self.LOAD_HVAC: "Climatización",
                self.LOAD_APPLIANCES: "Electrodomésticos",
                self.LOAD_WATER_HEATER: "Calentador de Agua",
                self.LOAD_ELECTRONICS: "Electrónicos",
                self.LOAD_OTHER: "Otros"
            }
            
            for load_type, load_data in status["loads"].items():
                name = load_names.get(load_type, load_type)
                state = "✅" if load_data["enabled"] else "❌"
                response_parts.append(
                    f"  {state} {name}: {load_data['consumption_kw']} kW / {load_data['power_kw']} kW"
                )
            
            if "alerts" in status:
                response_parts.append("\n📢 Alertas:")
                for alert in status["alerts"]:
                    response_parts.append(f"  {alert}")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            return f"❌ Error formateando respuesta: {str(e)}"
    
    def process_command(self, command: str) -> Dict[str, Any]:
        """Procesa comandos de control del consumo."""
        command = command.lower().strip()
        
        # Comando: obtener estado
        if any(word in command for word in ["estado", "status", "consumo"]):
            status = self.get_load_status()
            return {
                "success": True,
                "message": self.format_load_response(status),
                "data": status
            }
        
        # Comando: pronóstico
        elif "pronostico" in command or "forecast" in command or "prediccion" in command:
            hours_match = re.search(r'(\d+)\s*(horas?|hours?|h)?', command)
            hours = int(hours_match.group(1)) if hours_match else 24
            hours = min(hours, 72)  # Máximo 72 horas
            
            forecast = self.get_consumption_forecast(hours)
            
            message_parts = [
                f"📊 Pronóstico de Consumo ({hours} horas)\n",
                f"Total estimado: {forecast['total_estimated_kwh']} kWh\n"
            ]
            
            # Mostrar primeras 12 horas
            for entry in forecast['forecast'][:12]:
                message_parts.append(f"  {entry['time']}: {entry['estimated_load_kw']} kW")
            
            if len(forecast['forecast']) > 12:
                message_parts.append(f"  ... (+{len(forecast['forecast']) - 12} horas más)")
            
            return {
                "success": True,
                "message": "\n".join(message_parts),
                "data": forecast
            }
        
        # Comando: activar/desactivar carga
        elif "activar" in command or "desactivar" in command or "enable" in command or "disable" in command:
            enable = "activar" in command or "enable" in command
            
            # Buscar tipo de carga en el comando
            load_type = None
            for lt in self.loads.keys():
                if lt in command:
                    load_type = lt
                    break
            
            if not load_type:
                # Buscar por nombre en español
                spanish_names = {
                    "iluminacion": self.LOAD_LIGHTING,
                    "climatizacion": self.LOAD_HVAC,
                    "electrodomesticos": self.LOAD_APPLIANCES,
                    "calentador": self.LOAD_WATER_HEATER,
                    "electronicos": self.LOAD_ELECTRONICS
                }
                for spanish, english in spanish_names.items():
                    if spanish in command:
                        load_type = english
                        break
            
            if load_type:
                result = self.set_load(load_type, enable)
                result["message"] = self.format_load_response(result["status"])
                return result
            else:
                return {
                    "success": False,
                    "message": f"❌ No se especificó tipo de carga. Tipos disponibles: {', '.join(self.loads.keys())}"
                }
        
        # Comando: ajustar potencia
        elif "ajustar" in command or "potencia" in command or "power" in command:
            power_match = re.search(r'(\d+\.?\d*)\s*kw', command)
            if not power_match:
                return {
                    "success": False,
                    "message": "❌ Especifica la potencia en kW (ej: 'ajustar hvac 2.5 kw')"
                }
            
            power = float(power_match.group(1))
            
            # Buscar tipo de carga
            load_type = None
            for lt in self.loads.keys():
                if lt in command:
                    load_type = lt
                    break
            
            if load_type:
                result = self.adjust_load_power(load_type, power)
                if result["success"]:
                    result["message"] = self.format_load_response(result["status"])
                return result
        
        # Comando: cambiar perfil
        elif "perfil" in command or "profile" in command:
            for profile in [self.PROFILE_RESIDENTIAL, self.PROFILE_COMMERCIAL, 
                           self.PROFILE_INDUSTRIAL, self.PROFILE_CUSTOM]:
                if profile in command or (profile == "residential" and "residencial" in command):
                    result = self.set_profile(profile)
                    result["message"] = self.format_load_response(result["status"])
                    return result
        
        # Comando: multiplicador de demanda
        elif "multiplicador" in command or "multiplier" in command or "demanda" in command:
            mult_match = re.search(r'(\d+\.?\d*)', command)
            if mult_match:
                multiplier = float(mult_match.group(1))
                result = self.set_demand_multiplier(multiplier)
                result["message"] = self.format_load_response(result["status"])
                return result
        
        # Comando: reset
        elif "reset" in command or "reiniciar" in command:
            result = self.reset_statistics()
            result["message"] = self.format_load_response(result["status"])
            return result
        
        # Comando: auto mode
        elif "auto" in command:
            self.auto_mode = not self.auto_mode
            mode_text = "activado" if self.auto_mode else "desactivado"
            return {
                "success": True,
                "message": f"✅ Modo automático {mode_text}",
                "data": {"auto_mode": self.auto_mode}
            }
        
        # Comando no reconocido
        else:
            status = self.get_load_status()
            return {
                "success": False,
                "message": "❓ Comando no reconocido. Comandos disponibles:\n" +
                          "  - 'estado': Ver estado del consumo\n" +
                          "  - 'pronostico [horas]': Ver pronóstico de consumo\n" +
                          "  - 'activar/desactivar [tipo]': Control de cargas\n" +
                          "  - 'ajustar [tipo] [potencia] kw': Ajustar potencia\n" +
                          "  - 'perfil [residential/commercial/industrial]': Cambiar perfil\n" +
                          "  - 'multiplicador [valor]': Ajustar demanda\n" +
                          "  - 'reset': Reiniciar estadísticas\n" +
                          "  - 'auto': Toggle modo automático\n\n" +
                          self.format_load_response(status)
            }
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa solicitudes del protocolo A2A."""
        try:
            message_content = request.get("message", {}).get("content", "")
            result = self.process_command(message_content)
            
            return {
                "message": {
                    "parts": [{
                        "type": "text",
                        "content": result.get("message", "")
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
        """Interface de chat para el agente de consumo."""
        result = self.process_command(prompt)
        
        return {
            "message": {
                "parts": [{
                    "type": "text",
                    "content": result.get("message", "")
                }]
            }
        }
    
    def _process_task(self, task_id: str) -> Dict[str, Any]:
        """Procesa una tarea. Implementación requerida por IA2AIAAlgorithm."""
        task = self.task_manager.get_task(task_id)
        
        if not task:
            self.task_manager.update_task_status(task_id, "failed")
            return {
                "task_id": task_id,
                "status": "failed",
                "error": "Task not found"
            }
        
        messages = self.message_handler.get_messages(task_id)
        
        if not messages:
            self.task_manager.update_task_status(task_id, "completed")
            return {
                "task_id": task_id,
                "status": "completed",
                "result": {"content": "No messages to process"}
            }
        
        last_message = messages[-1]
        content = ""
        
        if "parts" in last_message:
            for part in last_message["parts"]:
                if part.get("type") == "text":
                    content = part.get("content", "")
                    break
        
        result = self.process_command(content)
        self.task_manager.update_task_status(task_id, "completed")
        
        return {
            "task_id": task_id,
            "status": "completed",
            "result": {
                "type": "load_control",
                "content": result.get("message", ""),
                "data": result.get("data", {})
            }
        }
    
    def _process_task_stream(self, task_id: str) -> Iterator[Dict[str, Any]]:
        """Procesa tarea con streaming. Implementación requerida por IA2AIAAlgorithm."""
        yield {"error": "Streaming not implemented for Load Agent"}
    
    def _get_ollama_messages(self, task_id: str) -> List[Dict[str, Any]]:
        """Obtiene mensajes para Ollama. Implementación requerida por IA2AIAAlgorithm."""
        return []
    
    def configure_mcp_client(self, mcp_client: Any) -> None:
        """Configura el cliente MCP. Implementación requerida por IA2AIAAlgorithm."""
        self.mcp_client = mcp_client
    
    def get_card(self) -> AgentCard:
        """Retorna la tarjeta del agente."""
        return self.agent_card


def main():
    """Ejecuta el servidor del agente de consumo eléctrico."""
    
    skills = [
        {
            "id": "load_monitoring",
            "name": "Load Monitoring",
            "description": "Monitorea el consumo eléctrico en tiempo real"
        },
        {
            "id": "load_control",
            "name": "Load Control",
            "description": "Controla y gestiona diferentes tipos de cargas eléctricas"
        },
        {
            "id": "consumption_forecast",
            "name": "Consumption Forecast",
            "description": "Genera pronósticos de consumo basados en patrones históricos"
        },
        {
            "id": "demand_management",
            "name": "Demand Management",
            "description": "Gestión de demanda con perfiles personalizables"
        }
    ]
    
    parser = argparse.ArgumentParser(description="Run Load Agent")
    parser.add_argument("--port", type=int, default=8008, help="Puerto del servidor")
    parser.add_argument("--profile", type=str, default="residential", 
                       choices=["residential", "commercial", "industrial", "custom"],
                       help="Perfil de consumo")
    parser.add_argument("--base-load", type=float, default=1.0, 
                       help="Carga base en kW")
    parser.add_argument("--orchestrator-url", type=str, default="http://localhost:8001", help="Orchestrator registry URL for auto-registration")
    
    args = parser.parse_args()
    
    # Crear el agente de consumo
    load_agent = LoadAgent(
        name="Load Agent",
        skills=skills,
        description="Agente de simulación y control de consumo eléctrico",
        endpoint=f"http://localhost:{args.port}",
        profile=args.profile,
        base_load_kw=args.base_load
    )
    
    print(f"🚀 Iniciando Load Agent en puerto {args.port}")
    print(f"🏠 Perfil: {args.profile}")
    print(f"⚡ Carga base: {args.base_load} kW")
    print(f"📊 Consumo inicial: {load_agent.current_load_kw:.2f} kW")
    
    # Iniciar el servidor A2A
    run_server(
        port=args.port,
        iaAlgorithm=load_agent,
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
