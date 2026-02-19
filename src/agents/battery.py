"""
Battery Storage Agent

Este agente controla y monitorea el sistema de almacenamiento de baterías.
Gestiona la carga, descarga, estado de salud y optimización del uso de baterías.
"""

import os
import sys
import re
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Iterator
import random

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a2a.serverOllama import run_server
from a2a.core.a2a_ia_algorithm_interface import IA2AIAAlgorithm
from a2a.core.agent_card import AgentCard
from a2a.core.task_manager import TaskManager
from a2a.core.message_handler import MessageHandler


class BatteryAgent(IA2AIAAlgorithm):
    """
    Agente de control y monitoreo de sistema de almacenamiento de baterías.
    Gestiona carga, descarga, SOC, y optimización energética.
    """
    
    # Constantes del sistema de baterías
    BATTERY_CAPACITY_KWH = 10.0  # Capacidad nominal en kWh
    MAX_VOLTAGE = 54.0  # Voltaje máximo (V)
    MIN_VOLTAGE = 42.0  # Voltaje mínimo (V)
    NOMINAL_VOLTAGE = 48.0  # Voltaje nominal (V)
    MAX_CHARGE_CURRENT = 50.0  # Corriente máxima de carga (A)
    MAX_DISCHARGE_CURRENT = 50.0  # Corriente máxima de descarga (A)
    MIN_SOC = 20.0  # SOC mínimo recomendado (%)
    MAX_SOC = 95.0  # SOC máximo recomendado (%)
    SAFE_TEMP_MIN = 5.0  # Temperatura mínima segura (°C)
    SAFE_TEMP_MAX = 45.0  # Temperatura máxima segura (°C)
    
    # Modos de operación
    MODE_CHARGING = "charging"
    MODE_DISCHARGING = "discharging"
    MODE_IDLE = "idle"
    MODE_STANDBY = "standby"
    MODE_EMERGENCY = "emergency"
    
    def __init__(
        self,
        name: str,
        description: str,
        skills: List[Dict[str, Any]],
        endpoint: str = "http://localhost:8005",
        initial_soc: float = 50.0,
        capacity_kwh: float = 10.0,
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
        
        # Estado del sistema de baterías
        self.capacity_kwh = capacity_kwh
        self.soc = initial_soc  # State of Charge (%)
        self.voltage = self._calculate_voltage_from_soc(initial_soc)
        self.current = 0.0  # Corriente actual (A)
        self.temperature = 25.0  # Temperatura (°C)
        self.mode = self.MODE_IDLE
        self.cycles = 0  # Ciclos de carga/descarga
        self.health = 100.0  # Estado de salud (%)
        self.power = 0.0  # Potencia actual (kW)
        
        # Historial
        self.history = []
        self.last_update = datetime.now()
        
        # Configuración de control
        self.auto_mode = True
        self.charge_efficiency = 0.95
        self.discharge_efficiency = 0.92
    
    def _calculate_voltage_from_soc(self, soc: float) -> float:
        """Calcula el voltaje aproximado basado en el SOC."""
        # Curva de descarga simplificada (lineal)
        voltage_range = self.MAX_VOLTAGE - self.MIN_VOLTAGE
        return self.MIN_VOLTAGE + (soc / 100.0) * voltage_range
    
    def _calculate_soc_from_voltage(self, voltage: float) -> float:
        """Calcula el SOC aproximado basado en el voltaje."""
        voltage_range = self.MAX_VOLTAGE - self.MIN_VOLTAGE
        return ((voltage - self.MIN_VOLTAGE) / voltage_range) * 100.0
    
    def get_battery_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual del sistema de baterías."""
        # Simular ligeras variaciones en temperatura
        temp_variation = random.uniform(-0.5, 0.5)
        self.temperature += temp_variation
        self.temperature = max(self.SAFE_TEMP_MIN, min(self.SAFE_TEMP_MAX, self.temperature))
        
        # Actualizar voltaje basado en SOC
        self.voltage = self._calculate_voltage_from_soc(self.soc)
        
        # Calcular potencia
        self.power = (self.voltage * self.current) / 1000.0  # kW
        
        # Determinar estado de salud basado en ciclos
        if self.cycles > 3000:
            self.health = max(70.0, 100.0 - (self.cycles - 3000) * 0.005)
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "soc": round(self.soc, 2),
            "voltage": round(self.voltage, 2),
            "current": round(self.current, 2),
            "power": round(self.power, 3),
            "temperature": round(self.temperature, 1),
            "mode": self.mode,
            "capacity_kwh": self.capacity_kwh,
            "available_kwh": round((self.soc / 100.0) * self.capacity_kwh, 2),
            "health": round(self.health, 1),
            "cycles": self.cycles,
            "auto_mode": self.auto_mode
        }
        
        # Agregar alertas si es necesario
        alerts = []
        if self.soc < self.MIN_SOC:
            alerts.append("⚠️ SOC bajo - Se recomienda cargar")
        if self.soc > self.MAX_SOC:
            alerts.append("⚠️ SOC alto - Reducir carga")
        if self.temperature > self.SAFE_TEMP_MAX - 5:
            alerts.append("🌡️ Temperatura elevada")
        if self.temperature < self.SAFE_TEMP_MIN + 5:
            alerts.append("❄️ Temperatura baja")
        if self.health < 80:
            alerts.append("🔧 Estado de salud degradado")
        
        if alerts:
            status["alerts"] = alerts
        
        # Guardar en historial
        self.history.append(status.copy())
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        
        return status
    
    def charge_battery(self, power_kw: float, duration_minutes: float = 1.0) -> Dict[str, Any]:
        """
        Carga la batería con la potencia especificada.
        
        Args:
            power_kw: Potencia de carga en kW
            duration_minutes: Duración de la carga en minutos
            
        Returns:
            Estado actualizado de la batería
        """
        if self.soc >= 100.0:
            return {
                "success": False,
                "message": "❌ Batería completamente cargada",
                "status": self.get_battery_status()
            }
        
        # Limitar potencia según corriente máxima
        max_power = (self.voltage * self.MAX_CHARGE_CURRENT) / 1000.0
        power_kw = min(power_kw, max_power)
        
        # Calcular energía agregada (kWh)
        energy_kwh = (power_kw * duration_minutes / 60.0) * self.charge_efficiency
        
        # Actualizar SOC
        soc_increase = (energy_kwh / self.capacity_kwh) * 100.0
        self.soc = min(100.0, self.soc + soc_increase)
        
        # Actualizar corriente
        self.current = (power_kw * 1000.0) / self.voltage
        
        # Cambiar modo
        self.mode = self.MODE_CHARGING
        
        # Simular aumento de temperatura durante la carga
        self.temperature += random.uniform(0.1, 0.5)
        
        return {
            "success": True,
            "message": f"✅ Cargando con {power_kw:.2f} kW",
            "energy_added_kwh": round(energy_kwh, 3),
            "status": self.get_battery_status()
        }
    
    def discharge_battery(self, power_kw: float, duration_minutes: float = 1.0) -> Dict[str, Any]:
        """
        Descarga la batería con la potencia especificada.
        
        Args:
            power_kw: Potencia de descarga en kW
            duration_minutes: Duración de la descarga en minutos
            
        Returns:
            Estado actualizado de la batería
        """
        if self.soc <= 0.0:
            return {
                "success": False,
                "message": "❌ Batería vacía",
                "status": self.get_battery_status()
            }
        
        # Limitar potencia según corriente máxima
        max_power = (self.voltage * self.MAX_DISCHARGE_CURRENT) / 1000.0
        power_kw = min(power_kw, max_power)
        
        # Calcular energía extraída (kWh)
        energy_kwh = (power_kw * duration_minutes / 60.0) / self.discharge_efficiency
        
        # Actualizar SOC
        soc_decrease = (energy_kwh / self.capacity_kwh) * 100.0
        self.soc = max(0.0, self.soc - soc_decrease)
        
        # Actualizar corriente (negativa para descarga)
        self.current = -(power_kw * 1000.0) / self.voltage
        
        # Cambiar modo
        self.mode = self.MODE_DISCHARGING
        
        # Simular aumento de temperatura durante la descarga
        self.temperature += random.uniform(0.2, 0.8)
        
        # Incrementar ciclos si completamos una descarga significativa
        if self.soc < 20.0 and self.mode == self.MODE_DISCHARGING:
            self.cycles += 0.1
        
        return {
            "success": True,
            "message": f"✅ Descargando {power_kw:.2f} kW",
            "energy_extracted_kwh": round(energy_kwh, 3),
            "status": self.get_battery_status()
        }
    
    def set_mode(self, mode: str) -> Dict[str, Any]:
        """Establece el modo de operación de la batería."""
        valid_modes = [
            self.MODE_CHARGING,
            self.MODE_DISCHARGING,
            self.MODE_IDLE,
            self.MODE_STANDBY,
            self.MODE_EMERGENCY
        ]
        
        if mode not in valid_modes:
            return {
                "success": False,
                "message": f"❌ Modo inválido. Modos válidos: {', '.join(valid_modes)}"
            }
        
        self.mode = mode
        
        if mode == self.MODE_IDLE or mode == self.MODE_STANDBY:
            self.current = 0.0
            self.power = 0.0
        
        return {
            "success": True,
            "message": f"✅ Modo cambiado a: {mode}",
            "status": self.get_battery_status()
        }
    
    def format_battery_response(self, status: Dict[str, Any]) -> str:
        """Formatea el estado de la batería en texto legible."""
        try:
            mode_emoji = {
                self.MODE_CHARGING: "🔋⚡",
                self.MODE_DISCHARGING: "🔋⬇️",
                self.MODE_IDLE: "🔋💤",
                self.MODE_STANDBY: "🔋⏸️",
                self.MODE_EMERGENCY: "🔋🚨"
            }
            
            emoji = mode_emoji.get(status["mode"], "🔋")
            
            response_parts = [
                f"{emoji} Estado del Sistema de Almacenamiento\n",
                f"⚡ Estado de Carga (SOC): {status['soc']}%",
                f"📊 Energía Disponible: {status['available_kwh']} kWh / {status['capacity_kwh']} kWh",
                f"🔌 Voltaje: {status['voltage']} V",
                f"⚡ Corriente: {status['current']} A",
                f"💪 Potencia: {status['power']} kW",
                f"🌡️ Temperatura: {status['temperature']}°C",
                f"🔄 Modo: {status['mode'].upper()}",
                f"❤️ Estado de Salud: {status['health']}%",
                f"🔁 Ciclos: {status['cycles']}",
            ]
            
            if "alerts" in status:
                response_parts.append("\n📢 Alertas:")
                for alert in status["alerts"]:
                    response_parts.append(f"  {alert}")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            return f"❌ Error formateando respuesta: {str(e)}"
    
    def process_command(self, command: str) -> Dict[str, Any]:
        """Procesa comandos de control de la batería."""
        command = command.lower().strip()
        
        # Comando: obtener estado
        if any(word in command for word in ["estado", "status", "info"]):
            status = self.get_battery_status()
            return {
                "success": True,
                "message": self.format_battery_response(status),
                "data": status
            }
        
        # Comando: cargar
        elif "cargar" in command or "charge" in command:
            # Extraer potencia del comando
            power_match = re.search(r'(\d+\.?\d*)\s*(kw|w)?', command)
            power_kw = float(power_match.group(1)) if power_match else 2.0
            if power_match and power_match.group(2) == 'w':
                power_kw /= 1000.0
            
            result = self.charge_battery(power_kw)
            result["message"] = self.format_battery_response(result["status"])
            return result
        
        # Comando: descargar
        elif "descargar" in command or "discharge" in command:
            # Extraer potencia del comando
            power_match = re.search(r'(\d+\.?\d*)\s*(kw|w)?', command)
            power_kw = float(power_match.group(1)) if power_match else 2.0
            if power_match and power_match.group(2) == 'w':
                power_kw /= 1000.0
            
            result = self.discharge_battery(power_kw)
            result["message"] = self.format_battery_response(result["status"])
            return result
        
        # Comando: cambiar modo
        elif "modo" in command or "mode" in command:
            for mode in [self.MODE_CHARGING, self.MODE_DISCHARGING, 
                        self.MODE_IDLE, self.MODE_STANDBY, self.MODE_EMERGENCY]:
                if mode in command:
                    result = self.set_mode(mode)
                    if result["success"]:
                        result["message"] = self.format_battery_response(result["status"])
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
            status = self.get_battery_status()
            return {
                "success": False,
                "message": "❓ Comando no reconocido. Comandos disponibles:\n" +
                          "  - 'estado' o 'status': Ver estado del sistema\n" +
                          "  - 'cargar [potencia] kw': Cargar batería\n" +
                          "  - 'descargar [potencia] kw': Descargar batería\n" +
                          "  - 'modo [idle/standby/emergency]': Cambiar modo\n" +
                          "  - 'auto': Togglear modo automático\n\n" +
                          self.format_battery_response(status)
            }
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa solicitudes del protocolo A2A."""
        try:
            message_content = request.get("message", {}).get("content", "")
            
            # Procesar comando
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
        """Interface de chat para el agente de baterías."""
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
        
        # Procesar comando
        result = self.process_command(content)
        
        self.task_manager.update_task_status(task_id, "completed")
        
        return {
            "task_id": task_id,
            "status": "completed",
            "result": {
                "type": "battery_control",
                "content": result.get("message", ""),
                "data": result.get("data", {})
            }
        }
    
    def _process_task_stream(self, task_id: str) -> Iterator[Dict[str, Any]]:
        """Procesa tarea con streaming. Implementación requerida por IA2AIAAlgorithm."""
        yield {"error": "Streaming not implemented for Battery Agent"}
    
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
    """Ejecuta el servidor del agente de baterías."""
    
    skills = [
        {
            "id": "battery_monitoring",
            "name": "Battery Monitoring",
            "description": "Monitorea el estado del sistema de almacenamiento: SOC, voltaje, corriente, temperatura"
        },
        {
            "id": "charge_control",
            "name": "Charge Control",
            "description": "Controla la carga de las baterías con gestión de potencia y eficiencia"
        },
        {
            "id": "discharge_control",
            "name": "Discharge Control",
            "description": "Controla la descarga de las baterías con optimización energética"
        },
        {
            "id": "battery_health",
            "name": "Battery Health",
            "description": "Monitorea el estado de salud, ciclos de vida y degradación de las baterías"
        }
    ]
    
    parser = argparse.ArgumentParser(description="Run Battery Storage Agent")
    parser.add_argument("--port", type=int, default=8005, help="Puerto del servidor")
    parser.add_argument("--soc", type=float, default=50.0, help="Estado de carga inicial (%)")
    parser.add_argument("--capacity", type=float, default=10.0, help="Capacidad de la batería (kWh)")
    parser.add_argument("--orchestrator-url", type=str, default="http://localhost:8001", help="Orchestrator registry URL for auto-registration")
    
    args = parser.parse_args()
    
    # Crear el agente de baterías
    battery_agent = BatteryAgent(
        name="Battery Storage Agent",
        skills=skills,
        description="Agente de control y monitoreo del sistema de almacenamiento de baterías",
        endpoint=f"http://localhost:{args.port}",
        initial_soc=args.soc,
        capacity_kwh=args.capacity
    )
    
    print(f"🚀 Iniciando Battery Storage Agent en puerto {args.port}")
    print(f"🔋 Capacidad: {args.capacity} kWh")
    print(f"⚡ SOC inicial: {args.soc}%")
    print(f"📊 Estado: {battery_agent.mode}")
    
    # Iniciar el servidor A2A
    run_server(
        port=args.port,
        iaAlgorithm=battery_agent,
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
