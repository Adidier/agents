"""
Dashboard Agent - Real-time monitoring display

Este agente lee el archivo JSON generado por el orchestrator y muestra
el estado actual del sistema en tiempo real en la terminal.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

# ANSI color codes for terminal output
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Regular colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    
    # Background colors
    BG_RED = '\033[101m'
    BG_GREEN = '\033[102m'
    BG_YELLOW = '\033[103m'


class Dashboard:
    """
    Dashboard para monitorear el estado del sistema multi-agente.
    """
    
    def __init__(self, json_file: str = "orchestrator_data.json", refresh_rate: int = 5):
        """
        Inicializa el dashboard.
        
        Args:
            json_file: Ruta al archivo JSON del orchestrator
            refresh_rate: Segundos entre actualizaciones
        """
        self.json_file = json_file
        self.refresh_rate = refresh_rate
        self.last_data = None
        self.last_modified = 0
    
    def clear_screen(self):
        """Limpia la pantalla de la terminal."""
        os.system('clear' if os.name != 'nt' else 'cls')
    
    def read_json_file(self) -> Optional[Dict[str, Any]]:
        """
        Lee el archivo JSON del orchestrator.
        
        Returns:
            Datos del JSON o None si hay error
        """
        try:
            if not os.path.exists(self.json_file):
                return None
            
            # Verificar si el archivo fue modificado
            current_modified = os.path.getmtime(self.json_file)
            
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.last_modified = current_modified
                return data
        except json.JSONDecodeError as e:
            print(f"{Colors.RED}Error al leer JSON: {e}{Colors.RESET}")
            return None
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.RESET}")
            return None
    
    def format_timestamp(self, iso_timestamp: str) -> str:
        """Formatea un timestamp ISO a formato legible."""
        try:
            dt = datetime.fromisoformat(iso_timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            return iso_timestamp
    
    def get_supervision_color(self, light_status: str) -> str:
        """Obtiene el color basado en el estado de supervisión."""
        if light_status == "green":
            return Colors.GREEN
        elif light_status == "yellow":
            return Colors.YELLOW
        elif light_status == "red":
            return Colors.RED
        return Colors.WHITE
        return Colors.WHITE
    
    def display_agent_data(self, agent_name: str, agent_data: Dict[str, Any]):
        """
        Muestra los datos de un agente de forma formateada.
        
        Args:
            agent_name: Nombre del agente
            agent_data: Datos del agente
        """
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'─' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}🤖 {agent_name.upper()} AGENT{Colors.RESET}")
        print(f"{Colors.CYAN}{'─' * 60}{Colors.RESET}")
        
        if not agent_data:
            print(f"{Colors.YELLOW}  No data available{Colors.RESET}")
            return
        
        # Verificar si hay un resultado
        if "result" in agent_data and isinstance(agent_data["result"], dict):
            result = agent_data["result"]
            
            # Datos de clasificación PV (Solar)
            if result.get("type") == "pv_classification":
                pred = result.get("predicted_power", 0)
                real = result.get("real_power", 0)
                deviation = result.get("deviation_percent", 0)
                deviation_instant = result.get("deviation_instant", 0)
                scenario = result.get("scenario", "UNKNOWN")
                supervision = result.get("supervision", {})
                metrics = result.get("metrics", {})
                
                # Emoji del escenario
                scenario_emoji = {
                    "NORMAL": "🟢",
                    "DEGRADED": "🟡",
                    "FAULT": "🔴"
                }.get(scenario, "⚪")
                
                # Color basado en el estado de supervisión
                light_status = supervision.get("light_status", "unknown")
                status_color = self.get_supervision_color(light_status)
                emoji = supervision.get("light_emoji", "⚪")
                message = supervision.get("message", "")
                
                print(f"  {scenario_emoji} {Colors.BOLD}Escenario:{Colors.RESET} {scenario}")
                print(f"  📊 {Colors.BOLD}Predicción:{Colors.RESET} {pred:.2f} kW")
                print(f"  📈 {Colors.BOLD}Real (Sim):{Colors.RESET} {real:.2f} kW")
                print(f"  {emoji} {status_color}{Colors.BOLD}Estado:{Colors.RESET} {message}{Colors.RESET}")
                print(f"  📉 {Colors.BOLD}Desviación:{Colors.RESET} {deviation:.2f}% (Inst: {deviation_instant:.2f}%)")
                
                if metrics:
                    mae = metrics.get("MAE", 0)
                    rmse = metrics.get("RMSE", 0)
                    print(f"  📐 {Colors.BOLD}Métricas:{Colors.RESET} MAE: {mae:.4f} | RMSE: {rmse:.4f}")
                
                # Mostrar estadísticas históricas si están disponibles
                if "recent_stats" in supervision and supervision.get("history_size", 0) > 0:
                    stats = supervision["recent_stats"]
                    total = stats.get("total", 0)
                    if total > 0:
                        print(f"  📊 {Colors.BOLD}Últimas {total} mediciones:{Colors.RESET}")
                        print(f"     {Colors.GREEN}🟢 {stats.get('green', 0)}{Colors.RESET} | "
                              f"{Colors.YELLOW}🟡 {stats.get('yellow', 0)}{Colors.RESET} | "
                              f"{Colors.RED}🔴 {stats.get('red', 0)}{Colors.RESET}")
            
            # Otros resultados (Weather, etc.)
            else:
                if "formatted" in result:
                    print(f"  {result['formatted']}")
                elif "value" in result:
                    unit = result.get("unit", "")
                    print(f"  📊 Valor: {result['value']} {unit}")
                else:
                    # Mostrar resultado genérico
                    for key, value in result.items():
                        if key not in ["type", "supervision"]:
                            print(f"  {Colors.BOLD}{key}:{Colors.RESET} {value}")
        
        # Si es una respuesta de mensaje simple
        elif "message" in agent_data and agent_data["message"]:
            message = agent_data["message"]
            if "parts" in message:
                for part in message["parts"]:
                    if part.get("type") == "text" and part.get("content"):
                        print(f"  💬 {part['content']}")
    
    def display_header(self, metadata: Dict[str, Any]):
        """
        Muestra el encabezado del dashboard.
        
        Args:
            metadata: Metadatos del orchestrator
        """
        total_iterations = metadata.get("total_iterations", 0)
        agents = metadata.get("agents", [])
        last_updated = metadata.get("last_updated", "")
        
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'═' * 80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'█' * 30} SYSTEM DASHBOARD {'█' * 30}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'═' * 80}{Colors.RESET}")
        
        print(f"\n{Colors.BOLD}📊 Total Iterations:{Colors.RESET} {total_iterations}")
        print(f"{Colors.BOLD}🤖 Active Agents:{Colors.RESET} {', '.join(agents)}")
        print(f"{Colors.BOLD}🕐 Last Updated:{Colors.RESET} {self.format_timestamp(last_updated)}")
    
    def display_dashboard(self):
        """Muestra el dashboard completo."""
        data = self.read_json_file()
        
        if not data:
            print(f"{Colors.YELLOW}⏳ Esperando datos del orchestrator...{Colors.RESET}")
            print(f"{Colors.YELLOW}   Archivo: {self.json_file}{Colors.RESET}")
            return
        
        # Mostrar encabezado
        metadata = data.get("metadata", {})
        self.display_header(metadata)
        
        # Obtener la última iteración
        history = data.get("history", [])
        if not history:
            print(f"\n{Colors.YELLOW}No hay datos históricos disponibles{Colors.RESET}")
            return
        
        latest = history[-1]
        iteration = latest.get("iteration", 0)
        timestamp = latest.get("timestamp", "")
        agents_data = latest.get("agents", {})
        
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}{'═' * 80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.MAGENTA}🔄 ITERATION #{iteration} - {self.format_timestamp(timestamp)}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.MAGENTA}{'═' * 80}{Colors.RESET}")
        
        # Mostrar datos de cada agente
        for agent_name, agent_data in agents_data.items():
            self.display_agent_data(agent_name, agent_data)
        
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'═' * 80}{Colors.RESET}")
        print(f"{Colors.CYAN}🔄 Actualizando cada {self.refresh_rate} segundos... (Ctrl+C para salir){Colors.RESET}")
    
    def run(self):
        """Ejecuta el dashboard en modo continuo."""
        print(f"{Colors.BOLD}{Colors.GREEN}Dashboard Agent iniciado{Colors.RESET}")
        print(f"Monitoreando: {self.json_file}")
        print(f"Tasa de actualización: {self.refresh_rate} segundos\n")
        
        try:
            while True:
                self.clear_screen()
                self.display_dashboard()
                time.sleep(self.refresh_rate)
        
        except KeyboardInterrupt:
            self.clear_screen()
            print(f"\n{Colors.BOLD}{Colors.GREEN}Dashboard Agent detenido{Colors.RESET}")
            print(f"Adiós! 👋\n")
        
        except Exception as e:
            print(f"\n{Colors.RED}Error inesperado: {e}{Colors.RESET}") 


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="A2A Multi-Agent Dashboard")
    parser.add_argument(
        "--input",
        type=str,
        default="orchestrator_data.json",
        help="Path to orchestrator JSON file"
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=5,
        help="Refresh rate in seconds (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Crear y ejecutar el dashboard
    dashboard = Dashboard(json_file=args.input, refresh_rate=args.refresh)
    dashboard.run()


if __name__ == "__main__":
    main()
