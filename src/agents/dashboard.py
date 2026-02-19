"""
Dashboard Agent - Real-time monitoring display

Este agente lee los datos desde MongoDB y muestra
el estado actual del sistema en tiempo real en la terminal.
El usuario puede hacer preguntas al LLM de Ollama sobre los datos mostrados.
"""

import os
import sys
import json
import time
import argparse
import threading
import queue
import requests
from datetime import datetime
from typing import Dict, Any, Optional
from pymongo import MongoClient

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
    Incluye chat con LLM para consultas sobre los datos.
    """
    
    def __init__(
        self, 
        mongodb_uri: str = "mongodb://localhost:27017/",
        db_name: str = "solar_energy",
        collection_name: str = "agent_data",
        refresh_rate: int = 30,
        ollama_host: str = "http://localhost:11434",
        ollama_model: str = "llama2:latest"
    ):
        """
        Inicializa el dashboard.
        
        Args:
            mongodb_uri: URI de conexión a MongoDB
            db_name: Nombre de la base de datos MongoDB
            collection_name: Nombre de la colección MongoDB
            refresh_rate: Segundos entre actualizaciones
            ollama_host: URL del servidor Ollama
            ollama_model: Modelo de Ollama a usar
        """
        self.mongodb_uri = mongodb_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.refresh_rate = refresh_rate
        self.ollama_host = ollama_host.rstrip("/")
        self.ollama_model = ollama_model
        self.last_data = None
        
        # Queue para manejar preguntas del usuario
        self.question_queue = queue.Queue()
        self.answer_queue = queue.Queue()
        self.chat_active = False
        self.stop_threads = False
        
        # Inicializar MongoDB
        self._init_mongodb()
    
    def clear_screen(self):
        """Limpia la pantalla de la terminal."""
        os.system('clear' if os.name != 'nt' else 'cls')
    
    def _init_mongodb(self):
        """Inicializa la conexión a MongoDB."""
        try:
            self.mongo_client = MongoClient(
                self.mongodb_uri,
                serverSelectionTimeoutMS=5000
            )
            # Test connection
            self.mongo_client.admin.command('ping')
            
            # Get database and collection
            db = self.mongo_client[self.db_name]
            self.mongo_collection = db[self.collection_name]
            
            print(f"{Colors.GREEN}✅ Conectado a MongoDB{Colors.RESET}")
            print(f"{Colors.CYAN}   Database: {self.db_name}{Colors.RESET}")
            print(f"{Colors.CYAN}   Collection: {self.collection_name}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}❌ Error al conectar a MongoDB: {e}{Colors.RESET}")
            sys.exit(1)
    
    def read_mongodb_data(self) -> Optional[Dict[str, Any]]:
        """
        Lee los datos desde MongoDB.
        
        Returns:
            Datos formateados o None si hay error
        """
        try:
            # Obtener todos los documentos ordenados por timestamp
            cursor = self.mongo_collection.find().sort("timestamp", -1)
            documents = list(cursor)
            
            if not documents:
                return None
            
            # Construir estructura compatible con el formato anterior
            history = []
            for doc in reversed(documents):  # Revertir para orden cronológico
                history.append({
                    "iteration": doc.get("iteration"),
                    "timestamp": doc.get("timestamp").isoformat() if hasattr(doc.get("timestamp"), 'isoformat') else str(doc.get("timestamp")),
                    "agents": doc.get("agents", {})
                })
            
            # Metadata
            latest = documents[0]
            metadata = {
                "total_iterations": len(documents),
                "agents": list(latest.get("agents", {}).keys()),
                "last_updated": latest.get("timestamp").isoformat() if hasattr(latest.get("timestamp"), 'isoformat') else str(latest.get("timestamp")),
                "mongodb_enabled": True,
                "db_info": {
                    "database": self.db_name,
                    "collection": self.collection_name
                }
            }
            
            return {
                "metadata": metadata,
                "history": history
            }
            
        except Exception as e:
            print(f"{Colors.RED}Error al leer MongoDB: {e}{Colors.RESET}")
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
    
    def get_context_for_llm(self) -> str:
        """
        Genera un contexto resumido de los datos actuales para el LLM.
        
        Returns:
            String con el contexto de los datos
        """
        if not self.last_data:
            return "No hay datos disponibles actualmente."
        
        history = self.last_data.get("history", [])
        if not history:
            return "No hay histórico de datos disponible."
        
        latest = history[-1]
        agents_data = latest.get("agents", {})
        
        context_parts = [
            f"Dashboard del sistema multi-agente - Iteración #{latest.get('iteration', 0)}",
            f"Timestamp: {latest.get('timestamp', '')}",
            "\nDatos actuales de los agentes:\n"
        ]
        
        for agent_name, agent_data in agents_data.items():
            context_parts.append(f"\n{agent_name.upper()} Agent:")
            
            if "result" in agent_data and isinstance(agent_data["result"], dict):
                result = agent_data["result"]
                
                # Clasificación PV
                if result.get("type") == "pv_classification":
                    pred = result.get("predicted_power", 0)
                    real = result.get("real_power", 0)
                    deviation = result.get("deviation_percent", 0)
                    scenario = result.get("scenario", "UNKNOWN")
                    supervision = result.get("supervision", {})
                    
                    context_parts.append(f"  - Escenario: {scenario}")
                    context_parts.append(f"  - Predicción: {pred:.2f} kW")
                    context_parts.append(f"  - Potencia real: {real:.2f} kW")
                    context_parts.append(f"  - Desviación: {deviation:.2f}%")
                    context_parts.append(f"  - Estado supervisión: {supervision.get('light_status', 'unknown')}")
                    context_parts.append(f"  - Mensaje: {supervision.get('message', '')}")
                
                # Energy Price Predictor (CENACE)
                elif result.get("type") == "cenace_market_data":
                    current_price = result.get("current_price", {})
                    statistics = result.get("statistics", {})
                    analysis = result.get("analysis", {})
                    
                    price = current_price.get("price", 0)
                    currency = current_price.get("currency", "MXN/MWh")
                    node = current_price.get("node", "N/A")
                    
                    context_parts.append(f"  - Precio actual: ${price:.2f} {currency}")
                    context_parts.append(f"  - Nodo: {node}")
                    
                    if statistics and statistics.get("status") != "no_data":
                        avg_24h = statistics.get("average_24h", 0)
                        min_24h = statistics.get("min_24h", 0)
                        max_24h = statistics.get("max_24h", 0)
                        context_parts.append(f"  - Promedio 24h: ${avg_24h:.2f}")
                        context_parts.append(f"  - Rango 24h: ${min_24h:.2f} - ${max_24h:.2f}")
                    
                    if analysis and analysis.get("status") == "active":
                        condition = analysis.get("condition", "unknown")
                        recommendation = analysis.get("recommendation", "")
                        context_parts.append(f"  - Condición del mercado: {condition}")
                        context_parts.append(f"  - Recomendación: {recommendation}")
                
                # Otros tipos de datos
                else:
                    if "content" in result:
                        content = result["content"]
                        # Limitar la longitud del contenido
                        if len(content) > 500:
                            content = content[:500] + "..."
                        context_parts.append(f"  {content}")
                    else:
                        for key, value in result.items():
                            if key not in ["type", "supervision"]:
                                context_parts.append(f"  - {key}: {value}")
        
        return "\n".join(context_parts)
    
    def ask_ollama(self, question: str) -> str:
        """
        Hace una pregunta a Ollama con el contexto de los datos actuales.
        
        Args:
            question: Pregunta del usuario
            
        Returns:
            Respuesta del LLM
        """
        try:
            # Obtener contexto de los datos
            context = self.get_context_for_llm()
            
            # Imprimir el contexto
            print(f"\n{Colors.MAGENTA}{'─' * 80}{Colors.RESET}")
            print(f"{Colors.MAGENTA}{Colors.BOLD}📋 CONTEXTO ENVIADO A OLLAMA:{Colors.RESET}")
            print(f"{Colors.MAGENTA}{'─' * 80}{Colors.RESET}")
            print(f"{Colors.WHITE}{context}{Colors.RESET}")
            print(f"{Colors.MAGENTA}{'─' * 80}{Colors.RESET}\n")
            
            # Preparar el prompt con contexto
            prompt = f"""Eres un asistente experto en sistemas de energía solar y monitoreo de sistemas multi-agente.

Contexto actual del sistema:
{context}

Pregunta del usuario: {question}

Proporciona una respuesta clara y concisa basada en los datos mostrados arriba."""
            
            # Hacer request a Ollama
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No se recibió respuesta del modelo.")
            else:
                return f"Error al consultar Ollama: HTTP {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return f"❌ No se puede conectar con Ollama en {self.ollama_host}. ¿Está corriendo?"
        except requests.exceptions.Timeout:
            return "❌ Timeout al consultar Ollama. El modelo puede estar ocupado."
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def input_thread_func(self):
        """Thread que lee input del usuario de forma asíncrona."""
        while not self.stop_threads:
            try:
                # Mostrar prompt solo si no hay chat activo
                if not self.chat_active:
                    print(f"\n{Colors.CYAN}💬 Pregunta (o 'Enter' para continuar): {Colors.RESET}", end="", flush=True)
                
                question = input().strip()
                
                if question:
                    self.chat_active = True
                    self.question_queue.put(question)
                    
                    # Esperar respuesta
                    answer = self.answer_queue.get()
                    
                    # Mostrar respuesta
                    print(f"\n{Colors.GREEN}🤖 Ollama:{Colors.RESET}")
                    print(f"{Colors.WHITE}{answer}{Colors.RESET}")
                    print(f"\n{Colors.YELLOW}Presiona Enter para continuar...{Colors.RESET}", end="", flush=True)
                    input()
                    
                    self.chat_active = False
                else:
                    # Solo Enter presionado, continuar con refresh
                    time.sleep(0.1)
                    
            except EOFError:
                break
            except Exception as e:
                print(f"{Colors.RED}Error en input thread: {e}{Colors.RESET}")
                break
    
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
            
            # Datos del Energy Price Predictor (CENACE)
            elif result.get("type") == "cenace_market_data":
                current_price_data = result.get("current_price", {})
                statistics = result.get("statistics", {})
                analysis = result.get("analysis", {})
                
                # Precio actual
                price = current_price_data.get("price", 0)
                currency = current_price_data.get("currency", "MXN/MWh")
                node = current_price_data.get("node", "N/A")
                timestamp = current_price_data.get("timestamp", "N/A")
                
                print(f"  💰 {Colors.BOLD}Precio Actual:{Colors.RESET} ${price:.2f} {currency}")
                print(f"  📍 {Colors.BOLD}Nodo:{Colors.RESET} {node}")
                print(f"  🕐 {Colors.BOLD}Timestamp:{Colors.RESET} {timestamp}")
                
                # Estadísticas
                if statistics and statistics.get("status") != "no_data":
                    avg_24h = statistics.get("average_24h", 0)
                    min_24h = statistics.get("min_24h", 0)
                    max_24h = statistics.get("max_24h", 0)
                    samples = statistics.get("samples", 0)
                    
                    print(f"\n  📊 {Colors.BOLD}Estadísticas 24h:{Colors.RESET}")
                    print(f"     Promedio: ${avg_24h:.2f} | Mín: ${min_24h:.2f} | Máx: ${max_24h:.2f}")
                    print(f"     Muestras: {samples}")
                
                # Análisis del mercado
                if analysis and analysis.get("status") == "active":
                    condition = analysis.get("condition", "unknown")
                    condition_emoji = {
                        "low_price": "🟢",
                        "normal": "🟡",
                        "high_price": "🔴"
                    }.get(condition, "⚪")
                    
                    price_ratio = analysis.get("price_ratio", 1.0)
                    recommendation = analysis.get("recommendation", "")
                    light_status = analysis.get("light_status", "yellow")
                    
                    # Color basado en el estado del mercado
                    market_color = self.get_supervision_color(light_status)
                    
                    print(f"\n  {condition_emoji} {market_color}{Colors.BOLD}Condición del Mercado:{Colors.RESET} {condition.replace('_', ' ').title()}{Colors.RESET}")
                    print(f"  📈 {Colors.BOLD}Ratio vs Promedio:{Colors.RESET} {price_ratio:.2f}x")
                    print(f"  💡 {Colors.BOLD}Recomendación:{Colors.RESET} {recommendation}")
            
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
        data = self.read_mongodb_data()
        
        if not data:
            print(f"{Colors.YELLOW}⏳ Esperando datos de MongoDB...{Colors.RESET}")
            print(f"{Colors.YELLOW}   Database: {self.db_name}.{self.collection_name}{Colors.RESET}")
            print(f"{Colors.YELLOW}   Verifica que el orchestrator esté ejecutándose con MongoDB habilitado{Colors.RESET}")
            return
        
        # Guardar los datos en last_data para uso del LLM
        self.last_data = data
        
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
        """Ejecuta el dashboard en modo continuo con chat interactivo."""
        print(f"{Colors.BOLD}{Colors.GREEN}Dashboard Agent iniciado{Colors.RESET}")
        print(f"Monitoreando MongoDB: {self.db_name}.{self.collection_name}")
        print(f"Tasa de actualización: {self.refresh_rate} segundos")
        print(f"Modelo Ollama: {self.ollama_model} @ {self.ollama_host}")
        print(f"\n{Colors.CYAN}💡 Puedes hacer preguntas sobre los datos en cualquier momento{Colors.RESET}\n")
        
        # Iniciar thread de input
        input_thread = threading.Thread(target=self.input_thread_func, daemon=True)
        input_thread.start()
        
        try:
            last_refresh = time.time()
            
            while True:
                current_time = time.time()
                
                # Actualizar dashboard si ha pasado el tiempo de refresh
                if current_time - last_refresh >= self.refresh_rate and not self.chat_active:
                    self.display_dashboard()
                    last_refresh = current_time
                
                # Procesar preguntas del usuario
                try:
                    question = self.question_queue.get_nowait()
                    
                    # Mostrar que estamos procesando
                    print(f"\n{Colors.YELLOW}🤔 Consultando a Ollama...{Colors.RESET}", flush=True)
                    
                    # Obtener respuesta
                    answer = self.ask_ollama(question)
                    
                    # Poner respuesta en queue
                    self.answer_queue.put(answer)
                    
                except queue.Empty:
                    pass
                
                # Sleep corto para no consumir mucho CPU
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            self.stop_threads = True
            print(f"\n{Colors.BOLD}{Colors.GREEN}Dashboard Agent detenido{Colors.RESET}")
            print(f"Adiós! 👋\n")
        
        except Exception as e:
            self.stop_threads = True
            print(f"\n{Colors.RED}Error inesperado: {e}{Colors.RESET}") 


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="A2A Multi-Agent Dashboard with MongoDB and Ollama Chat")
    parser.add_argument(
        "--mongodb-uri",
        type=str,
        default="mongodb://localhost:27017/",
        help="MongoDB connection URI (default: mongodb://localhost:27017/)"
    )
    parser.add_argument(
        "--db-name",
        type=str,
        default="solar_energy",
        help="MongoDB database name (default: solar_energy)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="agent_data",
        help="MongoDB collection name (default: agent_data)"
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=30,
        help="Refresh rate in seconds (default: 30)"
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="deepseek-r1:1.5b",
        help="Ollama model to use (default:deepseek-r1:1.5b)"
    )
    
    args = parser.parse_args()
    
    # Crear y ejecutar el dashboard
    dashboard = Dashboard(
        mongodb_uri=args.mongodb_uri,
        db_name=args.db_name,
        collection_name=args.collection,
        refresh_rate=args.refresh,
        ollama_host=args.ollama_host,
        ollama_model=args.ollama_model
    )
    dashboard.run()


if __name__ == "__main__":
    main()
