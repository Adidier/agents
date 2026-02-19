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
from flask import Flask, render_template_string, jsonify, request

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
        ollama_model: str = "llama2:latest",
        web_port: int = 5000,
        orchestrator_url: str = "http://localhost:8001"
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
            web_port: Puerto para el servidor web del dashboard
            orchestrator_url: URL del orchestrator (para obtener registro de agentes)
        """
        self.mongodb_uri = mongodb_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.refresh_rate = refresh_rate
        self.ollama_host = ollama_host.rstrip("/")
        self.ollama_model = ollama_model
        self.web_port = web_port
        self.orchestrator_url = orchestrator_url.rstrip("/")
        self.last_data = None
        
        # Queue para manejar preguntas del usuario
        self.question_queue = queue.Queue()
        self.answer_queue = queue.Queue()
        self.chat_active = False
        self.stop_threads = False
        
        # Flask app para web dashboard
        self.app = Flask(__name__)
        self._setup_flask_routes()
        self.web_thread = None
        
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
    
    def _setup_flask_routes(self):
        """Configura las rutas Flask para el dashboard web."""
        
        @self.app.route('/')
        def index():
            """Página principal del dashboard."""
            html_template = """
            <!DOCTYPE html>
            <html lang="es">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Agent Registry Dashboard</title>
                <style>
                    * {
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }
                    
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        padding: 20px;
                    }
                    
                    .container {
                        max-width: 1400px;
                        margin: 0 auto;
                    }
                    
                    .header {
                        text-align: center;
                        color: white;
                        margin-bottom: 30px;
                    }
                    
                    .header h1 {
                        font-size: 2.5em;
                        margin-bottom: 10px;
                        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                    }
                    
                    .header p {
                        font-size: 1.2em;
                        opacity: 0.9;
                    }
                    
                    .stats {
                        display: flex;
                        gap: 20px;
                        margin-bottom: 30px;
                        flex-wrap: wrap;
                    }
                    
                    .stat-card {
                        flex: 1;
                        min-width: 200px;
                        background: white;
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    }
                    
                    .stat-card h3 {
                        color: #667eea;
                        font-size: 0.9em;
                        text-transform: uppercase;
                        margin-bottom: 10px;
                    }
                    
                    .stat-card .value {
                        font-size: 2em;
                        font-weight: bold;
                        color: #333;
                    }
                    
                    .diagram-container {
                        background: white;
                        border-radius: 15px;
                        padding: 30px;
                        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
                        margin-bottom: 30px;
                    }
                    
                    .diagram-title {
                        font-size: 1.8em;
                        color: #333;
                        margin-bottom: 30px;
                        text-align: center;
                    }
                    
                    .architecture-diagram {
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        gap: 30px;
                    }
                    
                    .orchestrator-node {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 25px 40px;
                        border-radius: 15px;
                        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
                        text-align: center;
                        min-width: 250px;
                    }
                    
                    .orchestrator-node h3 {
                        font-size: 1.5em;
                        margin-bottom: 5px;
                    }
                    
                    .orchestrator-node .port {
                        font-size: 0.9em;
                        opacity: 0.9;
                    }
                    
                    .connection-line {
                        width: 2px;
                        height: 40px;
                        background: linear-gradient(to bottom, #667eea, #764ba2);
                        position: relative;
                    }
                    
                    .connection-line::before {
                        content: '↕';
                        position: absolute;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        background: white;
                        padding: 2px 5px;
                        border-radius: 3px;
                        font-size: 0.8em;
                    }
                    
                    .agents-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                        gap: 20px;
                        width: 100%;
                    }
                    
                    .agent-node {
                        background: white;
                        border: 3px solid #667eea;
                        border-radius: 10px;
                        padding: 20px;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                        transition: transform 0.3s, box-shadow 0.3s;
                    }
                    
                    .agent-node:hover {
                        transform: translateY(-5px);
                        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
                    }
                    
                    .agent-node h4 {
                        color: #667eea;
                        font-size: 1.3em;
                        margin-bottom: 10px;
                    }
                    
                    .agent-info {
                        font-size: 0.9em;
                        color: #666;
                        line-height: 1.6;
                    }
                    
                    .agent-info strong {
                        color: #333;
                    }
                    
                    .skills-list {
                        margin-top: 10px;
                        padding-top: 10px;
                        border-top: 1px solid #eee;
                    }
                    
                    .skill-tag {
                        display: inline-block;
                        background: #f0f0f0;
                        padding: 3px 8px;
                        border-radius: 5px;
                        font-size: 0.8em;
                        margin: 2px;
                    }
                    
                    .no-agents {
                        text-align: center;
                        padding: 40px;
                        color: #999;
                        font-size: 1.2em;
                    }
                    
                    .refresh-btn {
                        position: fixed;
                        bottom: 30px;
                        right: 30px;
                        background: #667eea;
                        color: white;
                        border: none;
                        padding: 15px 25px;
                        border-radius: 50px;
                        font-size: 1em;
                        cursor: pointer;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                        transition: background 0.3s;
                    }
                    
                    .refresh-btn:hover {
                        background: #764ba2;
                    }
                    
                    .error-message {
                        background: #ff4444;
                        color: white;
                        padding: 15px;
                        border-radius: 8px;
                        margin-bottom: 20px;
                        text-align: center;
                    }
                    
                    .chat-container {
                        background: white;
                        border-radius: 15px;
                        padding: 30px;
                        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
                        margin-bottom: 30px;
                        max-width: 1400px;
                        margin-left: auto;
                        margin-right: auto;
                    }
                    
                    .chat-title {
                        font-size: 1.8em;
                        color: #333;
                        margin-bottom: 20px;
                        text-align: center;
                    }
                    
                    .chat-messages {
                        height: 400px;
                        overflow-y: auto;
                        border: 2px solid #e0e0e0;
                        border-radius: 10px;
                        padding: 20px;
                        margin-bottom: 20px;
                        background: #f9f9f9;
                    }
                    
                    .chat-message {
                        margin-bottom: 15px;
                        padding: 12px 15px;
                        border-radius: 8px;
                        max-width: 80%;
                        word-wrap: break-word;
                    }
                    
                    .chat-message.user {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        margin-left: auto;
                        text-align: right;
                    }
                    
                    .chat-message.assistant {
                        background: white;
                        border: 2px solid #667eea;
                        color: #333;
                        margin-right: auto;
                    }
                    
                    .chat-message .message-header {
                        font-weight: bold;
                        margin-bottom: 5px;
                        font-size: 0.9em;
                        opacity: 0.8;
                    }
                    
                    .chat-input-container {
                        display: flex;
                        gap: 10px;
                    }
                    
                    .chat-input {
                        flex: 1;
                        padding: 15px;
                        border: 2px solid #667eea;
                        border-radius: 10px;
                        font-size: 1em;
                        font-family: inherit;
                        resize: vertical;
                        min-height: 60px;
                    }
                    
                    .chat-input:focus {
                        outline: none;
                        border-color: #764ba2;
                        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
                    }
                    
                    .chat-send-btn {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        border: none;
                        padding: 15px 30px;
                        border-radius: 10px;
                        font-size: 1em;
                        cursor: pointer;
                        transition: transform 0.2s, box-shadow 0.2s;
                        font-weight: bold;
                    }
                    
                    .chat-send-btn:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 6px 15px rgba(102, 126, 234, 0.4);
                    }
                    
                    .chat-send-btn:active {
                        transform: translateY(0);
                    }
                    
                    .chat-send-btn:disabled {
                        opacity: 0.6;
                        cursor: not-allowed;
                        transform: none;
                    }
                    
                    .typing-indicator {
                        display: none;
                        padding: 10px;
                        color: #667eea;
                        font-style: italic;
                    }
                    
                    .typing-indicator.active {
                        display: block;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🌐 Agent Registry Dashboard</h1>
                        <p>Multi-Agent System Architecture</p>
                    </div>
                    
                    <div class="stats" id="stats">
                        <div class="stat-card">
                            <h3>Total Agents</h3>
                            <div class="value" id="total-agents">-</div>
                        </div>
                        <div class="stat-card">
                            <h3>Orchestrator Port</h3>
                            <div class="value" id="orchestrator-port">-</div>
                        </div>
                        <div class="stat-card">
                            <h3>Last Update</h3>
                            <div class="value" id="last-update" style="font-size: 1.2em;">-</div>
                        </div>
                    </div>
                    
                    <div id="error-container"></div>
                    
                    <div class="diagram-container">
                        <h2 class="diagram-title">System Architecture</h2>
                        <div class="architecture-diagram">
                            <div class="orchestrator-node">
                                <h3>🎯 Orchestrator</h3>
                                <p class="port">Registry Server</p>
                                <p class="port" id="orch-port">Port: 8001</p>
                            </div>
                            
                            <div class="connection-line"></div>
                            
                            <div class="agents-grid" id="agents-container">
                                <div class="no-agents">Loading agents...</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="chat-container">
                        <h2 class="chat-title">💬 Chat with AI Assistant</h2>
                        <div class="chat-messages" id="chat-messages">
                            <div class="chat-message assistant">
                                <div class="message-header">🤖 AI Assistant</div>
                                <div>Hello! I can help you understand the agent system data. Ask me anything about the registered agents, their status, or system architecture.</div>
                            </div>
                        </div>
                        <div class="typing-indicator" id="typing-indicator">🤖 AI is thinking...</div>
                        <div class="chat-input-container">
                            <textarea 
                                id="chat-input" 
                                class="chat-input" 
                                placeholder="Ask a question about the system (e.g., 'How many agents are registered?', 'What can the Solar Generator do?')"
                                rows="3"
                            ></textarea>
                            <button class="chat-send-btn" id="chat-send-btn" onclick="sendMessage()">
                                📤 Send
                            </button>
                        </div>
                    </div>
                </div>
                
                <button class="refresh-btn" onclick="loadAgents()">🔄 Refresh</button>
                
                <script>
                    async function loadAgents() {
                        try {
                            const response = await fetch('/api/agents');
                            const data = await response.json();
                            
                            document.getElementById('error-container').innerHTML = '';
                            
                            if (data.error) {
                                document.getElementById('error-container').innerHTML = 
                                    `<div class="error-message">⚠️ ${data.error}</div>`;
                                return;
                            }
                            
                            // Update stats
                            document.getElementById('total-agents').textContent = data.count || 0;
                            document.getElementById('orchestrator-port').textContent = data.registry_port || '8001';
                            document.getElementById('last-update').textContent = 
                                new Date().toLocaleTimeString();
                            
                            // Update agents grid
                            const container = document.getElementById('agents-container');
                            
                            if (!data.agents || data.agents.length === 0) {
                                container.innerHTML = '<div class="no-agents">No agents registered</div>';
                                return;
                            }
                            
                            container.innerHTML = data.agents.map(agent => `
                                <div class="agent-node">
                                    <h4>🤖 ${agent.name}</h4>
                                    <div class="agent-info">
                                        <p><strong>ID:</strong> ${agent.agent_id.substring(0, 8)}...</p>
                                        <p><strong>Endpoint:</strong> ${agent.endpoint}</p>
                                        ${agent.skills && agent.skills.length > 0 ? `
                                            <div class="skills-list">
                                                <strong>Skills:</strong><br>
                                                ${agent.skills.map(skill => 
                                                    `<span class="skill-tag">${typeof skill === 'object' ? (skill.name || skill.id || JSON.stringify(skill)) : skill}</span>`
                                                ).join('')}
                                            </div>
                                        ` : ''}
                                    </div>
                                </div>
                            `).join('');
                            
                        } catch (error) {
                            document.getElementById('error-container').innerHTML = 
                                `<div class="error-message">❌ Error loading agents: ${error.message}</div>`;
                            console.error('Error loading agents:', error);
                        }
                    }
                    
                    // Load agents on page load
                    loadAgents();
                    
                    // Auto-refresh every 10 seconds
                    setInterval(loadAgents, 10000);
                    
                    // Chat functionality
                    async function sendMessage() {
                        const input = document.getElementById('chat-input');
                        const sendBtn = document.getElementById('chat-send-btn');
                        const messagesContainer = document.getElementById('chat-messages');
                        const typingIndicator = document.getElementById('typing-indicator');
                        
                        const question = input.value.trim();
                        if (!question) return;
                        
                        // Disable input and button
                        input.disabled = true;
                        sendBtn.disabled = true;
                        
                        // Add user message
                        const userMessage = document.createElement('div');
                        userMessage.className = 'chat-message user';
                        userMessage.innerHTML = `
                            <div class="message-header">👤 You</div>
                            <div>${escapeHtml(question)}</div>
                        `;
                        messagesContainer.appendChild(userMessage);
                        
                        // Clear input
                        input.value = '';
                        
                        // Show typing indicator
                        typingIndicator.classList.add('active');
                        
                        // Scroll to bottom
                        messagesContainer.scrollTop = messagesContainer.scrollHeight;
                        
                        try {
                            const response = await fetch('/api/chat', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json'
                                },
                                body: JSON.stringify({ question: question })
                            });
                            
                            const data = await response.json();
                            
                            // Hide typing indicator
                            typingIndicator.classList.remove('active');
                            
                            if (data.error) {
                                throw new Error(data.error);
                            }
                            
                            // Add assistant response
                            const assistantMessage = document.createElement('div');
                            assistantMessage.className = 'chat-message assistant';
                            assistantMessage.innerHTML = `
                                <div class="message-header">🤖 AI Assistant</div>
                                <div>${escapeHtml(data.answer)}</div>
                            `;
                            messagesContainer.appendChild(assistantMessage);
                            
                        } catch (error) {
                            typingIndicator.classList.remove('active');
                            
                            const errorMessage = document.createElement('div');
                            errorMessage.className = 'chat-message assistant';
                            errorMessage.innerHTML = `
                                <div class="message-header">⚠️ Error</div>
                                <div>Sorry, I couldn't process your question: ${escapeHtml(error.message)}</div>
                            `;
                            messagesContainer.appendChild(errorMessage);
                        }
                        
                        // Re-enable input and button
                        input.disabled = false;
                        sendBtn.disabled = false;
                        input.focus();
                        
                        // Scroll to bottom
                        messagesContainer.scrollTop = messagesContainer.scrollHeight;
                    }
                    
                    // Handle Enter key in textarea (Shift+Enter for new line)
                    document.getElementById('chat-input').addEventListener('keydown', function(e) {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            sendMessage();
                        }
                    });
                    
                    // Escape HTML to prevent XSS
                    function escapeHtml(text) {
                        const div = document.createElement('div');
                        div.textContent = text;
                        return div.innerHTML;
                    }
                </script>
            </body>
            </html>
            """
            return render_template_string(html_template)
        
        @self.app.route('/api/agents')
        def api_agents():
            """API endpoint para obtener la lista de agentes registrados."""
            try:
                # Primero intentar obtener del orchestrator
                response = requests.get(f"{self.orchestrator_url}/agents", timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    data['registry_port'] = self.orchestrator_url.split(':')[-1]
                    return jsonify(data)
            except Exception as e:
                # Si falla, intentar leer desde MongoDB
                try:
                    db = self.mongo_client[self.db_name]
                    registry_collection = db['agent_registry']
                    
                    # Obtener el snapshot más reciente
                    latest = registry_collection.find_one(sort=[("timestamp", -1)])
                    
                    if latest:
                        return jsonify({
                            "agents": latest.get('agents', []),
                            "count": latest.get('total_agents', 0),
                            "registry_port": latest.get('registry_port', '8001'),
                            "source": "mongodb"
                        })
                except Exception as mongo_error:
                    pass
            
            return jsonify({
                "error": "Could not fetch agents from orchestrator or MongoDB",
                "agents": [],
                "count": 0
            })
        
        @self.app.route('/api/chat', methods=['POST'])
        def api_chat():
            """API endpoint para chat con el LLM usando datos del sistema."""
            try:
                data = request.json
                question = data.get('question', '')
                
                if not question:
                    return jsonify({"error": "No question provided"}), 400
                
                # Get context about registered agents
                try:
                    response = requests.get(f"{self.orchestrator_url}/agents", timeout=2)
                    if response.status_code == 200:
                        agents_data = response.json()
                        agent_list = agents_data.get('agents', [])
                        
                        # Build context
                        context_parts = ["Current System Status:\n"]
                        context_parts.append(f"- Total Agents Registered: {len(agent_list)}\n")
                        
                        for agent in agent_list:
                            context_parts.append(f"\nAgent: {agent['name']}")
                            context_parts.append(f"  Endpoint: {agent['endpoint']}")
                            if agent.get('skills'):
                                skills_names = []
                                for skill in agent['skills']:
                                    if isinstance(skill, dict):
                                        skills_names.append(skill.get('name', skill.get('id', str(skill))))
                                    else:
                                        skills_names.append(str(skill))
                                context_parts.append(f"  Skills: {', '.join(skills_names)}")
                        
                        context = "\n".join(context_parts)
                    else:
                        context = "Could not retrieve current agent data from orchestrator."
                except Exception as e:
                    context = f"Error retrieving agent data: {str(e)}"
                
                # Get monitoring data context
                try:
                    monitoring_data = self.read_mongodb_data()
                    if monitoring_data and monitoring_data.get('history'):
                        latest = monitoring_data['history'][-1]
                        context += f"\n\nLatest Monitoring Data (Iteration #{latest.get('iteration')}):\n"
                        for agent_name, agent_data in latest.get('agents', {}).items():
                            context += f"\n{agent_name.upper()}:\n"
                            if isinstance(agent_data, dict):
                                content = agent_data.get('content', str(agent_data))
                                # Limit content length
                                if len(content) > 500:
                                    content = content[:500] + "..."
                                context += f"  {content}\n"
                except Exception as e:
                    pass
                
                # Call Ollama
                answer = self.ask_ollama(question, context)
                
                return jsonify({
                    "answer": answer,
                    "context_used": len(context) > 0
                })
                
            except Exception as e:
                return jsonify({"error": str(e)}), 500
    
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
    
    def ask_ollama(self, question: str, context: str = None) -> str:
        """
        Hace una pregunta a Ollama con el contexto de los datos actuales.
        
        Args:
            question: Pregunta del usuario
            context: Contexto opcional (si no se proporciona, se obtiene automáticamente)
            
        Returns:
            Respuesta del LLM
        """
        try:
            # Obtener contexto de los datos si no se proporciona
            if context is None:
                context = self.get_context_for_llm()
            
            # Imprimir el contexto (solo en terminal, no en web)
            if context == self.get_context_for_llm():
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
    
    def _start_web_server(self):
        """Inicia el servidor web Flask en un thread separado."""
        def run_flask():
            self.app.run(host='0.0.0.0', port=self.web_port, debug=False, use_reloader=False)
        
        self.web_thread = threading.Thread(target=run_flask, daemon=True)
        self.web_thread.start()
        print(f"{Colors.GREEN}🌐 Web Dashboard started on http://localhost:{self.web_port}{Colors.RESET}")
    
    def run(self):
        """Ejecuta el dashboard en modo continuo con chat interactivo."""
        print(f"{Colors.BOLD}{Colors.GREEN}Dashboard Agent iniciado{Colors.RESET}")
        print(f"Monitoreando MongoDB: {self.db_name}.{self.collection_name}")
        print(f"Tasa de actualización: {self.refresh_rate} segundos")
        print(f"Modelo Ollama: {self.ollama_model} @ {self.ollama_host}")
        print(f"\n{Colors.CYAN}💡 Puedes hacer preguntas sobre los datos en cualquier momento{Colors.RESET}\n")
        
        # Iniciar servidor web
        self._start_web_server()
        
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
    parser.add_argument(
        "--web-port",
        type=int,
        default=5000,
        help="Web dashboard port (default: 5000)"
    )
    parser.add_argument(
        "--orchestrator-url",
        type=str,
        default="http://localhost:8001",
        help="Orchestrator URL for agent registry (default: http://localhost:8001)"
    )
    
    args = parser.parse_args()
    
    # Crear y ejecutar el dashboard
    dashboard = Dashboard(
        mongodb_uri=args.mongodb_uri,
        db_name=args.db_name,
        collection_name=args.collection,
        refresh_rate=args.refresh,
        ollama_host=args.ollama_host,
        ollama_model=args.ollama_model,
        web_port=args.web_port,
        orchestrator_url=args.orchestrator_url
    )
    dashboard.run()


if __name__ == "__main__":
    main()
