# Agentes de Energía Solar

Sistema multi-agente para gestión inteligente de energía solar con almacenamiento.

## 🚀 Inicio Rápido

### 1. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 2. Iniciar MongoDB (Opcional pero recomendado)
```bash
# Usando Docker
docker run -d -p 27017:27017 --name mongodb mongo:latest

# O instalar localmente
# Ubuntu/Debian: sudo apt install mongodb
# macOS: brew install mongodb-community
```

### 3. Lanzar Todos los Agentes
```bash
# Inicia todos los agentes del sistema en segundo plano
./launch_all_agents.sh

# Los agentes se inician en los siguientes puertos:
# - Generator (Solar): 8002
# - Weather: 8004
# - Battery: 8005
# - Load: 8006
# - Energy Price Predictor: 8007
```

### 4. Iniciar Orchestrator (Nueva terminal)
```bash
# Coordina todos los agentes y guarda en MongoDB
./launch_orchestrator.sh
```

### 5. Visualizar Dashboard (Nueva terminal)
```bash
# Monitoreo en tiempo real con datos del mercado eléctrico
./launch_dashboard.sh

# El dashboard muestra:
# - Generación solar y predicciones LSTM
# - Precios de energía en tiempo real 💰
# - Condiciones del mercado eléctrico
# - Estado de baterías y consumo
# - Recomendaciones inteligentes basadas en precios

# Dashboard Web con Diagrama de Arquitectura:
# Accede a http://localhost:5000 en tu navegador
# - Visualización interactiva de agentes registrados 🌐
# - Diagrama de arquitectura del sistema multi-agente
# - Actualización automática cada 10 segundos
# - Vista de endpoints y skills de cada agente
```

---

## 🎯 Flujo de Trabajo Completo

1. **Agentes recopilan datos** → Generator, Weather, Battery, Load, Energy Price Predictor
2. **Orchestrator coordina** → Consulta a todos los agentes cada 10 segundos
3. **Datos se almacenan** → MongoDB (solar_energy.agent_data)
4. **Dashboard visualiza** → Muestra datos en tiempo real con análisis del mercado
5. **Sistema experto decide** → Recomendaciones basadas en precios de energía

---

## 📺 Dashboard
```bash
# Con MongoDB y todos los agentes (recomendado)
./launch_orchestrator.sh

# Solo con JSON local
python src/agents/orchestrator.py \
  --generator-endpoint http://localhost:8002 \
  --weather-endpoint http://localhost:8004 \
  --battery-endpoint http://localhost:8005 \
  --load-endpoint http://localhost:8006 \
  --energy-price-endpoint http://localhost:8007 \
  --mongodb-uri "mongodb://localhost:27017/" \
  --db-name solar_energy \
  --collection agent_data
```

---

## � Dashboard

El dashboard muestra en tiempo real:
- 🔆 **Generación Solar**: Predicciones LSTM, escenarios (Normal/Degradado/Fallo), desviaciones
- 🌤️ **Condiciones Climáticas**: Temperatura, irradiancia, viento
- 🔋 **Estado de Baterías**: SoC, voltaje, corriente, carga/descarga
- ⚡ **Consumo de Cargas**: Demanda actual, perfil de consumo
- 💰 **Precios de Energía**: Precio actual del mercado, estadísticas 24h, condiciones del mercado, recomendaciones

Características:
- Visualización con códigos de color (🟢 Normal, 🟡 Advertencia, 🔴 Crítico)
- Integración con Ollama para consultas en lenguaje natural
- Actualización automática desde MongoDB
- Métricas históricas y tendencias

```bash
python src/agents/dashboard.py --mongodb-uri "mongodb://localhost:27017/"
```

---

## �📚 Documentación

- **[Arquitectura del Sistema](docs/ARCHITECTURE.md)** - Visión completa del sistema
- **[Battery Agent](docs/BATTERY_AGENT.md)** - Almacenamiento de baterías
- **[Load Agent](docs/LOAD_AGENT.md)** - Simulador de consumo
- **[Orchestrator MCP](docs/ORCHESTRATOR_MCP.md)** - Coordinación con MongoDB
- **[Weather Mock](docs/MOCK_WEATHER_SERVER.md)** - Servidor meteorológico simulado

---

## 🔧 Agentes Disponibles

### ☀️ Generator Agent (Puerto 8002)
- Predicción LSTM de generación solar
- Simulación PV Supervisor
- Sistema de semáforos

### 🔋 Battery Agent (Puerto 8005)
- Control de carga/descarga
- Monitoreo SOC, voltaje, corriente
- Gestión de ciclos de vida

### ⚡ Load Agent (Puerto 8006)
- Simulación de consumo residencial/comercial/industrial
- Control de cargas individuales
- Pronósticos de consumo

### 🌤️ Weather Agent (Puerto 8004)
- Datos NASA POWER API
- Servidor mock para desarrollo
- Irradiancia, temperatura, viento

### 💰 Energy Price Predictor Agent (Puerto 8007)
- Predicción de precios de electricidad en tiempo real
- Análisis del mercado eléctrico mexicano (CENACE)
- Recomendaciones inteligentes de consumo
- Pronósticos de precios futuros

### 🎯 Orchestrator
- Coordina todos los agentes (Generator, Weather, Battery, Load, Energy Price Predictor)
- Sistema experto de toma de decisiones basado en precios de energía
- Guarda datos en MongoDB con soporte para pymongo directo
- Backup en JSON local para monitoreo
- Análisis de condiciones del mercado eléctrico en tiempo real

---

## 🗄️ MongoDB MCP Server

### ¿Qué es MCP?
**MCP (Model Context Protocol)** permite que el orchestrator escriba datos directamente a MongoDB sin necesidad de drivers nativos de Python.

### Ventajas
- ✅ Persistencia en la nube (MongoDB Atlas)
- ✅ Consultas avanzadas y agregaciones
- ✅ Escalabilidad y backups automáticos
- ✅ Acceso desde múltiples aplicaciones

### Importante: Mapeo de Puertos
El servidor MCP **debe** iniciarse con `-p 3000:3000` para ser accesible desde el host:

```bash
# ❌ INCORRECTO - Sin mapeo de puertos
docker run --rm -i \
  -e MDB_MCP_CONNECTION_STRING="..." \
  mongodb/mongodb-mcp-server:latest

# ✅ CORRECTO - Con mapeo de puertos
docker run --rm -i \
  -e MDB_MCP_CONNECTION_STRING="..." \
  -p 3000:3000 \
  mongodb/mongodb-mcp-server:latest
```

### Scripts de Gestión

| Script | Descripción |
|--------|-------------|
| `./start_mcp_server.sh` | Inicia servidor MCP correctamente |
| `./check_mcp_server.sh` | Verifica configuración y conectividad |
| `./launch_orchestrator.sh` | Lanza orchestrator con MCP |

### Verificar Servidor MCP

```bash
# Verificar estado completo
./check_mcp_server.sh

# Probar manualmente
curl http://localhost:3000/.well-known/mcp.json
curl http://localhost:3000/tools
```

---

## 📋 Registro Dinámico de Agentes (JADE-like)

El sistema implementa un **registro dinámico de agentes** similar al Directory Facilitator (DF) de JADE:

### Características

- **Auto-registro**: Los agentes se registran automáticamente al iniciar
- **Descubrimiento dinámico**: El orchestrator descubre agentes sin configuración previa
- **Heartbeat**: Monitoreo de salud de agentes
- **Persistencia**: Snapshots del registro en MongoDB (colección `agent_registry`)
- **Web Dashboard**: Visualización en tiempo real de la arquitectura del sistema

### Arquitectura

```
┌─────────────────────┐
│   Orchestrator      │  Puerto 8001 - Registry Server
│   (Registry API)    │  
└──────────┬──────────┘
           │
           │ REST API (/register, /deregister, /agents, /heartbeat)
           │
    ┌──────┴───────┬───────────┬──────────┬───────────┐
    │              │           │          │           │
┌───▼───┐    ┌────▼────┐  ┌───▼───┐  ┌──▼────┐  ┌───▼────┐
│Solar  │    │Weather  │  │Battery│  │Load   │  │Price   │
│Agent  │    │Agent    │  │Agent  │  │Agent  │  │Pred    │
│:8002  │    │:8004    │  │:8005  │  │:8006  │  │:8007   │
└───────┘    └─────────┘  └───────┘  └───────┘  └────────┘
```

### Dashboard Web 🌐

Accede a `http://localhost:5000` para ver:

- **Diagrama de Arquitectura**: Visualización interactiva de todos los agentes
- **Estadísticas**: Total de agentes, puertos, última actualización
- **Detalles de Agentes**: Endpoints, skills, IDs únicos
- **Auto-refresh**: Actualización cada 10 segundos

```bash
# Lanzar dashboard web
./launch_dashboard.sh

# Acceder a:
# - Terminal: Datos en tiempo real + chat con Ollama
# - Web: http://localhost:5000 para diagrama de agentes
```

### Endpoints de Registry API

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/register` | POST | Registrar un nuevo agente |
| `/deregister` | POST | Des-registrar un agente |
| `/agents` | GET | Listar todos los agentes registrados |
| `/heartbeat` | POST | Actualizar estado de un agente |

### Datos en MongoDB

El registro se guarda en dos colecciones:

1. **`agent_data`**: Datos de monitoreo de cada iteración
2. **`agent_registry`**: Snapshots del registro de agentes

```javascript
// Ejemplo de documento en agent_registry
{
  "timestamp": ISODate("2024-01-15T10:30:00Z"),
  "agents": [
    {
      "agent_id": "uuid-123...",
      "name": "Solar Generator Agent",
      "endpoint": "http://localhost:8002",
      "skills": ["solar_generation", "lstm_prediction"],
      "registered_at": ISODate("2024-01-15T10:00:00Z"),
      "last_heartbeat": ISODate("2024-01-15T10:29:55Z")
    }
  ],
  "total_agents": 5,
  "registry_port": "8001"
}
```

---

## LSTM PV Prediction

Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train (example):

```bash
python src/models/train_lstm_pv.py --data examples/LSTM3miso/completeDataF.csv --epochs 30
```

3. Predict next value:

```bash
python src/models/predict_lstm_pv.py --model models/lstm_pv.h5 --scaler models/scaler.pkl --data examples/LSTM3miso/completeDataF.csv
```

Notes

- The scripts follow the preprocessing from `examples/LSTM3miso/misopredict.ipynb` (default uses columns 2:8 and time_steps=24). Adjust arguments if your CSV differs.
- Model and scaler are saved under `models/` by default.



# agents

## Weather Agent con Mock Server

El Weather Agent puede funcionar con o sin conexión a internet:

```bash
# Modo normal (intenta NASA API, fallback a mock)
python src/agents/weather.py

# Modo simulación (siempre usa mock)
./launch_mock_weather.sh  # Terminal 1: Servidor mock
python src/agents/weather.py --mock  # Terminal 2: Agente
```

Ver [documentación completa del Mock Server](docs/MOCK_WEATHER_SERVER.md)

## Otros Agentes

```bash
# Agente de Generación Solar
python src/agents/generator.py

# Agente de Clima (NASA API o mock)
python src/agents/weather.py --mock 

# Agente de Baterías/Almacenamiento
python src/agents/battery.py --port 8005 --soc 50 --capacity 10

# Agente de Consumo Eléctrico
python src/agents/load.py --port 8006 --profile residential --base-load 1.5

# Agente Predictor de Precios de Energía (CENACE)
python src/agents/energy_price_predictor.py --port 8007 --sistema SIN --mercado MDA --nodo 06MTY-115

# Dashboard (Monitoreo en tiempo real con MongoDB)
python src/agents/dashboard.py \
  --mongodb-uri "mongodb://localhost:27017/" \
  --db-name solar_energy \
  --collection agent_data \
  --refresh 30 \
  --ollama-model deepseek-r1:1.5b

# Servidor Mock de Clima
python tools/mock_weather_server.py 

# Orquestador
python src/agents/orchestrator.py 
``` 



source venv/bin/activate

pip install flask

ssh fear@IP





DataFrame → Selecciona columnas → Normaliza → Secuencia[24] → LSTM → Desnormaliza → Predicción



python src/agents/dashboard.py \
  --mongodb-uri "mongodb://localhost:27017/" \
  --db-name solar_energy \
  --collection agent_data \
  --refresh 30 \
  --web-port 5000 \
  --orchestrator-url http://localhost:8001 \
  --ollama-model deepseek-r1:1.5b


  python src/agents/dashboard.py --mongodb-uri "mongodb://localhost:27017/" --db-name solar_energy --collection agent_data --refresh 30 --web-port 5000 --orchestrator-url http://localhost:8001 --ollama-model deepseek-r1:1.5b

sudo docker start mongodb && echo "✅ MongoDB iniciado correctamente"



sudo systemctl stop ollama
OLLAMA_NUM_GPU=0 ollama serve &

