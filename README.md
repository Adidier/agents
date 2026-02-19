# Agentes de Energía Solar

Sistema multi-agente para gestión inteligente de energía solar con almacenamiento.

## 🚀 Inicio Rápido

### 1. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 2. Iniciar Servidor MCP de MongoDB (Opcional)
```bash
# Configurar variable de entorno con tu conexión MongoDB
export MDB_MCP_CONNECTION_STRING="mongodb+srv://usuario:password@cluster.mongodb.net/database"

# Iniciar servidor MCP con mapeo de puertos
./start_mcp_server.sh

# Verificar que está corriendo
./check_mcp_server.sh
```

### 3. Lanzar Agentes
```bash
# Opción A: Sistema completo
./launch_system.sh

# Opción B: Agentes individuales
./launch_battery.sh   # Puerto 8005
./launch_load.sh      # Puerto 8006
python src/agents/generator.py  # Puerto 8002
python src/agents/weather.py    # Puerto 8004
```

### 4. Iniciar Orchestrator
```bash
# Con MongoDB MCP (recomendado)
./launch_orchestrator.sh

# Solo con JSON local
python src/agents/orchestrator.py \
  --solar-endpoint http://localhost:8002 \
  --battery-endpoint http://localhost:8005 \
  --load-endpoint http://localhost:8006 \
  --weather-endpoint http://localhost:8004
```

---

## 📚 Documentación

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

### 🎯 Orchestrator
- Coordina todos los agentes
- Guarda datos en MongoDB (vía MCP)
- Backup en JSON local

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

# Dashboard
python src/agents/dashboard.py 

# Servidor Mock de Clima
python tools/mock_weather_server.py 

# Orquestador
python src/agents/orchestrator.py 
``` 



source venv/bin/activate

pip install flask

ssh fear@IP





DataFrame → Selecciona columnas → Normaliza → Secuencia[24] → LSTM → Desnormaliza → Predicción