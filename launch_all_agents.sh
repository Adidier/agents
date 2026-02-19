#!/bin/bash
# Launch all agents in the system with auto-registration

echo "🚀 Launching Multi-Agent Solar Energy System"
echo "=============================================="
echo ""
echo "📋 Agents will auto-register with orchestrator at: http://localhost:8001"
echo ""

# Kill existing processes on ports
echo "🧹 Cleaning up existing processes..."
for PORT in {8001..8007}; do
    if lsof -i:$PORT > /dev/null 2>&1; then
        echo "  Killing process on port $PORT..."
        fuser -k $PORT/tcp 2>/dev/null || true
        sleep 1
    fi
done

# Start orchestrator first (port 8001)
echo ""
echo "📋 Starting Orchestrator Registry (Port 8001)..."
python src/agents/orchestrator.py \
  --registry-port 8001 \
  --mongodb-uri "mongodb://localhost:27017/" \
  --db-name solar_energy \
  --collection agent_data > /tmp/orchestrator.log 2>&1 &
sleep 3

echo "🔆 Starting Generator Agent (Port 8002)..."
python src/agents/generator.py --port 8002 --orchestrator-url http://localhost:8001 > /tmp/generator.log 2>&1 &
sleep 2

echo "🌤️  Starting Weather Agent (Port 8004)..."
python src/agents/weather.py --port 8004 --orchestrator-url http://localhost:8001 > /tmp/weather.log 2>&1 &
sleep 2

echo "🔋 Starting Battery Agent (Port 8005)..."
python src/agents/battery.py --port 8005 --soc 50 --capacity 10 --orchestrator-url http://localhost:8001 > /tmp/battery.log 2>&1 &
sleep 2

echo "⚡ Starting Load Agent (Port 8006)..."
python src/agents/load.py --port 8006 --profile residential --base-load 1.5 --orchestrator-url http://localhost:8001 > /tmp/load.log 2>&1 &
sleep 2

echo "💰 Starting Energy Price Predictor Agent (Port 8007)..."
python src/agents/energy_price_predictor.py --port 8007 --orchestrator-url http://localhost:8001 > /tmp/energy_price.log 2>&1 &
sleep 3

echo ""
echo "✅ All agents started successfully!"
echo ""
echo "Agent Status:"
echo "  Orchestrator:    http://localhost:8001/agents (registry API)"
echo "  Generator:       http://localhost:8002/.well-known/agent.json"
echo "  Weather:         http://localhost:8004/.well-known/agent.json"
echo "  Battery:         http://localhost:8005/.well-known/agent.json"
echo "  Load:            http://localhost:8006/.well-known/agent.json"
echo "  Price Predictor: http://localhost:8007/.well-known/agent.json"
echo ""
echo "📊 View registered agents: curl http://localhost:8001/agents"
echo "📺 Monitor with: python src/agents/dashboard.py"
echo ""
echo "💡 Press Ctrl+C to stop all agents"

# Wait for all background jobs
wait
