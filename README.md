


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


python src/agents/orchestrator.py 

python3 -c "
from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017/')
db = client['solar_energy']
print(f'Eliminados {db.agent_data.delete_many({}).deleted_count} docs de agent_data')
print(f'Eliminados {db.agent_registry.delete_many({}).deleted_count} docs de agent_registry')
client.close()
"
