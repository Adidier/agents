# Tools / Herramientas

Esta carpeta contiene herramientas auxiliares y de utilidad para el sistema de agentes.

## Contenido

### mock_weather_server.py

Servidor mock que simula la API POWER de la NASA para datos meteorológicos.

**Uso:**
```bash
python tools/mock_weather_server.py --port 8005
```

**Propósito:**
- Permite operación sin conexión a internet
- Genera datos meteorológicos sintéticos pero realistas
- Sirve como fallback automático para el Weather Agent
- Útil para desarrollo y testing

**Documentación completa:** [MOCK_WEATHER_SERVER.md](../docs/MOCK_WEATHER_SERVER.md)

## Scripts de Lanzamiento

En la raíz del proyecto hay scripts para facilitar el uso de estas herramientas:

- `./launch_mock_weather.sh` - Inicia el servidor mock
- `./test_mock_weather.sh` - Ejecuta pruebas del sistema mock
