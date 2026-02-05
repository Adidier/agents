"""
Mock Weather Server - Simulador de NASA POWER API

Servidor mock que simula respuestas de la API POWER de la NASA
cuando no hay conexión a internet. Genera datos meteorológicos
realistas basados en patrones estacionales y ubicación geográfica.
"""

import random
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List
from flask import Flask, request, jsonify

app = Flask(__name__)


class WeatherSimulator:
    """Genera datos meteorológicos simulados realistas."""
    
    def __init__(self):
        self.seed = random.randint(0, 1000000)
    
    def _get_seasonal_factor(self, date: datetime, latitude: float) -> float:
        """
        Calcula el factor estacional basado en la fecha y latitud.
        Retorna valor entre 0.5 y 1.5
        """
        day_of_year = date.timetuple().tm_yday
        # Fase estacional (más radiación en verano del hemisferio correspondiente)
        phase = 2 * math.pi * day_of_year / 365.25
        
        # Ajustar por hemisferio
        if latitude < 0:
            phase += math.pi  # Desplazar 6 meses para hemisferio sur
        
        return 1.0 + 0.4 * math.sin(phase)
    
    def _add_noise(self, value: float, noise_factor: float = 0.1) -> float:
        """Agrega ruido aleatorio al valor."""
        noise = random.uniform(-noise_factor, noise_factor)
        return value * (1 + noise)
    
    def generate_solar_irradiance(
        self, 
        latitude: float, 
        date: datetime,
        clear_sky: bool = False
    ) -> float:
        """
        Genera irradiancia solar simulada (kWh/m²/día).
        
        Args:
            latitude: Latitud de la ubicación
            date: Fecha para la simulación
            clear_sky: Si True, genera valores para cielo despejado
        """
        # Base según latitud (mayor irradiancia cerca del ecuador)
        lat_factor = 1.0 - abs(latitude) / 180.0 * 0.3
        
        # Factor estacional
        seasonal_factor = self._get_seasonal_factor(date, latitude)
        
        # Valor base: 4-6 kWh/m²/día
        base_value = 5.0 * lat_factor * seasonal_factor
        
        if clear_sky:
            # Cielo despejado: 10-20% más radiación
            value = base_value * 1.15
        else:
            # Condiciones reales con variabilidad por nubes
            cloud_factor = random.uniform(0.7, 1.0)
            value = base_value * cloud_factor
        
        return max(0.5, self._add_noise(value, 0.15))
    
    def generate_temperature(
        self, 
        latitude: float, 
        date: datetime,
        time_of_day: str = "avg"
    ) -> float:
        """
        Genera temperatura simulada (°C).
        
        Args:
            latitude: Latitud de la ubicación
            date: Fecha para la simulación
            time_of_day: "avg", "max", o "min"
        """
        # Base según latitud
        base_temp = 25.0 - abs(latitude) / 3.0
        
        # Factor estacional
        seasonal_factor = self._get_seasonal_factor(date, latitude)
        temp = base_temp + (seasonal_factor - 1.0) * 10.0
        
        # Ajustar según hora del día
        if time_of_day == "max":
            temp += random.uniform(5, 12)
        elif time_of_day == "min":
            temp -= random.uniform(5, 12)
        
        return self._add_noise(temp, 0.1)
    
    def generate_humidity(self, latitude: float, date: datetime) -> float:
        """Genera humedad relativa simulada (%)."""
        # Mayor humedad cerca del ecuador
        base_humidity = 60.0 + (30.0 - abs(latitude)) / 90.0 * 20.0
        
        # Variación estacional
        seasonal = self._get_seasonal_factor(date, latitude)
        humidity = base_humidity * (0.9 + seasonal * 0.1)
        
        return max(20, min(95, self._add_noise(humidity, 0.15)))
    
    def generate_wind_speed(self) -> float:
        """Genera velocidad del viento simulada (m/s)."""
        # Distribución Weibull típica para vientos
        base = random.weibullvariate(3.0, 2.0)
        return max(0, min(15, base))
    
    def generate_precipitation(self, date: datetime, latitude: float) -> float:
        """Genera precipitación simulada (mm/día)."""
        # Factor estacional
        seasonal = self._get_seasonal_factor(date, latitude)
        
        # 70% de días sin lluvia
        if random.random() < 0.7:
            return 0.0
        
        # Días con lluvia: 0.5 a 30 mm
        base = random.expovariate(1/8.0)
        return min(30, base * seasonal)
    
    def generate_pressure(self, latitude: float) -> float:
        """Genera presión atmosférica simulada (kPa)."""
        # Presión base al nivel del mar
        base_pressure = 101.325
        
        # Variación por latitud y condiciones meteorológicas
        variation = random.uniform(-3, 3)
        
        return base_pressure + variation
    
    def generate_weather_data(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        parameters: List[str]
    ) -> Dict[str, Any]:
        """
        Genera conjunto completo de datos meteorológicos simulados.
        
        Args:
            latitude: Latitud (-90 a 90)
            longitude: Longitud (-180 a 180)
            start_date: Fecha inicio (YYYYMMDD)
            end_date: Fecha fin (YYYYMMDD)
            parameters: Lista de parámetros a generar
        """
        # Parsear fechas
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        
        # Generar datos para cada día
        param_data = {}
        
        current = start
        while current <= end:
            date_key = current.strftime("%Y%m%d")
            
            for param in parameters:
                if param not in param_data:
                    param_data[param] = {}
                
                # Generar valor según el parámetro
                if param == "ALLSKY_SFC_SW_DWN":
                    value = self.generate_solar_irradiance(latitude, current, False)
                elif param == "CLRSKY_SFC_SW_DWN":
                    value = self.generate_solar_irradiance(latitude, current, True)
                elif param == "T2M":
                    value = self.generate_temperature(latitude, current, "avg")
                elif param == "T2M_MAX":
                    value = self.generate_temperature(latitude, current, "max")
                elif param == "T2M_MIN":
                    value = self.generate_temperature(latitude, current, "min")
                elif param == "RH2M":
                    value = self.generate_humidity(latitude, current)
                elif param == "PRECTOTCORR":
                    value = self.generate_precipitation(current, latitude)
                elif param == "WS2M":
                    value = self.generate_wind_speed()
                elif param == "PS":
                    value = self.generate_pressure(latitude)
                else:
                    value = 0.0
                
                param_data[param][date_key] = round(value, 2)
            
            current += timedelta(days=1)
        
        # Construir respuesta en formato NASA POWER
        response = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [longitude, latitude, 0]
            },
            "properties": {
                "parameter": param_data
            },
            "header": {
                "title": "NASA/POWER CERES/MERRA2 Native Resolution Daily Data (MOCK)",
                "api": {
                    "version": "v2.5.0 (MOCK)",
                    "name": "POWER Mock API"
                }
            }
        }
        
        return response


# Instancia global del simulador
simulator = WeatherSimulator()


@app.route('/api/temporal/daily/point', methods=['GET'])
def get_weather_data():
    """Endpoint que simula la API de NASA POWER."""
    try:
        # Obtener parámetros de la solicitud
        latitude = float(request.args.get('latitude', 20.1011))
        longitude = float(request.args.get('longitude', -98.7625))
        start_date = request.args.get('start')
        end_date = request.args.get('end')
        parameters = request.args.get('parameters', '').split(',')
        
        # Validar parámetros
        if not start_date or not end_date:
            return jsonify({"error": "start and end dates are required"}), 400
        
        if not parameters or parameters == ['']:
            return jsonify({"error": "parameters are required"}), 400
        
        # Generar datos simulados
        data = simulator.generate_weather_data(
            latitude, longitude, start_date, end_date, parameters
        )
        
        return jsonify(data)
    
    except ValueError as e:
        return jsonify({"error": f"Invalid parameter: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "Mock Weather Server",
        "message": "Simulating NASA POWER API"
    })


@app.route('/', methods=['GET'])
def index():
    """Información del servidor."""
    return jsonify({
        "name": "Mock Weather Server",
        "description": "Servidor mock que simula la API POWER de la NASA",
        "version": "1.0.0",
        "endpoints": {
            "/api/temporal/daily/point": "Obtener datos meteorológicos simulados",
            "/health": "Health check"
        },
        "parameters": [
            "ALLSKY_SFC_SW_DWN - Irradiancia solar (kWh/m²/day)",
            "CLRSKY_SFC_SW_DWN - Irradiancia cielo despejado",
            "T2M - Temperatura a 2m (°C)",
            "T2M_MAX - Temperatura máxima",
            "T2M_MIN - Temperatura mínima",
            "RH2M - Humedad relativa (%)",
            "PRECTOTCORR - Precipitación (mm/day)",
            "WS2M - Velocidad del viento (m/s)",
            "PS - Presión atmosférica (kPa)"
        ]
    })


def run_mock_server(host: str = "0.0.0.0", port: int = 8005):
    """Inicia el servidor mock."""
    print("=" * 60)
    print("🔧 Mock Weather Server - Simulador NASA POWER API")
    print("=" * 60)
    print(f"🌐 Servidor iniciado en http://{host}:{port}")
    print(f"📡 Endpoint: http://{host}:{port}/api/temporal/daily/point")
    print(f"💚 Health check: http://{host}:{port}/health")
    print("=" * 60)
    print("⚠️  MODO SIMULACIÓN: Generando datos meteorológicos sintéticos")
    print("=" * 60)
    
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Mock Weather Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host para el servidor")
    parser.add_argument("--port", type=int, default=8005, help="Puerto del servidor")
    
    args = parser.parse_args()
    
    run_mock_server(host=args.host, port=args.port)
