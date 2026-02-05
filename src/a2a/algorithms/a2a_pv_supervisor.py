"""
A2A PV Supervisor

Sistema de supervisión automática para sistemas fotovoltaicos basado en:
"An Automatic Supervision Method for Photovoltaic-Arrays Systems"
IAENG International Journal of Computer Science, Vol 53, Issue 1, January 2026

Implementa:
- Clasificación por integral discreta
- Sistema de semáforo (Verde/Amarillo/Rojo)
- Métricas MAE y RMSE
- Comparación de valores reales vs predichos
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum


class TrafficLightStatus(Enum):
    """Estado del sistema fotovoltaico basado en desviación de predicción."""
    GREEN = "green"      # Desviación < 15% - Sistema operando correctamente
    YELLOW = "yellow"    # Desviación 15-30% - Advertencia
    RED = "red"          # Desviación > 30% - Falla crítica


class PVSupervisor:
    """
    Supervisor automático para sistemas fotovoltaicos.
    Compara valores reales con predicciones usando integral discreta.
    """
    
    def __init__(
        self,
        green_threshold: float = 15.0,
        yellow_threshold: float = 30.0,
        history_size: int = 100
    ):
        """
        Inicializa el supervisor.
        
        Args:
            green_threshold: Umbral para luz verde (% desviación)
            yellow_threshold: Umbral para luz amarilla (% desviación)
            history_size: Tamaño del historial de predicciones
        """
        self.green_threshold = green_threshold
        self.yellow_threshold = yellow_threshold
        self.history_size = history_size
        
        # Historial de predicciones y valores reales
        self.prediction_history: List[float] = []
        self.real_history: List[float] = []
        self.timestamps: List[str] = []
        self.status_history: List[TrafficLightStatus] = []
        
    def discrete_integral(self, values: List[float]) -> float:
        """
        Calcula la integral discreta de una serie de valores.
        Usa el método del trapecio para aproximar la integral.
        
        Args:
            values: Lista de valores a integrar
            
        Returns:
            Valor de la integral discreta
        """
        if len(values) < 2:
            return sum(values)
        
        return np.trapz(values)
    
    def calculate_metrics(
        self,
        real_values: List[float],
        predicted_values: List[float]
    ) -> Dict[str, float]:
        """
        Calcula métricas de evaluación MAE y RMSE.
        
        Args:
            real_values: Valores reales
            predicted_values: Valores predichos
            
        Returns:
            Diccionario con métricas MAE y RMSE
        """
        real_array = np.array(real_values)
        pred_array = np.array(predicted_values)
        
        # MAE: Mean Absolute Error
        mae = np.mean(np.abs(real_array - pred_array))
        
        # RMSE: Root Mean Squared Error
        rmse = np.sqrt(np.mean((real_array - pred_array) ** 2))
        
        return {
            "MAE": float(mae),
            "RMSE": float(rmse)
        }
    
    def classify_performance(
        self,
        predicted_power: float,
        real_power: float,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Clasifica el rendimiento del sistema usando integral discreta.
        
        Basado en el algoritmo del paper:
        FpredI = integrate(Fpred)
        FrealI = integrate(Freal)
        outputPerCent = FrealI/FpredI * 100
        
        Args:
            predicted_power: Potencia predicha (kW)
            real_power: Potencia real medida (kW)
            timestamp: Timestamp opcional de la medición
            
        Returns:
            Diccionario con clasificación y estado del semáforo
        """
        # Agregar al historial
        self.prediction_history.append(predicted_power)
        self.real_history.append(real_power)
        if timestamp:
            self.timestamps.append(timestamp)
        
        # Mantener solo el historial reciente
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
            self.real_history.pop(0)
            if self.timestamps:
                self.timestamps.pop(0)
        
        # Calcular integrales discretas
        fpred_integral = self.discrete_integral(self.prediction_history)
        freal_integral = self.discrete_integral(self.real_history)
        
        # Calcular porcentaje de salida
        if fpred_integral > 0:
            output_percent = (freal_integral / fpred_integral) * 100
        else:
            output_percent = 0.0
        
        # Calcular desviación promedio (integral)
        deviation_integral = abs(100 - output_percent)
        
        # Calcular también desviación instantánea del valor actual
        if predicted_power > 0:
            instant_deviation = abs((real_power - predicted_power) / predicted_power) * 100
        else:
            instant_deviation = 0.0
        
        # Usar la desviación MAYOR entre integral e instantánea
        # Esto hace que el sistema responda rápido a problemas inmediatos
        deviation = max(deviation_integral, instant_deviation)
        
        # Clasificar estado del semáforo
        if deviation < self.green_threshold:
            light_status = TrafficLightStatus.GREEN
            message = "Sistema operando correctamente"
        elif deviation < self.yellow_threshold:
            light_status = TrafficLightStatus.YELLOW
            message = "Advertencia: Desviación moderada detectada"
        else:
            light_status = TrafficLightStatus.RED
            message = "ALERTA: Falla crítica detectada"
        
        self.status_history.append(light_status)
        if len(self.status_history) > self.history_size:
            self.status_history.pop(0)
        
        # Calcular métricas si hay suficiente historial
        metrics = {}
        if len(self.prediction_history) >= 2:
            metrics = self.calculate_metrics(
                self.real_history,
                self.prediction_history
            )
        
        result = {
            "predicted_power": predicted_power,
            "real_power": real_power,
            "predicted_integral": fpred_integral,
            "real_integral": freal_integral,
            "output_percent": output_percent,
            "deviation_percent": deviation,
            "deviation_integral": deviation_integral,
            "deviation_instant": instant_deviation,
            "light_status": light_status.value,
            "message": message,
            "metrics": metrics,
            "timestamp": timestamp
        }
        
        return result
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del sistema.
        
        Returns:
            Diccionario con el estado actual del sistema
        """
        if not self.status_history:
            return {
                "light_status": "unknown",
                "message": "No hay datos de supervisión disponibles",
                "history_size": 0
            }
        
        current_light = self.status_history[-1]
        
        # Contar ocurrencias de cada estado en historial reciente
        recent_history = self.status_history[-10:]  # Últimos 10
        green_count = sum(1 for s in recent_history if s == TrafficLightStatus.GREEN)
        yellow_count = sum(1 for s in recent_history if s == TrafficLightStatus.YELLOW)
        red_count = sum(1 for s in recent_history if s == TrafficLightStatus.RED)
        
        return {
            "light_status": current_light.value,
            "history_size": len(self.status_history),
            "recent_stats": {
                "green": green_count,
                "yellow": yellow_count,
                "red": red_count,
                "total": len(recent_history)
            }
        }
    
    def reset_history(self):
        """Reinicia el historial de supervisión."""
        self.prediction_history.clear()
        self.real_history.clear()
        self.timestamps.clear()
        self.status_history.clear()
