"""
A2A PV Power Simulator

Simula valores reales de potencia fotovoltaica para pruebas del sistema de supervisión.
Genera valores con diferentes escenarios: normal, degradado, falla.
"""

import random
from typing import Literal


class PVSimulator:
    """
    Simulador de potencia fotovoltaica real.
    Útil para pruebas del sistema de supervisión sin necesidad de hardware real.
    """
    
    def __init__(self, seed: int = None):
        """
        Inicializa el simulador.
        
        Args:
            seed: Semilla para reproducibilidad de valores aleatorios
        """
        if seed is not None:
            random.seed(seed)
    
    def simulate_real_power(
        self,
        predicted_power: float,
        scenario: Literal["normal", "degraded", "fault"] = "normal"
    ) -> float:
        """
        Simula el valor real de potencia basado en la predicción.
        
        Args:
            predicted_power: Valor predicho de potencia (kW)
            scenario: Escenario de simulación:
                - "normal": Desviación < 15% (luz verde)
                - "degraded": Desviación 15-30% (luz amarilla)
                - "fault": Desviación > 30% (luz roja)
        
        Returns:
            Valor simulado de potencia real (kW)
        """
        if scenario == "normal":
            # Desviación entre -10% y +10%
            deviation = random.uniform(-0.10, 0.10)
        elif scenario == "degraded":
            # Desviación entre -25% y -15% o +15% y +25%
            if random.random() < 0.5:
                deviation = random.uniform(-0.25, -0.15)
            else:
                deviation = random.uniform(0.15, 0.25)
        elif scenario == "fault":
            # Desviación entre -50% y -30% o +30% y +50%
            if random.random() < 0.5:
                deviation = random.uniform(-0.50, -0.30)
            else:
                deviation = random.uniform(0.30, 0.50)
        else:
            deviation = 0.0
        
        real_power = predicted_power * (1 + deviation)
        
        # Asegurar que no sea negativo
        return max(0.0, real_power)
    
    def simulate_hourly_pattern(
        self,
        hour: int,
        max_power: float = 5.0,
        scenario: Literal["normal", "cloudy", "rainy"] = "normal"
    ) -> float:
        """
        Simula un patrón horario típico de generación fotovoltaica.
        
        Args:
            hour: Hora del día (0-23)
            max_power: Potencia máxima al mediodía (kW)
            scenario: Condiciones climáticas
        
        Returns:
            Potencia simulada para esa hora (kW)
        """
        # Patrón base (curva gaussiana centrada al mediodía)
        import math
        base_power = max_power * math.exp(-((hour - 12) ** 2) / (2 * 3 ** 2))
        
        # Ajustar según escenario
        if scenario == "cloudy":
            # Reducción del 40-60%
            factor = random.uniform(0.4, 0.6)
            base_power *= factor
        elif scenario == "rainy":
            # Reducción del 70-90%
            factor = random.uniform(0.1, 0.3)
            base_power *= factor
        
        # Agregar ruido aleatorio pequeño
        noise = random.uniform(-0.1, 0.1)
        return max(0.0, base_power * (1 + noise))
