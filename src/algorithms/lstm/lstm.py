"""
A2A LSTM Agent

Provides LSTM-based predictions for PV (photovoltaic) output.
Uses a pre-trained LSTM model for time-series forecasting.
Includes automatic supervision system with traffic light classification.
"""

from typing import Dict, Any, List, Iterator, Optional
import os
import time
import pandas as pd
from a2a.core.a2a_ia_algorithm_interface import IA2AIAAlgorithm
from a2a.core.agent_card import AgentCard
from a2a.core.task_manager import TaskManager
from a2a.core.message_handler import MessageHandler
from algorithms.lstm.lstm_model import A2ALSTMModel
from algorithms.lstm.pv_supervisor import PVSupervisor
from algorithms.lstm.pv_simulator import PVSimulator


class A2ALSTM(IA2AIAAlgorithm):
    """
    LSTM-based PV prediction agent.
    Prioritizes LSTM model predictions; returns error message if model not loaded.
    """
    
    def __init__(
        self,
        model: str,
        name: str,
        description: str,
        skills: List[Dict[str, Any]],
        host: str = "http://localhost:11434",
        endpoint: str = "http://localhost:8000",
        auto_simulate: bool = True,
    ):
        self.model = model
        self.agent_card = AgentCard(
            name=name,
            description=description,
            endpoint=endpoint,
            skills=skills,
        )
        self.task_manager = TaskManager()
        self.message_handler = MessageHandler()
        self.mcp_client = None
        # Optional LSTM helper (integrated from a2a_lstm_model)
        self.lstm_helper: A2ALSTMModel | None = None
        # PV Supervisor for automatic classification
        self.supervisor = PVSupervisor(
            green_threshold=15.0,
            yellow_threshold=30.0,
            history_size=100
        )
        # Store last prediction for comparison
        self.last_prediction: Optional[float] = None
        
        # Auto-simulation for testing
        self.auto_simulate = auto_simulate
        self.simulator = PVSimulator(seed=None) if auto_simulate else None
        # Escenarios rotativos para simulación
        self.scenario_cycle = [
            "normal",      # 🟢 Verde
            "normal",      # 🟢 Verde
            "degraded",    # 🟡 Amarillo
            "normal",      # 🟢 Verde
            "fault",       # 🔴 Rojo
            "degraded",    # 🟡 Amarillo
            "fault",       # 🔴 Rojo
            "normal",      # 🟢 Verde
        ]
        self.scenario_index = 0

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            message_content = request.get("message", {}).get("content", "")
            # Try LSTM prediction first if helper is loaded
            if self.lstm_helper:
                try:
                    pred = self._try_predict_lstm()
                    if pred is not None:
                        response_content = f"LSTM Prediction (PV Output): {pred:.2f} kW"
                        return {
                            "message": {"parts": [{"type": "text", "content": response_content}]}
                        }
                except Exception:
                    pass
            # Fallback to generic response
            response_content = self._get_fallback_response(message_content)
            return {
                "message": {"parts": [{"type": "text", "content": response_content}]}
            }
        except Exception as e:
            return {"message": {"parts": [{"type": "text", "content": f"Error processing request: {str(e)}"}]}}

    def chat(self, prompt: str) -> Dict[str, Any]:
        """
        Chat interface with LSTM prediction and supervision status.
        Checks for real power data in prompt for classification.
        """
        # Check if the prompt contains real power data
        if prompt.startswith("REAL_POWER:"):
            try:
                real_power_str = prompt.replace("REAL_POWER:", "").strip()
                real_power = float(real_power_str)
                
                # Submit real power for classification
                result = self.submit_real_power(real_power)
                
                # Format response with classification
                if "error" not in result:
                    light = result['light_status']
                    emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(light, "⚪")
                    
                    response_parts = [
                        f"📊 Predicción: {result['predicted_power']:.2f} kW",
                        f"\n📈 Real: {result['real_power']:.2f} kW",
                        f"\n{emoji} {result['message']}",
                        f"\n   Desviación: {result['deviation_percent']:.2f}%"
                    ]
                    
                    if result.get('metrics'):
                        metrics = result['metrics']
                        response_parts.append(f"\n   MAE: {metrics.get('MAE', 0):.4f} | RMSE: {metrics.get('RMSE', 0):.4f}")
                    
                    return {"message": {"parts": [{"type": "text", "content": "".join(response_parts)}]}}
                else:
                    return {"message": {"parts": [{"type": "text", "content": result['message']}]}}
                    
            except (ValueError, KeyError) as e:
                return {"message": {"parts": [{"type": "text", "content": f"Error processing real power: {e}"}]}}
        
        # Try to get LSTM prediction
        prediction = None
        if self.lstm_helper:
            try:
                prediction = self._try_predict_lstm()
                self.last_prediction = prediction
            except Exception:
                pass
        
        # Build response with prediction and supervision status
        supervisor_status = self.supervisor.get_current_status()
        
        if prediction is not None:
            response_parts = [
                f"🔮 LSTM Prediction: {prediction:.2f} kW",
                f"\n📊 System Status: {supervisor_status['light_status'].upper()}",
            ]
            
            # Add traffic light emoji
            if supervisor_status['light_status'] == 'green':
                response_parts.append("🟢 Sistema operando correctamente")
            elif supervisor_status['light_status'] == 'yellow':
                response_parts.append("🟡 Advertencia: Desviación moderada")
            elif supervisor_status['light_status'] == 'red':
                response_parts.append("🔴 ALERTA: Falla crítica detectada")
            
            if supervisor_status.get('history_size', 0) > 0:
                stats = supervisor_status.get('recent_stats', {})
                response_parts.append(
                    f"\n📈 Últimas {stats.get('total', 0)} mediciones: "
                    f"Verde={stats.get('green', 0)} | "
                    f"Amarillo={stats.get('yellow', 0)} | "
                    f"Rojo={stats.get('red', 0)}"
                )
            
            response_content = "".join(response_parts)
        else:
            response_content = "LSTM model not loaded. Please configure and load the LSTM model to enable PV predictions."
        
        return {"message": {"parts": [{"type": "text", "content": response_content}]}}
    
    def submit_real_power(self, real_power: float, timestamp: Optional[str] = None) -> Dict[str, Any]:
        """
        Submit real power measurement for supervision classification.
        Compares with last prediction to classify system performance.
        
        Args:
            real_power: Real power measured from inverter (kW)
            timestamp: Optional timestamp of measurement
            
        Returns:
            Classification result with traffic light status
        """
        if self.last_prediction is None:
            return {
                "error": "No prediction available for comparison",
                "message": "Please make a prediction first"
            }
        
        if timestamp is None:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Classify performance using supervisor
        result = self.supervisor.classify_performance(
            predicted_power=self.last_prediction,
            real_power=real_power,
            timestamp=timestamp
        )
        
        return result

    def _try_predict_lstm(self) -> Any:
        """Try to predict using LSTM with default CSV path. Returns None if fails."""
        if not self.lstm_helper:
            return None
        # Try multiple possible CSV paths
        csv_paths = [
            "examples/LSTM3miso/completeDataF.csv",
            "models/completeDataF.csv",
            "../examples/LSTM3miso/completeDataF.csv",
            "../models/completeDataF.csv",
        ]
        df = None
        for csv_path in csv_paths:
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    break
                except Exception:
                    continue
        
        if df is None:
            return None
        
        try:
            prediction = self.lstm_helper.predict_next(df)
            return prediction
        except Exception:
            return None

    def _get_fallback_response(self, prompt: str) -> str:
        """Return a generic fallback message when LSTM prediction is unavailable."""
        return "LSTM model not loaded. Please configure and load the LSTM model to enable PV predictions."

    # --- LSTM integration helpers ---
    def attach_lstm_helper(self, helper: A2ALSTMModel) -> None:
        """Attach an existing A2ALSTMModel instance for predictions."""
        self.lstm_helper = helper

    def train_lstm_from_csv(self, csv_path: str, **kwargs) -> None:
        """Train an LSTM model from CSV and attach helper when done."""
        helper = A2ALSTMModel()
        helper.fit_from_csv(csv_path, **kwargs)
        self.lstm_helper = helper

    def load_lstm(self, model_path: str, scaler_path: str) -> None:
        """Load an existing LSTM model and scaler and attach helper."""
        helper = A2ALSTMModel()
        helper.load(model_path, scaler_path)
        self.lstm_helper = helper

    def predict_lstm_next(self, df) -> Any:
        """Predict next value using the attached LSTM helper. Returns None if no helper."""
        if not self.lstm_helper:
            return None
        try:
            return self.lstm_helper.predict_next(df)
        except Exception:
            return None

    # --- Abstract method implementations (mocks) ---
    def _process_task(self, task_id: str) -> Dict[str, Any]:
        """Process task: attempt LSTM prediction and return result with supervision status."""
        try:
            # Get messages from the task to check for real power data
            messages = self.message_handler.get_messages(task_id)
            
            # Check if any message contains real power data
            for message in messages:
                if message.get("parts"):
                    for part in message["parts"]:
                        if part.get("type") == "text":
                            content = part.get("content", "")
                            if content.startswith("REAL_POWER:"):
                                # Process real power submission
                                try:
                                    real_power_str = content.replace("REAL_POWER:", "").strip()
                                    real_power = float(real_power_str)
                                    
                                    if self.last_prediction is None:
                                        return {
                                            "task_id": task_id,
                                            "status": "failed",
                                            "error": "No prediction available for comparison"
                                        }
                                    
                                    # Classify performance
                                    result = self.supervisor.classify_performance(
                                        predicted_power=self.last_prediction,
                                        real_power=real_power,
                                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                                    )
                                    
                                    # Return classification result
                                    return {
                                        "task_id": task_id,
                                        "status": "completed",
                                        "result": {
                                            "type": "pv_classification",
                                            "predicted_power": result['predicted_power'],
                                            "real_power": result['real_power'],
                                            "deviation_percent": result['deviation_percent'],
                                            "deviation_instant": result.get('deviation_instant', 0),
                                            "supervision": {
                                                "light_status": result['light_status'],
                                                "light_emoji": {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(result['light_status'], "⚪"),
                                                "message": result['message']
                                            },
                                            "metrics": result.get('metrics', {}),
                                            "formatted": f"{result['predicted_power']:.2f} kW (Real: {result['real_power']:.2f} kW)"
                                        }
                                    }
                                except (ValueError, KeyError) as e:
                                    return {
                                        "task_id": task_id,
                                        "status": "failed",
                                        "error": f"Error processing real power: {e}"
                                    }
            
            # No real power data, process as normal prediction
            if not self.lstm_helper:
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": "LSTM model not loaded"
                }
            pred = self._try_predict_lstm()
            if pred is not None:
                # Store prediction for future comparison
                self.last_prediction = pred
                
                # Auto-simulate real power if enabled
                if self.auto_simulate and self.simulator:
                    # Get scenario from cycle
                    scenario = self.scenario_cycle[self.scenario_index % len(self.scenario_cycle)]
                    self.scenario_index += 1
                    
                    # Simulate real power
                    real_power = self.simulator.simulate_real_power(pred, scenario=scenario)
                    
                    # Classify automatically
                    classification = self.supervisor.classify_performance(
                        predicted_power=pred,
                        real_power=real_power,
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                    )
                    
                    # Return classification result
                    return {
                        "task_id": task_id,
                        "status": "completed",
                        "result": {
                            "type": "pv_classification",
                            "predicted_power": classification['predicted_power'],
                            "real_power": classification['real_power'],
                            "deviation_percent": classification['deviation_percent'],
                            "deviation_instant": classification.get('deviation_instant', 0),
                            "scenario": scenario.upper(),
                            "supervision": {
                                "light_status": classification['light_status'],
                                "light_emoji": {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(classification['light_status'], "⚪"),
                                "message": classification['message']
                            },
                            "metrics": classification.get('metrics', {}),
                            "formatted": f"{classification['predicted_power']:.2f} kW (Real: {classification['real_power']:.2f} kW)"
                        }
                    }
                
                # No auto-simulation, just return prediction
                # Get supervisor status
                supervisor_status = self.supervisor.get_current_status()
                light = supervisor_status.get('light_status', 'unknown')
                
                # Traffic light emoji
                light_emoji = {
                    'green': '🟢',
                    'yellow': '🟡',
                    'red': '🔴',
                    'unknown': '⚪'
                }.get(light, '⚪')
                
                # Build result with supervision info
                result = {
                    "task_id": task_id,
                    "status": "completed",
                    "result": {
                        "type": "pv_prediction",
                        "value": float(pred),
                        "unit": "kW",
                        "formatted": f"{pred:.2f} kW",
                        "supervision": {
                            "light_status": light,
                            "light_emoji": light_emoji,
                            "history_size": supervisor_status.get('history_size', 0)
                        }
                    }
                }
                
                # Add recent stats if available
                if supervisor_status.get('recent_stats'):
                    result["result"]["supervision"]["recent_stats"] = supervisor_status['recent_stats']
                
                return result
            else:
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": "Could not generate LSTM prediction"
                }
        except Exception as e:
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(e)
            }

    def _process_task_stream(self, task_id: str) -> Iterator[Dict[str, Any]]:
        """Mock implementation of abstract method."""
        return iter([])

    def _get_ollama_messages(self, task_id: str) -> List[Dict[str, Any]]:
        """Mock implementation of abstract method."""
        return []

    def configure_mcp_client(self, mcp_client: Any) -> None:
        """Mock implementation of abstract method."""
        pass

    def discover_agent(self) -> Dict[str, Any]:
        """Mock implementation of abstract method."""
        return {}
