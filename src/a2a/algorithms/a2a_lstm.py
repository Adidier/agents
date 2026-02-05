"""
A2A LSTM Agent

Provides LSTM-based predictions for PV (photovoltaic) output.
Uses a pre-trained LSTM model for time-series forecasting.
"""

from typing import Dict, Any, List, Iterator
import os
import pandas as pd
from a2a.core.a2a_ia_algorithm_interface import IA2AIAAlgorithm
from a2a.core.agent_card import AgentCard
from a2a.core.task_manager import TaskManager
from a2a.core.message_handler import MessageHandler
from a2a.algorithms.a2a_lstm_model import A2ALSTMModel


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
        """Return generic response. LSTM prediction available via process_request()."""
        response_content = "Chat interface not yet configured. Use process_request() for LSTM predictions."
        return {"message": {"parts": [{"type": "text", "content": response_content}]}}

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
        """Process task: attempt LSTM prediction and return result."""
        try:
            if not self.lstm_helper:
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": "LSTM model not loaded"
                }
            pred = self._try_predict_lstm()
            if pred is not None:
                return {
                    "task_id": task_id,
                    "status": "completed",
                    "result": {
                        "type": "pv_prediction",
                        "value": float(pred),
                        "unit": "kW",
                        "formatted": f"{pred:.2f} kW"
                    }
                }
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
