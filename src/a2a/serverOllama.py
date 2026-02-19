
from flask import Flask, request, jsonify, Response, stream_with_context
from typing import Dict, Any, List, Optional, Callable

from a2a.server import A2AServer
from a2a.core.a2a_ia_algorithm_interface import IA2AIAAlgorithm

def run_server(
    port: int = 8000,
    endpoint: str = None,
    webhook_url: str = None,
    iaAlgorithm: IA2AIAAlgorithm = None,
    orchestrator_url: str = None
):
    """
    Run A2A server with automatic orchestrator registration.
    
    Args:
        port: Server port
        endpoint: Public endpoint URL
        webhook_url: Webhook URL for notifications
        iaAlgorithm: IA algorithm implementation
        orchestrator_url: Orchestrator registry URL for auto-registration (e.g., http://localhost:8001)
    """
    server = A2AServer(
        port=port,
        endpoint=endpoint,
        webhook_url=webhook_url,
        iaAlgorithm=iaAlgorithm,
        orchestrator_url=orchestrator_url
    )
    
    server.run()