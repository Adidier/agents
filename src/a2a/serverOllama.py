
from flask import Flask, request, jsonify, Response, stream_with_context
from typing import Dict, Any, List, Optional, Callable

from a2a.server import A2AServer
from a2a.core.a2a_ia_algorithm_interface import IA2AIAAlgorithm

def run_server(
    port: int = 8000,
    endpoint: str = None,
    webhook_url: str = None,
    iaAlgorithm: IA2AIAAlgorithm = None
):

    server = A2AServer(
        port=port,
        endpoint=endpoint,
        webhook_url=webhook_url,
        iaAlgorithm=iaAlgorithm
    )
    
    server.run()