from abc import ABC, abstractmethod
from typing import Dict, Any, List, Iterator

class IA2AIAAlgorithm(ABC):
    @abstractmethod
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process an A2A request and return the response."""
        pass

    @abstractmethod
    def _process_task(self, task_id: str) -> Dict[str, Any]:
        """Process a task and return the result."""
        pass

    @abstractmethod
    def _process_task_stream(self, task_id: str) -> Iterator[Dict[str, Any]]:
        """Process a task and yield partial results (streaming)."""
        pass

    @abstractmethod
    def _get_ollama_messages(self, task_id: str) -> List[Dict[str, Any]]:
        """Convert A2A messages to Algorithm format."""
        pass

    @abstractmethod
    def configure_mcp_client(self, mcp_client: Any) -> None:
        """Configure the MCP client for tool access."""
        pass