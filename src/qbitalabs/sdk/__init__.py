"""
QBitaLabs Partner SDK

Client libraries and integration tools for partners:
- Python SDK for API access
- Webhook handling
- Event streaming
- Batch processing
- Partner authentication

Authored by: QbitaLab
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union
from urllib.parse import urljoin
import base64

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Authentication
# =============================================================================

@dataclass
class APICredentials:
    """API credentials for authentication."""
    api_key: str
    api_secret: str
    partner_id: Optional[str] = None

    def generate_signature(self, payload: str, timestamp: int) -> str:
        """Generate HMAC signature for request authentication."""
        message = f"{timestamp}.{payload}"
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()
        return signature


@dataclass
class AuthToken:
    """JWT authentication token."""
    access_token: str
    refresh_token: str
    expires_at: datetime
    token_type: str = "Bearer"

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() >= self.expires_at

    def to_header(self) -> Dict[str, str]:
        return {"Authorization": f"{self.token_type} {self.access_token}"}


# =============================================================================
# HTTP Client
# =============================================================================

class HTTPMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


@dataclass
class HTTPRequest:
    """HTTP request representation."""
    method: HTTPMethod
    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    json_body: Optional[Dict[str, Any]] = None
    timeout: float = 30.0


@dataclass
class HTTPResponse:
    """HTTP response representation."""
    status_code: int
    headers: Dict[str, str]
    body: Any
    elapsed_ms: float

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> Dict[str, Any]:
        if isinstance(self.body, dict):
            return self.body
        return json.loads(self.body) if self.body else {}


class HTTPClient(ABC):
    """Abstract HTTP client interface."""

    @abstractmethod
    async def request(self, request: HTTPRequest) -> HTTPResponse:
        """Execute an HTTP request."""
        pass


class SimpleHTTPClient(HTTPClient):
    """Simple async HTTP client using urllib."""

    def __init__(self, base_url: str = ""):
        self.base_url = base_url
        self._logger = structlog.get_logger("HTTPClient")

    async def request(self, request: HTTPRequest) -> HTTPResponse:
        """Execute an HTTP request."""
        import urllib.request
        import urllib.error

        url = urljoin(self.base_url, request.url)
        if request.params:
            from urllib.parse import urlencode
            url = f"{url}?{urlencode(request.params)}"

        headers = {**request.headers}
        body = None

        if request.json_body:
            body = json.dumps(request.json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=body, headers=headers, method=request.method.value)

        start_time = time.time()
        try:
            with urllib.request.urlopen(req, timeout=request.timeout) as response:
                response_body = response.read().decode("utf-8")
                elapsed_ms = (time.time() - start_time) * 1000

                return HTTPResponse(
                    status_code=response.status,
                    headers=dict(response.headers),
                    body=json.loads(response_body) if response_body else None,
                    elapsed_ms=elapsed_ms,
                )
        except urllib.error.HTTPError as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return HTTPResponse(
                status_code=e.code,
                headers=dict(e.headers),
                body=e.read().decode("utf-8"),
                elapsed_ms=elapsed_ms,
            )
        except Exception as e:
            self._logger.error("Request failed", url=url, error=str(e))
            raise


# =============================================================================
# QBitaLabs SDK Client
# =============================================================================

class QBitaLabsSDK:
    """
    Official QBitaLabs Python SDK for partner integrations.

    Usage:
        sdk = QBitaLabsSDK(
            base_url="https://api.qbitalabs.com",
            credentials=APICredentials(api_key="...", api_secret="...")
        )

        # Spawn swarm agents
        result = await sdk.swarm.spawn_agents("protein_analysis", count=10)

        # Run quantum simulation
        result = await sdk.quantum.run_molecular_simulation(
            smiles="CCO",
            method="VQE"
        )

        # Create digital twin
        twin = await sdk.digital_twin.create(patient_data)
    """

    def __init__(
        self,
        base_url: str,
        credentials: APICredentials,
        http_client: Optional[HTTPClient] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.credentials = credentials
        self.http_client = http_client or SimpleHTTPClient(self.base_url)
        self.timeout = timeout
        self.max_retries = max_retries
        self._token: Optional[AuthToken] = None
        self._logger = structlog.get_logger("QBitaLabsSDK")

        # Initialize service clients
        self.swarm = SwarmClient(self)
        self.quantum = QuantumClient(self)
        self.digital_twin = DigitalTwinClient(self)
        self.data = DataClient(self)
        self.models = ModelsClient(self)

    async def authenticate(self) -> AuthToken:
        """Authenticate with the API."""
        response = await self._request(
            HTTPMethod.POST,
            "/api/v1/auth/token",
            json_body={
                "api_key": self.credentials.api_key,
                "api_secret": self.credentials.api_secret,
            },
        )
        if not response.ok:
            raise AuthenticationError(f"Authentication failed: {response.body}")

        data = response.json()
        self._token = AuthToken(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=datetime.utcnow() + timedelta(seconds=data.get("expires_in", 3600)),
        )
        return self._token

    async def refresh_token(self) -> AuthToken:
        """Refresh the authentication token."""
        if not self._token:
            return await self.authenticate()

        response = await self._request(
            HTTPMethod.POST,
            "/api/v1/auth/refresh",
            json_body={"refresh_token": self._token.refresh_token},
        )
        if not response.ok:
            # Token refresh failed, re-authenticate
            return await self.authenticate()

        data = response.json()
        self._token = AuthToken(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", self._token.refresh_token),
            expires_at=datetime.utcnow() + timedelta(seconds=data.get("expires_in", 3600)),
        )
        return self._token

    async def _request(
        self,
        method: HTTPMethod,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        authenticated: bool = True,
    ) -> HTTPResponse:
        """Make an authenticated API request."""
        headers = {"Accept": "application/json"}

        if authenticated and self._token:
            if self._token.is_expired:
                await self.refresh_token()
            headers.update(self._token.to_header())

        # Add request signature
        timestamp = int(time.time())
        payload = json.dumps(json_body) if json_body else ""
        signature = self.credentials.generate_signature(payload, timestamp)
        headers["X-QBita-Timestamp"] = str(timestamp)
        headers["X-QBita-Signature"] = signature

        request = HTTPRequest(
            method=method,
            url=path,
            headers=headers,
            params=params or {},
            json_body=json_body,
            timeout=self.timeout,
        )

        # Retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await self.http_client.request(request)
                if response.ok or response.status_code < 500:
                    return response
                last_error = f"HTTP {response.status_code}: {response.body}"
            except Exception as e:
                last_error = str(e)

            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        raise APIError(f"Request failed after {self.max_retries} attempts: {last_error}")


class ServiceClient:
    """Base class for service-specific clients."""

    def __init__(self, sdk: QBitaLabsSDK):
        self.sdk = sdk
        self._logger = structlog.get_logger(self.__class__.__name__)


class SwarmClient(ServiceClient):
    """Client for SWARM agent operations."""

    async def spawn_agents(
        self,
        agent_type: str,
        count: int = 1,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Spawn swarm agents."""
        response = await self.sdk._request(
            HTTPMethod.POST,
            "/api/v1/swarm/spawn",
            json_body={
                "agent_type": agent_type,
                "count": count,
                "config": config or {},
            },
        )
        return response.json()

    async def get_status(self) -> Dict[str, Any]:
        """Get swarm status."""
        response = await self.sdk._request(HTTPMethod.GET, "/api/v1/swarm/status")
        return response.json()

    async def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 5,
    ) -> Dict[str, Any]:
        """Submit a task to the swarm."""
        response = await self.sdk._request(
            HTTPMethod.POST,
            "/api/v1/swarm/task",
            json_body={
                "task_type": task_type,
                "payload": payload,
                "priority": priority,
            },
        )
        return response.json()

    async def get_task_result(self, task_id: str) -> Dict[str, Any]:
        """Get task result by ID."""
        response = await self.sdk._request(
            HTTPMethod.GET,
            f"/api/v1/swarm/task/{task_id}",
        )
        return response.json()


class QuantumClient(ServiceClient):
    """Client for quantum computing operations."""

    async def run_circuit(
        self,
        circuit_type: str,
        n_qubits: int,
        backend: str = "simulator",
        shots: int = 1000,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a quantum circuit."""
        response = await self.sdk._request(
            HTTPMethod.POST,
            "/api/v1/quantum/run",
            json_body={
                "circuit_type": circuit_type,
                "n_qubits": n_qubits,
                "backend": backend,
                "shots": shots,
                "params": params or {},
            },
        )
        return response.json()

    async def run_molecular_simulation(
        self,
        smiles: str,
        method: str = "VQE",
        basis: str = "sto-3g",
    ) -> Dict[str, Any]:
        """Run molecular quantum simulation."""
        response = await self.sdk._request(
            HTTPMethod.POST,
            "/api/v1/quantum/molecular",
            json_body={
                "smiles": smiles,
                "method": method,
                "basis": basis,
            },
        )
        return response.json()

    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get quantum job status."""
        response = await self.sdk._request(
            HTTPMethod.GET,
            f"/api/v1/quantum/job/{job_id}",
        )
        return response.json()

    async def wait_for_job(
        self,
        job_id: str,
        timeout: float = 300,
        poll_interval: float = 2.0,
    ) -> Dict[str, Any]:
        """Wait for a quantum job to complete."""
        start = time.time()
        while time.time() - start < timeout:
            result = await self.get_job_status(job_id)
            if result.get("status") in ("completed", "failed"):
                return result
            await asyncio.sleep(poll_interval)
        raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


class DigitalTwinClient(ServiceClient):
    """Client for digital twin operations."""

    async def create(
        self,
        patient_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a digital twin."""
        response = await self.sdk._request(
            HTTPMethod.POST,
            "/api/v1/twin/create",
            json_body=patient_data,
        )
        return response.json()

    async def get(self, twin_id: str) -> Dict[str, Any]:
        """Get digital twin by ID."""
        response = await self.sdk._request(
            HTTPMethod.GET,
            f"/api/v1/twin/{twin_id}",
        )
        return response.json()

    async def simulate(
        self,
        twin_id: str,
        duration_days: int,
        interventions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run simulation on a digital twin."""
        response = await self.sdk._request(
            HTTPMethod.POST,
            "/api/v1/twin/simulate",
            json_body={
                "twin_id": twin_id,
                "duration_days": duration_days,
                "interventions": interventions or [],
            },
        )
        return response.json()

    async def predict_treatment(
        self,
        twin_id: str,
        drug: str,
        dose: float,
    ) -> Dict[str, Any]:
        """Predict treatment response."""
        response = await self.sdk._request(
            HTTPMethod.POST,
            "/api/v1/twin/treatment",
            json_body={
                "twin_id": twin_id,
                "drug": drug,
                "dose": dose,
            },
        )
        return response.json()


class DataClient(ServiceClient):
    """Client for data operations."""

    async def upload_dataset(
        self,
        name: str,
        data_type: str,
        records: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Upload a dataset."""
        response = await self.sdk._request(
            HTTPMethod.POST,
            "/api/v1/data/upload",
            json_body={
                "name": name,
                "data_type": data_type,
                "records": records,
                "metadata": metadata or {},
            },
        )
        return response.json()

    async def list_datasets(
        self,
        data_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List available datasets."""
        params = {"limit": limit}
        if data_type:
            params["data_type"] = data_type

        response = await self.sdk._request(
            HTTPMethod.GET,
            "/api/v1/data/list",
            params=params,
        )
        return response.json().get("datasets", [])

    async def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Get dataset by ID."""
        response = await self.sdk._request(
            HTTPMethod.GET,
            f"/api/v1/data/{dataset_id}",
        )
        return response.json()


class ModelsClient(ServiceClient):
    """Client for ML model operations."""

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        response = await self.sdk._request(HTTPMethod.GET, "/api/v1/models/list")
        return response.json().get("models", [])

    async def get_model(self, model_id: str) -> Dict[str, Any]:
        """Get model details."""
        response = await self.sdk._request(
            HTTPMethod.GET,
            f"/api/v1/models/{model_id}",
        )
        return response.json()

    async def predict(
        self,
        model_id: str,
        inputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run predictions with a model."""
        response = await self.sdk._request(
            HTTPMethod.POST,
            f"/api/v1/models/{model_id}/predict",
            json_body={"inputs": inputs},
        )
        return response.json()


# =============================================================================
# Webhook Handling
# =============================================================================

@dataclass
class WebhookEvent:
    """Represents a webhook event."""
    event_type: str
    payload: Dict[str, Any]
    timestamp: datetime
    event_id: str
    signature: str

    def verify_signature(self, secret: str) -> bool:
        """Verify the webhook signature."""
        payload_str = json.dumps(self.payload, sort_keys=True)
        expected = hmac.new(
            secret.encode(),
            f"{self.event_id}.{payload_str}".encode(),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, self.signature)


class WebhookHandler:
    """
    Handle incoming webhooks from QBitaLabs.

    Usage:
        handler = WebhookHandler(secret="your_webhook_secret")

        @handler.on("swarm.task.completed")
        async def handle_task_completed(event: WebhookEvent):
            print(f"Task completed: {event.payload}")

        # In your web framework:
        await handler.process(request_body, signature_header)
    """

    def __init__(self, secret: str):
        self.secret = secret
        self.handlers: Dict[str, List[Callable]] = {}
        self._logger = structlog.get_logger("WebhookHandler")

    def on(self, event_type: str) -> Callable:
        """Decorator to register an event handler."""
        def decorator(func: Callable) -> Callable:
            if event_type not in self.handlers:
                self.handlers[event_type] = []
            self.handlers[event_type].append(func)
            return func
        return decorator

    async def process(
        self,
        body: Union[str, bytes, Dict],
        signature: str,
    ) -> bool:
        """Process an incoming webhook."""
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        if isinstance(body, str):
            payload = json.loads(body)
        else:
            payload = body

        event = WebhookEvent(
            event_type=payload.get("event_type", "unknown"),
            payload=payload.get("data", {}),
            timestamp=datetime.fromisoformat(payload.get("timestamp", datetime.utcnow().isoformat())),
            event_id=payload.get("event_id", ""),
            signature=signature,
        )

        # Verify signature
        if not event.verify_signature(self.secret):
            self._logger.warning("Invalid webhook signature", event_id=event.event_id)
            return False

        # Call handlers
        handlers = self.handlers.get(event.event_type, []) + self.handlers.get("*", [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                self._logger.error(
                    "Webhook handler error",
                    event_type=event.event_type,
                    error=str(e),
                )

        return True


# =============================================================================
# Event Streaming
# =============================================================================

class EventStream:
    """
    Stream events from QBitaLabs in real-time.

    Usage:
        async for event in sdk.stream_events(["swarm.*", "quantum.*"]):
            print(event)
    """

    def __init__(self, sdk: QBitaLabsSDK, event_patterns: List[str]):
        self.sdk = sdk
        self.event_patterns = event_patterns
        self._running = False
        self._logger = structlog.get_logger("EventStream")

    async def __aiter__(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Async iterator for streaming events."""
        self._running = True
        last_event_id = None

        while self._running:
            try:
                response = await self.sdk._request(
                    HTTPMethod.GET,
                    "/api/v1/events/stream",
                    params={
                        "patterns": ",".join(self.event_patterns),
                        "last_event_id": last_event_id,
                    },
                )

                if response.ok:
                    events = response.json().get("events", [])
                    for event in events:
                        last_event_id = event.get("event_id")
                        yield event
                else:
                    self._logger.warning("Event stream error", status=response.status_code)

            except Exception as e:
                self._logger.error("Event stream exception", error=str(e))

            await asyncio.sleep(1)

    def stop(self) -> None:
        """Stop the event stream."""
        self._running = False


# =============================================================================
# Batch Processing
# =============================================================================

@dataclass
class BatchJob:
    """Represents a batch processing job."""
    job_id: str
    job_type: str
    status: str
    items_total: int
    items_processed: int
    items_failed: int
    created_at: datetime
    completed_at: Optional[datetime] = None
    results_url: Optional[str] = None
    error_message: Optional[str] = None

    @property
    def progress(self) -> float:
        return self.items_processed / self.items_total if self.items_total > 0 else 0.0


class BatchProcessor:
    """
    Submit and manage batch processing jobs.

    Usage:
        batch = BatchProcessor(sdk)
        job = await batch.submit(
            job_type="molecular_screening",
            items=[{"smiles": "CCO"}, {"smiles": "CC(=O)O"}]
        )
        result = await batch.wait_for_completion(job.job_id)
    """

    def __init__(self, sdk: QBitaLabsSDK):
        self.sdk = sdk
        self._logger = structlog.get_logger("BatchProcessor")

    async def submit(
        self,
        job_type: str,
        items: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ) -> BatchJob:
        """Submit a batch job."""
        response = await self.sdk._request(
            HTTPMethod.POST,
            "/api/v1/batch/submit",
            json_body={
                "job_type": job_type,
                "items": items,
                "config": config or {},
            },
        )

        data = response.json()
        return BatchJob(
            job_id=data["job_id"],
            job_type=job_type,
            status=data.get("status", "pending"),
            items_total=len(items),
            items_processed=0,
            items_failed=0,
            created_at=datetime.utcnow(),
        )

    async def get_status(self, job_id: str) -> BatchJob:
        """Get batch job status."""
        response = await self.sdk._request(
            HTTPMethod.GET,
            f"/api/v1/batch/{job_id}",
        )

        data = response.json()
        return BatchJob(
            job_id=data["job_id"],
            job_type=data["job_type"],
            status=data["status"],
            items_total=data["items_total"],
            items_processed=data["items_processed"],
            items_failed=data["items_failed"],
            created_at=datetime.fromisoformat(data["created_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            results_url=data.get("results_url"),
            error_message=data.get("error_message"),
        )

    async def wait_for_completion(
        self,
        job_id: str,
        timeout: float = 3600,
        poll_interval: float = 5.0,
        progress_callback: Optional[Callable[[BatchJob], None]] = None,
    ) -> BatchJob:
        """Wait for a batch job to complete."""
        start = time.time()
        while time.time() - start < timeout:
            job = await self.get_status(job_id)

            if progress_callback:
                progress_callback(job)

            if job.status in ("completed", "failed"):
                return job

            await asyncio.sleep(poll_interval)

        raise TimeoutError(f"Batch job {job_id} did not complete within {timeout}s")

    async def get_results(self, job_id: str) -> List[Dict[str, Any]]:
        """Get batch job results."""
        response = await self.sdk._request(
            HTTPMethod.GET,
            f"/api/v1/batch/{job_id}/results",
        )
        return response.json().get("results", [])


# =============================================================================
# Partner Registry
# =============================================================================

@dataclass
class PartnerInfo:
    """Information about a partner."""
    partner_id: str
    name: str
    tier: str  # free, standard, enterprise
    rate_limits: Dict[str, int]
    capabilities: List[str]
    webhook_url: Optional[str] = None
    contact_email: Optional[str] = None


class PartnerRegistry:
    """Registry for partner management and capabilities."""

    def __init__(self, sdk: QBitaLabsSDK):
        self.sdk = sdk
        self._logger = structlog.get_logger("PartnerRegistry")

    async def get_info(self) -> PartnerInfo:
        """Get current partner information."""
        response = await self.sdk._request(
            HTTPMethod.GET,
            "/api/v1/partner/info",
        )

        data = response.json()
        return PartnerInfo(
            partner_id=data["partner_id"],
            name=data["name"],
            tier=data["tier"],
            rate_limits=data["rate_limits"],
            capabilities=data["capabilities"],
            webhook_url=data.get("webhook_url"),
            contact_email=data.get("contact_email"),
        )

    async def update_webhook_url(self, webhook_url: str) -> bool:
        """Update partner webhook URL."""
        response = await self.sdk._request(
            HTTPMethod.PUT,
            "/api/v1/partner/webhook",
            json_body={"webhook_url": webhook_url},
        )
        return response.ok

    async def list_capabilities(self) -> List[str]:
        """List available capabilities for this partner."""
        response = await self.sdk._request(
            HTTPMethod.GET,
            "/api/v1/partner/capabilities",
        )
        return response.json().get("capabilities", [])


# =============================================================================
# Exceptions
# =============================================================================

class SDKError(Exception):
    """Base exception for SDK errors."""
    pass


class AuthenticationError(SDKError):
    """Authentication failed."""
    pass


class APIError(SDKError):
    """API request failed."""
    pass


class RateLimitError(APIError):
    """Rate limit exceeded."""
    pass


class ValidationError(SDKError):
    """Validation failed."""
    pass


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Authentication
    "APICredentials",
    "AuthToken",

    # HTTP
    "HTTPMethod",
    "HTTPRequest",
    "HTTPResponse",
    "HTTPClient",
    "SimpleHTTPClient",

    # SDK
    "QBitaLabsSDK",
    "SwarmClient",
    "QuantumClient",
    "DigitalTwinClient",
    "DataClient",
    "ModelsClient",

    # Webhooks
    "WebhookEvent",
    "WebhookHandler",

    # Events
    "EventStream",

    # Batch processing
    "BatchJob",
    "BatchProcessor",

    # Partner
    "PartnerInfo",
    "PartnerRegistry",

    # Exceptions
    "SDKError",
    "AuthenticationError",
    "APIError",
    "RateLimitError",
    "ValidationError",
]
