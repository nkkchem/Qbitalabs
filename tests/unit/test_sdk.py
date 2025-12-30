"""
QbitaLab: Unit tests for Partner SDK.

Tests:
- Authentication
- SDK client
- Webhook handling
- Batch processing
"""

import pytest
import json
import hashlib
import hmac
from datetime import datetime, timedelta


class TestAPICredentials:
    """Tests for API credentials."""

    def test_signature_generation(self):
        """Test HMAC signature generation."""
        from qbitalabs.sdk import APICredentials

        creds = APICredentials(
            api_key="test_key",
            api_secret="test_secret",
        )

        payload = '{"test": "data"}'
        timestamp = 1234567890

        signature = creds.generate_signature(payload, timestamp)

        # Verify signature format
        assert len(signature) == 64  # SHA256 hex digest
        assert all(c in "0123456789abcdef" for c in signature)

        # Verify deterministic
        sig2 = creds.generate_signature(payload, timestamp)
        assert signature == sig2

        # Different payload = different signature
        sig3 = creds.generate_signature('{"other": "data"}', timestamp)
        assert signature != sig3


class TestAuthToken:
    """Tests for authentication tokens."""

    def test_token_expiration(self):
        """Test token expiration check."""
        from qbitalabs.sdk import AuthToken

        # Not expired
        token = AuthToken(
            access_token="test",
            refresh_token="refresh",
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        assert not token.is_expired

        # Expired
        expired_token = AuthToken(
            access_token="test",
            refresh_token="refresh",
            expires_at=datetime.utcnow() - timedelta(hours=1),
        )
        assert expired_token.is_expired

    def test_token_header(self):
        """Test authorization header generation."""
        from qbitalabs.sdk import AuthToken

        token = AuthToken(
            access_token="my_access_token",
            refresh_token="refresh",
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )

        header = token.to_header()
        assert header["Authorization"] == "Bearer my_access_token"


class TestWebhookHandler:
    """Tests for webhook handling."""

    def test_webhook_event_verification(self):
        """Test webhook signature verification."""
        from qbitalabs.sdk import WebhookEvent

        secret = "webhook_secret"
        event_id = "evt_123"
        payload = {"key": "value"}

        # Generate valid signature
        payload_str = json.dumps(payload, sort_keys=True)
        valid_signature = hmac.new(
            secret.encode(),
            f"{event_id}.{payload_str}".encode(),
            hashlib.sha256,
        ).hexdigest()

        event = WebhookEvent(
            event_type="test.event",
            payload=payload,
            timestamp=datetime.utcnow(),
            event_id=event_id,
            signature=valid_signature,
        )

        assert event.verify_signature(secret)

        # Invalid signature
        event.signature = "invalid_signature"
        assert not event.verify_signature(secret)

    @pytest.mark.asyncio
    async def test_webhook_handler_registration(self):
        """Test handler registration and dispatch."""
        from qbitalabs.sdk import WebhookHandler

        handler = WebhookHandler(secret="test_secret")
        received_events = []

        @handler.on("test.event")
        async def handle_test(event):
            received_events.append(event)

        # Create valid webhook payload
        payload = {
            "event_type": "test.event",
            "event_id": "evt_123",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {"test": "data"},
        }

        payload_str = json.dumps({"test": "data"}, sort_keys=True)
        signature = hmac.new(
            "test_secret".encode(),
            f"evt_123.{payload_str}".encode(),
            hashlib.sha256,
        ).hexdigest()

        result = await handler.process(payload, signature)

        assert result
        assert len(received_events) == 1
        assert received_events[0].payload == {"test": "data"}

    @pytest.mark.asyncio
    async def test_wildcard_handler(self):
        """Test wildcard event handler."""
        from qbitalabs.sdk import WebhookHandler

        handler = WebhookHandler(secret="secret")
        all_events = []

        @handler.on("*")
        def handle_all(event):
            all_events.append(event.event_type)

        payload = {
            "event_type": "any.event.type",
            "event_id": "evt_1",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {},
        }

        signature = hmac.new(
            "secret".encode(),
            "evt_1.{}".encode(),
            hashlib.sha256,
        ).hexdigest()

        await handler.process(payload, signature)

        assert "any.event.type" in all_events


class TestBatchJob:
    """Tests for batch job data structure."""

    def test_batch_job_progress(self):
        """Test batch job progress calculation."""
        from qbitalabs.sdk import BatchJob

        job = BatchJob(
            job_id="batch_123",
            job_type="molecular_screening",
            status="running",
            items_total=100,
            items_processed=50,
            items_failed=5,
            created_at=datetime.utcnow(),
        )

        assert job.progress == 0.5
        assert job.items_processed == 50

    def test_batch_job_zero_items(self):
        """Test progress with zero items."""
        from qbitalabs.sdk import BatchJob

        job = BatchJob(
            job_id="batch_empty",
            job_type="test",
            status="pending",
            items_total=0,
            items_processed=0,
            items_failed=0,
            created_at=datetime.utcnow(),
        )

        assert job.progress == 0.0


class TestSDKExceptions:
    """Tests for SDK exceptions."""

    def test_exception_hierarchy(self):
        """Test exception inheritance."""
        from qbitalabs.sdk import (
            SDKError,
            AuthenticationError,
            APIError,
            RateLimitError,
            ValidationError,
        )

        assert issubclass(AuthenticationError, SDKError)
        assert issubclass(APIError, SDKError)
        assert issubclass(RateLimitError, APIError)
        assert issubclass(ValidationError, SDKError)

    def test_exception_messages(self):
        """Test exception messages."""
        from qbitalabs.sdk import AuthenticationError, APIError

        auth_error = AuthenticationError("Invalid credentials")
        assert str(auth_error) == "Invalid credentials"

        api_error = APIError("Request failed")
        assert str(api_error) == "Request failed"


class TestHTTPRequest:
    """Tests for HTTP request handling."""

    def test_request_creation(self):
        """Test HTTP request object creation."""
        from qbitalabs.sdk import HTTPRequest, HTTPMethod

        request = HTTPRequest(
            method=HTTPMethod.POST,
            url="/api/v1/test",
            headers={"Content-Type": "application/json"},
            json_body={"key": "value"},
            timeout=30.0,
        )

        assert request.method == HTTPMethod.POST
        assert request.url == "/api/v1/test"
        assert request.json_body == {"key": "value"}

    def test_response_ok(self):
        """Test HTTP response status check."""
        from qbitalabs.sdk import HTTPResponse

        ok_response = HTTPResponse(
            status_code=200,
            headers={},
            body='{"result": "success"}',
            elapsed_ms=50.0,
        )
        assert ok_response.ok

        error_response = HTTPResponse(
            status_code=400,
            headers={},
            body='{"error": "bad request"}',
            elapsed_ms=50.0,
        )
        assert not error_response.ok

        server_error = HTTPResponse(
            status_code=500,
            headers={},
            body='{"error": "internal error"}',
            elapsed_ms=50.0,
        )
        assert not server_error.ok


class TestPartnerInfo:
    """Tests for partner information."""

    def test_partner_info_structure(self):
        """Test PartnerInfo data structure."""
        from qbitalabs.sdk import PartnerInfo

        partner = PartnerInfo(
            partner_id="partner_123",
            name="Test Partner",
            tier="enterprise",
            rate_limits={"requests_per_minute": 1000},
            capabilities=["swarm", "quantum", "twin"],
            webhook_url="https://example.com/webhook",
            contact_email="contact@example.com",
        )

        assert partner.partner_id == "partner_123"
        assert partner.tier == "enterprise"
        assert "swarm" in partner.capabilities
        assert partner.rate_limits["requests_per_minute"] == 1000
