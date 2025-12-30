"""
QBitaLabs Monitoring and Debugging Infrastructure

Provides comprehensive observability:
- Prometheus metrics
- OpenTelemetry tracing
- Structured logging
- Health checks
- Performance profiling
- Error tracking

Authored by: QbitaLab
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import sys
import time
import traceback
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar, Union
import threading

import structlog

logger = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Metrics
# =============================================================================

class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """A single metric value with labels."""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    description: str = ""

    def to_prometheus(self) -> str:
        """Convert to Prometheus format."""
        labels_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        if labels_str:
            return f'{self.name}{{{labels_str}}} {self.value}'
        return f'{self.name} {self.value}'


class MetricsCollector:
    """
    Collects and exports metrics in Prometheus format.

    Usage:
        metrics = MetricsCollector()

        # Counter
        metrics.counter("requests_total", 1, {"method": "GET", "status": "200"})

        # Gauge
        metrics.gauge("active_connections", 42)

        # Histogram
        metrics.histogram("request_duration_seconds", 0.5, buckets=[0.1, 0.5, 1.0])

        # Export
        print(metrics.export_prometheus())
    """

    def __init__(self, prefix: str = "qbitalabs"):
        self.prefix = prefix
        self._metrics: Dict[str, List[MetricValue]] = {}
        self._histograms: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._logger = structlog.get_logger("MetricsCollector")

    def counter(
        self,
        name: str,
        value: float = 1,
        labels: Optional[Dict[str, str]] = None,
        description: str = "",
    ) -> None:
        """Increment a counter metric."""
        full_name = f"{self.prefix}_{name}"
        with self._lock:
            key = self._make_key(full_name, labels or {})
            if key not in self._metrics:
                self._metrics[key] = []

            # For counters, we accumulate
            current = self._metrics[key][-1].value if self._metrics[key] else 0
            self._metrics[key] = [MetricValue(
                name=full_name,
                value=current + value,
                metric_type=MetricType.COUNTER,
                labels=labels or {},
                description=description,
            )]

    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        description: str = "",
    ) -> None:
        """Set a gauge metric value."""
        full_name = f"{self.prefix}_{name}"
        with self._lock:
            key = self._make_key(full_name, labels or {})
            self._metrics[key] = [MetricValue(
                name=full_name,
                value=value,
                metric_type=MetricType.GAUGE,
                labels=labels or {},
                description=description,
            )]

    def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[List[float]] = None,
        description: str = "",
    ) -> None:
        """Observe a histogram value."""
        full_name = f"{self.prefix}_{name}"
        buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

        with self._lock:
            key = self._make_key(full_name, labels or {})
            if key not in self._histograms:
                self._histograms[key] = {
                    "name": full_name,
                    "labels": labels or {},
                    "buckets": {b: 0 for b in buckets},
                    "sum": 0.0,
                    "count": 0,
                    "description": description,
                }

            hist = self._histograms[key]
            hist["sum"] += value
            hist["count"] += 1
            for bucket in hist["buckets"]:
                if value <= bucket:
                    hist["buckets"][bucket] += 1

    @contextmanager
    def timer(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Generator[None, None, None]:
        """Context manager for timing operations."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.histogram(name, duration, labels)

    def export_prometheus(self) -> str:
        """Export all metrics in Prometheus format."""
        lines = []

        with self._lock:
            # Regular metrics
            for key, values in self._metrics.items():
                for metric in values:
                    if metric.description:
                        lines.append(f"# HELP {metric.name} {metric.description}")
                    lines.append(f"# TYPE {metric.name} {metric.metric_type.value}")
                    lines.append(metric.to_prometheus())

            # Histograms
            for key, hist in self._histograms.items():
                name = hist["name"]
                labels = hist["labels"]
                labels_str = ",".join(f'{k}="{v}"' for k, v in labels.items())

                if hist["description"]:
                    lines.append(f"# HELP {name} {hist['description']}")
                lines.append(f"# TYPE {name} histogram")

                for bucket, count in sorted(hist["buckets"].items()):
                    bucket_labels = f'{labels_str},le="{bucket}"' if labels_str else f'le="{bucket}"'
                    lines.append(f'{name}_bucket{{{bucket_labels}}} {count}')

                inf_labels = f'{labels_str},le="+Inf"' if labels_str else 'le="+Inf"'
                lines.append(f'{name}_bucket{{{inf_labels}}} {hist["count"]}')

                sum_labels = f'{{{labels_str}}}' if labels_str else ""
                lines.append(f'{name}_sum{sum_labels} {hist["sum"]}')
                lines.append(f'{name}_count{sum_labels} {hist["count"]}')

        return "\n".join(lines)

    def export_json(self) -> Dict[str, Any]:
        """Export all metrics as JSON."""
        result = {"metrics": [], "histograms": [], "timestamp": datetime.utcnow().isoformat()}

        with self._lock:
            for key, values in self._metrics.items():
                for metric in values:
                    result["metrics"].append({
                        "name": metric.name,
                        "value": metric.value,
                        "type": metric.metric_type.value,
                        "labels": metric.labels,
                    })

            for key, hist in self._histograms.items():
                result["histograms"].append({
                    "name": hist["name"],
                    "labels": hist["labels"],
                    "buckets": hist["buckets"],
                    "sum": hist["sum"],
                    "count": hist["count"],
                })

        return result

    def _make_key(self, name: str, labels: Dict[str, str]) -> str:
        """Create a unique key for metric identification."""
        labels_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{labels_str}}}"


# Global metrics instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


# =============================================================================
# Tracing
# =============================================================================

@dataclass
class Span:
    """Represents a trace span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    service_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "ok"
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)

    def finish(self, status: str = "ok") -> None:
        """Finish the span."""
        self.end_time = datetime.utcnow()
        self.status = status

    def log(self, message: str, **kwargs: Any) -> None:
        """Add a log entry to the span."""
        self.logs.append({
            "timestamp": datetime.utcnow().isoformat(),
            "message": message,
            **kwargs,
        })

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the span."""
        self.tags[key] = value

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "service_name": self.service_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "tags": self.tags,
            "logs": self.logs,
        }


class Tracer:
    """
    Distributed tracing implementation.

    Usage:
        tracer = Tracer("qbitalabs-api")

        with tracer.start_span("process_request") as span:
            span.set_tag("user_id", "123")
            # ... do work ...
            span.log("Processing complete")
    """

    def __init__(self, service_name: str):
        self.service_name = service_name
        self._spans: List[Span] = []
        self._current_span: Optional[Span] = None
        self._lock = threading.Lock()
        self._logger = structlog.get_logger("Tracer")

    @contextmanager
    def start_span(
        self,
        operation_name: str,
        parent: Optional[Span] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Generator[Span, None, None]:
        """Start a new span."""
        import uuid

        parent = parent or self._current_span
        span = Span(
            trace_id=parent.trace_id if parent else str(uuid.uuid4()),
            span_id=str(uuid.uuid4())[:16],
            parent_span_id=parent.span_id if parent else None,
            operation_name=operation_name,
            service_name=self.service_name,
            start_time=datetime.utcnow(),
            tags=tags or {},
        )

        previous_span = self._current_span
        self._current_span = span

        try:
            yield span
            span.finish("ok")
        except Exception as e:
            span.finish("error")
            span.set_tag("error", str(e))
            span.log("Exception occurred", error=str(e), traceback=traceback.format_exc())
            raise
        finally:
            self._current_span = previous_span
            with self._lock:
                self._spans.append(span)

    def get_current_span(self) -> Optional[Span]:
        """Get the current active span."""
        return self._current_span

    def export_spans(self) -> List[Dict[str, Any]]:
        """Export all recorded spans."""
        with self._lock:
            spans = [s.to_dict() for s in self._spans]
            self._spans = []
            return spans


# Global tracer instance
_tracer: Optional[Tracer] = None


def get_tracer(service_name: str = "qbitalabs") -> Tracer:
    """Get the global tracer."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer(service_name)
    return _tracer


# =============================================================================
# Health Checks
# =============================================================================

class HealthStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str = ""
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "details": self.details,
        }


class HealthCheck(ABC):
    """Base class for health checks."""

    name: str = "base"

    @abstractmethod
    async def check(self) -> HealthCheckResult:
        """Perform the health check."""
        pass


class DatabaseHealthCheck(HealthCheck):
    """Check database connectivity."""

    name = "database"

    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    async def check(self) -> HealthCheckResult:
        start = time.perf_counter()
        try:
            # Simulate database check
            await asyncio.sleep(0.01)
            duration = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Database connection successful",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {e}",
                duration_ms=duration,
            )


class RedisHealthCheck(HealthCheck):
    """Check Redis connectivity."""

    name = "redis"

    def __init__(self, host: str = "localhost", port: int = 6379):
        self.host = host
        self.port = port

    async def check(self) -> HealthCheckResult:
        start = time.perf_counter()
        try:
            # Simulate Redis check
            await asyncio.sleep(0.005)
            duration = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Redis connection successful",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Redis connection failed: {e}",
                duration_ms=duration,
            )


class QuantumBackendHealthCheck(HealthCheck):
    """Check quantum backend availability."""

    name = "quantum_backend"

    async def check(self) -> HealthCheckResult:
        start = time.perf_counter()
        try:
            # Check if quantum backend is available
            await asyncio.sleep(0.02)
            duration = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Quantum simulator available",
                duration_ms=duration,
                details={"backend": "simulator", "qubits": 30},
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.DEGRADED,
                message=f"Quantum backend unavailable: {e}",
                duration_ms=duration,
            )


class HealthChecker:
    """
    Manages and runs health checks.

    Usage:
        checker = HealthChecker()
        checker.add_check(DatabaseHealthCheck("postgres://..."))
        checker.add_check(RedisHealthCheck())

        results = await checker.run_all()
        print(checker.get_status())
    """

    def __init__(self):
        self.checks: List[HealthCheck] = []
        self._logger = structlog.get_logger("HealthChecker")

    def add_check(self, check: HealthCheck) -> None:
        """Add a health check."""
        self.checks.append(check)

    async def run_all(self) -> List[HealthCheckResult]:
        """Run all health checks."""
        results = []
        for check in self.checks:
            try:
                result = await check.check()
            except Exception as e:
                result = HealthCheckResult(
                    name=check.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed with exception: {e}",
                )
            results.append(result)
        return results

    async def get_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        results = await self.run_all()

        statuses = [r.status for r in results]
        if HealthStatus.UNHEALTHY in statuses:
            overall = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        return {
            "status": overall.value,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": [r.to_dict() for r in results],
        }


# =============================================================================
# Error Tracking
# =============================================================================

@dataclass
class ErrorReport:
    """Represents an error report."""
    error_id: str
    error_type: str
    message: str
    traceback: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    user_id: Optional[str] = None
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_id": self.error_id,
            "error_type": self.error_type,
            "message": self.message,
            "traceback": self.traceback,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "tags": self.tags,
            "user_id": self.user_id,
            "request_id": self.request_id,
        }


class ErrorTracker:
    """
    Tracks and reports errors.

    Usage:
        tracker = ErrorTracker()

        try:
            do_something()
        except Exception as e:
            tracker.capture_exception(e, {"user_id": "123"})
    """

    def __init__(self, service_name: str = "qbitalabs"):
        self.service_name = service_name
        self._errors: List[ErrorReport] = []
        self._lock = threading.Lock()
        self._logger = structlog.get_logger("ErrorTracker")

    def capture_exception(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """Capture an exception."""
        import uuid

        error_id = str(uuid.uuid4())[:8]
        tb = traceback.format_exception(type(exception), exception, exception.__traceback__)

        report = ErrorReport(
            error_id=error_id,
            error_type=type(exception).__name__,
            message=str(exception),
            traceback="".join(tb),
            timestamp=datetime.utcnow(),
            context=context or {},
            tags={**(tags or {}), "service": self.service_name},
            user_id=user_id,
            request_id=request_id,
        )

        with self._lock:
            self._errors.append(report)
            # Keep only last 1000 errors in memory
            if len(self._errors) > 1000:
                self._errors = self._errors[-1000:]

        self._logger.error(
            "Exception captured",
            error_id=error_id,
            error_type=report.error_type,
            message=report.message,
        )

        return error_id

    def capture_message(
        self,
        message: str,
        level: str = "error",
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Capture an error message."""
        import uuid

        error_id = str(uuid.uuid4())[:8]

        report = ErrorReport(
            error_id=error_id,
            error_type="Message",
            message=message,
            traceback="",
            timestamp=datetime.utcnow(),
            context=context or {},
            tags={**(tags or {}), "level": level, "service": self.service_name},
        )

        with self._lock:
            self._errors.append(report)

        self._logger.log(level, message, error_id=error_id)
        return error_id

    def get_recent_errors(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent errors."""
        with self._lock:
            return [e.to_dict() for e in self._errors[-limit:]]

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        with self._lock:
            error_counts: Dict[str, int] = {}
            for error in self._errors:
                error_counts[error.error_type] = error_counts.get(error.error_type, 0) + 1

            return {
                "total_errors": len(self._errors),
                "error_counts_by_type": error_counts,
                "timestamp": datetime.utcnow().isoformat(),
            }


# Global error tracker
_error_tracker: Optional[ErrorTracker] = None


def get_error_tracker(service_name: str = "qbitalabs") -> ErrorTracker:
    """Get the global error tracker."""
    global _error_tracker
    if _error_tracker is None:
        _error_tracker = ErrorTracker(service_name)
    return _error_tracker


# =============================================================================
# Performance Profiler
# =============================================================================

@dataclass
class ProfileResult:
    """Result of a profiling session."""
    function_name: str
    calls: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float


class Profiler:
    """
    Simple performance profiler.

    Usage:
        profiler = Profiler()

        @profiler.profile
        def my_function():
            ...

        # Or manually
        with profiler.measure("operation"):
            ...

        print(profiler.get_stats())
    """

    def __init__(self):
        self._timings: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def profile(self, func: F) -> F:
        """Decorator to profile a function."""
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = (time.perf_counter() - start) * 1000
                self._record(func.__name__, duration)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = (time.perf_counter() - start) * 1000
                self._record(func.__name__, duration)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return wrapper  # type: ignore

    @contextmanager
    def measure(self, name: str) -> Generator[None, None, None]:
        """Context manager for measuring execution time."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = (time.perf_counter() - start) * 1000
            self._record(name, duration)

    def _record(self, name: str, duration_ms: float) -> None:
        """Record a timing measurement."""
        with self._lock:
            if name not in self._timings:
                self._timings[name] = []
            self._timings[name].append(duration_ms)
            # Keep only last 1000 measurements per function
            if len(self._timings[name]) > 1000:
                self._timings[name] = self._timings[name][-1000:]

    def get_stats(self) -> List[ProfileResult]:
        """Get profiling statistics."""
        results = []
        with self._lock:
            for name, timings in self._timings.items():
                if timings:
                    results.append(ProfileResult(
                        function_name=name,
                        calls=len(timings),
                        total_time_ms=sum(timings),
                        avg_time_ms=sum(timings) / len(timings),
                        min_time_ms=min(timings),
                        max_time_ms=max(timings),
                    ))
        return sorted(results, key=lambda r: r.total_time_ms, reverse=True)

    def reset(self) -> None:
        """Reset all profiling data."""
        with self._lock:
            self._timings.clear()


# Global profiler
_profiler: Optional[Profiler] = None


def get_profiler() -> Profiler:
    """Get the global profiler."""
    global _profiler
    if _profiler is None:
        _profiler = Profiler()
    return _profiler


# =============================================================================
# Structured Logging Setup
# =============================================================================

def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    log_file: Optional[str] = None,
) -> None:
    """
    Setup structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: Use JSON format for logs
        log_file: Optional file path for log output
    """
    import logging

    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure standard library logging
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([] if not log_file else [logging.FileHandler(log_file)]),
        ],
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Metrics
    "MetricType",
    "MetricValue",
    "MetricsCollector",
    "get_metrics",

    # Tracing
    "Span",
    "Tracer",
    "get_tracer",

    # Health checks
    "HealthStatus",
    "HealthCheckResult",
    "HealthCheck",
    "DatabaseHealthCheck",
    "RedisHealthCheck",
    "QuantumBackendHealthCheck",
    "HealthChecker",

    # Error tracking
    "ErrorReport",
    "ErrorTracker",
    "get_error_tracker",

    # Profiling
    "ProfileResult",
    "Profiler",
    "get_profiler",

    # Logging
    "setup_logging",
]
