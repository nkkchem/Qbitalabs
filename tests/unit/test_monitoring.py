"""
QbitaLab: Unit tests for monitoring infrastructure.

Tests:
- Metrics collection
- Tracing
- Health checks
- Error tracking
- Profiling
"""

import pytest
import asyncio
from datetime import datetime


class TestMetricsCollector:
    """Tests for metrics collection."""

    def test_counter(self):
        """Test counter metric."""
        from qbitalabs.monitoring import MetricsCollector

        collector = MetricsCollector(prefix="test")

        collector.counter("requests_total", 1, {"method": "GET"})
        collector.counter("requests_total", 1, {"method": "GET"})
        collector.counter("requests_total", 1, {"method": "POST"})

        export = collector.export_prometheus()

        assert "test_requests_total" in export
        assert 'method="GET"' in export

    def test_gauge(self):
        """Test gauge metric."""
        from qbitalabs.monitoring import MetricsCollector

        collector = MetricsCollector(prefix="test")

        collector.gauge("temperature", 25.5)
        collector.gauge("temperature", 26.0)  # Override

        export = collector.export_prometheus()
        assert "test_temperature 26.0" in export

    def test_histogram(self):
        """Test histogram metric."""
        from qbitalabs.monitoring import MetricsCollector

        collector = MetricsCollector(prefix="test")

        for value in [0.1, 0.2, 0.3, 0.5, 1.0, 2.0]:
            collector.histogram("request_duration", value)

        export = collector.export_prometheus()

        assert "test_request_duration_bucket" in export
        assert "test_request_duration_sum" in export
        assert "test_request_duration_count" in export

    def test_timer_context_manager(self):
        """Test timer context manager."""
        from qbitalabs.monitoring import MetricsCollector
        import time

        collector = MetricsCollector(prefix="test")

        with collector.timer("operation_duration"):
            time.sleep(0.01)

        export = collector.export_json()

        assert len(export["histograms"]) == 1
        hist = export["histograms"][0]
        assert hist["count"] == 1
        assert hist["sum"] > 0

    def test_export_json(self):
        """Test JSON export."""
        from qbitalabs.monitoring import MetricsCollector

        collector = MetricsCollector(prefix="test")

        collector.counter("count", 5)
        collector.gauge("gauge", 10.0)

        export = collector.export_json()

        assert "metrics" in export
        assert "timestamp" in export
        assert len(export["metrics"]) == 2


class TestTracer:
    """Tests for distributed tracing."""

    def test_span_creation(self):
        """Test span creation and finishing."""
        from qbitalabs.monitoring import Tracer

        tracer = Tracer("test-service")

        with tracer.start_span("test_operation") as span:
            span.set_tag("user_id", "123")
            span.log("Processing started")

        assert span.end_time is not None
        assert span.status == "ok"
        assert span.tags["user_id"] == "123"
        assert len(span.logs) == 1

    def test_span_hierarchy(self):
        """Test parent-child span relationship."""
        from qbitalabs.monitoring import Tracer

        tracer = Tracer("test-service")

        with tracer.start_span("parent") as parent:
            with tracer.start_span("child") as child:
                pass

        assert child.parent_span_id == parent.span_id
        assert child.trace_id == parent.trace_id

    def test_span_error_handling(self):
        """Test span error capture."""
        from qbitalabs.monitoring import Tracer

        tracer = Tracer("test-service")

        with pytest.raises(ValueError):
            with tracer.start_span("failing_operation") as span:
                raise ValueError("Test error")

        assert span.status == "error"
        assert "error" in span.tags
        assert any("ValueError" in log.get("message", "") or "Test error" in str(log) for log in span.logs)

    def test_span_duration(self):
        """Test span duration calculation."""
        from qbitalabs.monitoring import Tracer
        import time

        tracer = Tracer("test-service")

        with tracer.start_span("timed_operation") as span:
            time.sleep(0.05)

        assert span.duration_ms >= 40  # At least 40ms

    def test_export_spans(self):
        """Test span export."""
        from qbitalabs.monitoring import Tracer

        tracer = Tracer("test-service")

        with tracer.start_span("op1"):
            pass
        with tracer.start_span("op2"):
            pass

        spans = tracer.export_spans()

        assert len(spans) == 2
        assert spans[0]["operation_name"] == "op1"
        assert spans[1]["operation_name"] == "op2"

        # Export clears spans
        spans2 = tracer.export_spans()
        assert len(spans2) == 0


class TestHealthChecker:
    """Tests for health checks."""

    @pytest.mark.asyncio
    async def test_health_check_execution(self):
        """Test running health checks."""
        from qbitalabs.monitoring import HealthChecker, DatabaseHealthCheck, RedisHealthCheck

        checker = HealthChecker()
        checker.add_check(DatabaseHealthCheck("test://db"))
        checker.add_check(RedisHealthCheck())

        results = await checker.run_all()

        assert len(results) == 2
        assert all(r.duration_ms >= 0 for r in results)

    @pytest.mark.asyncio
    async def test_health_status_aggregation(self):
        """Test overall health status."""
        from qbitalabs.monitoring import HealthChecker, HealthCheck, HealthCheckResult, HealthStatus

        class HealthyCheck(HealthCheck):
            name = "healthy"

            async def check(self):
                return HealthCheckResult(name=self.name, status=HealthStatus.HEALTHY)

        class UnhealthyCheck(HealthCheck):
            name = "unhealthy"

            async def check(self):
                return HealthCheckResult(name=self.name, status=HealthStatus.UNHEALTHY)

        checker = HealthChecker()
        checker.add_check(HealthyCheck())
        checker.add_check(UnhealthyCheck())

        status = await checker.get_status()

        assert status["status"] == "unhealthy"  # One unhealthy = overall unhealthy

    @pytest.mark.asyncio
    async def test_degraded_status(self):
        """Test degraded health status."""
        from qbitalabs.monitoring import HealthChecker, HealthCheck, HealthCheckResult, HealthStatus

        class HealthyCheck(HealthCheck):
            name = "healthy"

            async def check(self):
                return HealthCheckResult(name=self.name, status=HealthStatus.HEALTHY)

        class DegradedCheck(HealthCheck):
            name = "degraded"

            async def check(self):
                return HealthCheckResult(name=self.name, status=HealthStatus.DEGRADED)

        checker = HealthChecker()
        checker.add_check(HealthyCheck())
        checker.add_check(DegradedCheck())

        status = await checker.get_status()

        assert status["status"] == "degraded"


class TestErrorTracker:
    """Tests for error tracking."""

    def test_capture_exception(self):
        """Test exception capture."""
        from qbitalabs.monitoring import ErrorTracker

        tracker = ErrorTracker("test-service")

        try:
            raise ValueError("Test error")
        except Exception as e:
            error_id = tracker.capture_exception(
                e,
                context={"user_id": "123"},
                tags={"component": "test"},
            )

        assert len(error_id) == 8

        errors = tracker.get_recent_errors(limit=10)
        assert len(errors) == 1
        assert errors[0]["error_type"] == "ValueError"
        assert errors[0]["message"] == "Test error"

    def test_capture_message(self):
        """Test message capture."""
        from qbitalabs.monitoring import ErrorTracker

        tracker = ErrorTracker("test-service")

        error_id = tracker.capture_message(
            "Something went wrong",
            level="warning",
            context={"operation": "test"},
        )

        errors = tracker.get_recent_errors()
        assert len(errors) == 1
        assert errors[0]["message"] == "Something went wrong"
        assert errors[0]["tags"]["level"] == "warning"

    def test_error_stats(self):
        """Test error statistics."""
        from qbitalabs.monitoring import ErrorTracker

        tracker = ErrorTracker("test-service")

        # Capture various errors
        for _ in range(3):
            try:
                raise ValueError("test")
            except Exception as e:
                tracker.capture_exception(e)

        for _ in range(2):
            try:
                raise KeyError("test")
            except Exception as e:
                tracker.capture_exception(e)

        stats = tracker.get_error_stats()

        assert stats["total_errors"] == 5
        assert stats["error_counts_by_type"]["ValueError"] == 3
        assert stats["error_counts_by_type"]["KeyError"] == 2


class TestProfiler:
    """Tests for performance profiling."""

    def test_profile_decorator(self):
        """Test profiling decorator."""
        from qbitalabs.monitoring import Profiler
        import time

        profiler = Profiler()

        @profiler.profile
        def slow_function():
            time.sleep(0.01)
            return 42

        result = slow_function()
        assert result == 42

        stats = profiler.get_stats()
        assert len(stats) == 1
        assert stats[0].function_name == "slow_function"
        assert stats[0].calls == 1
        assert stats[0].avg_time_ms >= 5

    def test_measure_context_manager(self):
        """Test measure context manager."""
        from qbitalabs.monitoring import Profiler
        import time

        profiler = Profiler()

        with profiler.measure("operation"):
            time.sleep(0.01)

        with profiler.measure("operation"):
            time.sleep(0.02)

        stats = profiler.get_stats()
        op_stats = next(s for s in stats if s.function_name == "operation")

        assert op_stats.calls == 2
        assert op_stats.min_time_ms >= 5
        assert op_stats.max_time_ms >= 10

    @pytest.mark.asyncio
    async def test_async_profiling(self):
        """Test profiling async functions."""
        from qbitalabs.monitoring import Profiler

        profiler = Profiler()

        @profiler.profile
        async def async_operation():
            await asyncio.sleep(0.01)
            return "done"

        result = await async_operation()
        assert result == "done"

        stats = profiler.get_stats()
        assert len(stats) == 1
        assert stats[0].avg_time_ms >= 5

    def test_profiler_reset(self):
        """Test profiler reset."""
        from qbitalabs.monitoring import Profiler

        profiler = Profiler()

        @profiler.profile
        def operation():
            pass

        operation()
        assert len(profiler.get_stats()) == 1

        profiler.reset()
        assert len(profiler.get_stats()) == 0


class TestGlobalInstances:
    """Tests for global metric/tracer/profiler instances."""

    def test_get_metrics(self):
        """Test global metrics collector."""
        from qbitalabs.monitoring import get_metrics

        metrics1 = get_metrics()
        metrics2 = get_metrics()

        assert metrics1 is metrics2

    def test_get_tracer(self):
        """Test global tracer."""
        from qbitalabs.monitoring import get_tracer

        tracer1 = get_tracer("test")
        tracer2 = get_tracer("test")

        assert tracer1 is tracer2

    def test_get_profiler(self):
        """Test global profiler."""
        from qbitalabs.monitoring import get_profiler

        profiler1 = get_profiler()
        profiler2 = get_profiler()

        assert profiler1 is profiler2
