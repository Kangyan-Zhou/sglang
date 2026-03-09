"""
gRPC server wrapper — delegates to smg-grpc-servicer package.

Handles Prometheus metrics setup/teardown around the gRPC server lifecycle
when --enable-metrics is used.
"""

import logging

logger = logging.getLogger(__name__)

# Port offset for the metrics HTTP server relative to the gRPC port
_METRICS_PORT_OFFSET = 1


async def _start_metrics_server(host: str, port: int):
    """Start a lightweight HTTP server to expose Prometheus /metrics endpoint.

    This is the standard pattern for gRPC services: serve gRPC on one port,
    serve Prometheus metrics on a separate HTTP port.
    """
    from aiohttp import web
    from prometheus_client import (
        CollectorRegistry,
        multiprocess,
    )
    from prometheus_client.openmetrics.exposition import (
        CONTENT_TYPE_LATEST as OPENMETRICS_CONTENT_TYPE,
        generate_latest as openmetrics_generate_latest,
    )

    async def metrics_handler(request):
        try:
            # Create a fresh registry on each request to read the latest
            # .db files from PROMETHEUS_MULTIPROC_DIR. This is the recommended
            # pattern for prometheus multiprocess mode.
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
            data = openmetrics_generate_latest(registry)
            resp = web.Response(body=data)
            resp.content_type = "application/openmetrics-text"
            resp.headers["Content-Type"] = OPENMETRICS_CONTENT_TYPE
            return resp
        except Exception:
            logger.exception("Failed to generate Prometheus metrics")
            return web.Response(status=500, text="Failed to generate metrics")

    app = web.Application()
    app.router.add_get("/metrics", metrics_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    return runner


async def serve_grpc(server_args, model_info=None):
    """Start the standalone gRPC server with integrated scheduler."""
    try:
        from smg_grpc_servicer.sglang.server import serve_grpc as _serve_grpc
    except ImportError as e:
        raise ImportError(
            "gRPC mode requires the smg-grpc-servicer package. "
            "If not installed, run: pip install smg-grpc-servicer[sglang]. "
            "If already installed, there may be a broken import due to a "
            "version mismatch — see the chained exception above for details."
        ) from e

    # Set up Prometheus multiprocess dir before launching scheduler subprocesses,
    # so the env var is inherited and metrics are collected across all processes.
    metrics_runner = None
    if server_args.enable_metrics:
        from sglang.srt.observability.func_timer import enable_func_timer
        from sglang.srt.utils.common import set_prometheus_multiproc_dir

        set_prometheus_multiproc_dir()
        enable_func_timer()

        # Start metrics HTTP server on a side port
        metrics_port = server_args.port + _METRICS_PORT_OFFSET
        try:
            metrics_runner = await _start_metrics_server(
                server_args.host, metrics_port
            )
            logger.info(
                "Prometheus metrics HTTP server listening on "
                "http://%s:%s/metrics",
                server_args.host,
                metrics_port,
            )
        except OSError as e:
            logger.error(
                "Failed to start Prometheus metrics server on port %s: "
                "%s. gRPC server will continue without metrics.",
                metrics_port,
                e,
            )

    try:
        await _serve_grpc(server_args, model_info)
    finally:
        if metrics_runner:
            try:
                await metrics_runner.cleanup()
            except Exception:
                logger.exception("Error cleaning up metrics server")
