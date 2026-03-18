"""
Thin gRPC server wrapper — delegates to smg-grpc-servicer package.

When --enable-metrics is set, a lightweight HTTP server is started on
a separate port (gRPC port + 1) to expose Prometheus /metrics.
"""

import logging

logger = logging.getLogger(__name__)


async def _start_metrics_server(host: str, port: int):
    """Start an HTTP server exposing Prometheus /metrics.

    The caller is responsible for calling ``runner.cleanup()`` on the returned
    AppRunner when shutting down.  The server begins accepting requests before
    this function returns.
    """
    from aiohttp import web
    from prometheus_client import (
        CollectorRegistry,
        generate_latest,
        multiprocess,
    )
    from prometheus_client.exposition import CONTENT_TYPE_LATEST

    async def metrics_handler(request):
        try:
            # Create a fresh registry on each request so that
            # MultiProcessCollector re-reads the .db files from
            # PROMETHEUS_MULTIPROC_DIR.  This is necessary when not using
            # make_wsgi_app/make_asgi_app, which handle this internally.
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
            data = generate_latest(registry)
            return web.Response(
                body=data,
                headers={"Content-Type": CONTENT_TYPE_LATEST},
            )
        except Exception:
            logger.exception("Failed to generate Prometheus metrics")
            return web.Response(status=500, text="Failed to generate metrics")

    app = web.Application()
    app.router.add_get("/metrics", metrics_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info("Prometheus metrics server started on http://%s:%d/metrics", host, port)
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

    metrics_runner = None
    if server_args.enable_metrics:
        from sglang.srt.observability.func_timer import enable_func_timer
        from sglang.srt.utils import set_prometheus_multiproc_dir

        # Must set PROMETHEUS_MULTIPROC_DIR before any prometheus_client import
        # -- both in this process (the metrics server imports it) and in child
        # processes (schedulers).
        set_prometheus_multiproc_dir()
        enable_func_timer()

        metrics_port = server_args.port + 1
        try:
            metrics_runner = await _start_metrics_server(server_args.host, metrics_port)
        except OSError as e:
            logger.error(
                "Failed to start metrics server on port %d: %s. "
                "Continuing without metrics.",
                metrics_port,
                e,
            )

    try:
        await _serve_grpc(server_args, model_info)
    finally:
        if metrics_runner is not None:
            try:
                await metrics_runner.cleanup()
            except Exception:
                logger.exception(
                    "Failed to cleanly shut down Prometheus metrics server"
                )
