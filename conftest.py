"""Shared setup for the tests of every package and application: one
PyTorch thread and no GPU per test process when pytest-xdist runs the tests
in parallel (ci/tests.py), the `on_the_cpu` fixture, whether Docker runs
on this machine and a Debian-based image the sandbox tests run in, how two
passes over the same tokens may differ in routing, a store in the test's
folder, and a local OpenTelemetry backend. Test files reach them as
fixtures (they cannot import this file under --import-mode=importlib)."""

import http.server
import os
import shutil
import subprocess
import threading

import pytest

from wpt_store import open_store

NO_GPU = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}  # the settings that hide every GPU from a program

# In a pytest-xdist worker, PyTorch and the programs it starts use 1 thread
# and no GPU, as in CI: every worker taking every core makes the machine
# unusable, and on a GPU that shares the machine's memory every worker
# holds its own share of it. The threading libraries read these variables
# when torch is first imported, and apps/wpt/tests/conftest.py imports it before
# pytest_configure runs, so they are set here, when pytest loads this file;
# a serial run keeps the defaults and the GPU.
if "PYTEST_XDIST_WORKER" in os.environ:
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
    os.environ.update(NO_GPU)


def pytest_configure(config):
    """In a pytest-xdist worker, also limits torch's own thread pool to 1,
    which applies even when torch was imported before this hook."""
    if "PYTEST_XDIST_WORKER" not in os.environ:
        return
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(1)


@pytest.fixture
def on_the_cpu(monkeypatch):
    """Every program the test starts sees no GPU, in a serial run too, where
    this machine's GPU would otherwise be theirs (docs/architecture.md, "Checks")."""
    for name, value in NO_GPU.items():
        monkeypatch.setenv(name, value)


@pytest.fixture(scope="session")
def docker_runs():
    """Skips the test, with the reason, when Docker is not installed or its
    daemon does not answer."""
    docker = shutil.which("docker")
    if docker is None or subprocess.run([docker, "info"], capture_output=True).returncode:
        pytest.skip("Docker is not installed or its daemon does not answer")


IMAGE = "ubuntu:24.04"  # bash, coreutils' timeout and dpkg, as task images have


@pytest.fixture(scope="session")
def image(docker_runs):
    """IMAGE, pulled when this machine does not have it."""
    if subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode:
        subprocess.run(["docker", "pull", "-q", IMAGE], capture_output=True, check=True)
    return IMAGE


@pytest.fixture(scope="session")
def routing():
    """How two passes over the same tokens may differ in routing (docs/training.md,
    "Comparing routing in tests"): ROUNDING, how far the passes may put a router
    score (a sigmoid in (0, 1), in FP32), the FP32 rounding of every layer before
    the router added up; ARITHMETIC, what a deciding comparison's own arithmetic
    may round, 4 units of FP32 rounding; first_difference(model, text, routes)."""
    import torch
    from wpt_model.experts import Experts, Routes

    class Routing:
        ROUNDING = 2.0 ** -14
        ARITHMETIC = 4 * torch.finfo(torch.float32).eps

        @classmethod
        def first_difference(cls, model, text, routes):
            """Reads text (token ids of one sequence) whole in model, recording its
            routes, and compares them with routes (a Routes that covers the
            text, such as a cache's): the first position where the routed
            experts differ, None when none does. A position may differ only at
            a near tie, where the whole pass's k-th and (k+1)-th biased scores
            lie no farther apart than two scores each moved by ROUNDING, plus
            ARITHMETIC; the caller compares outputs only before that position."""
            gaps, hooks = {}, []
            for experts in (m for m in model.modules() if isinstance(m, Experts) and not m.hashed):
                def gap(_, __, logits, experts=experts):
                    top = torch.topk(torch.sigmoid(logits.float()) + experts.bias, experts.k + 1).values
                    gaps[experts] = top[:, experts.k - 1] - top[:, experts.k]
                hooks.append(experts.router.register_forward_hook(gap))
            recorded = Routes()
            try:
                with torch.no_grad():
                    model(torch.tensor([text]), record=recorded)
            finally:
                for hook in hooks:
                    hook.remove()
            first = None
            for experts, chosen in recorded.chosen.items():
                known = routes.replay(experts, len(text))
                differ = (chosen.sort(-1).values != known.sort(-1).values).any(-1)
                assert (gaps[experts][differ] <= 2 * cls.ROUNDING + cls.ARITHMETIC).all(), \
                    "a token whose routing is clear went to other experts"
                if differ.any():
                    position = int(differ.nonzero()[0])
                    first = position if first is None else min(first, position)
            return first

    return Routing


@pytest.fixture
def store(tmp_path):
    return open_store((tmp_path / "store").as_uri())


class OtlpBackend:
    """A local OpenTelemetry backend: an HTTP server that keeps what it is
    sent over OTLP (the protocol), decoded."""

    def __init__(self):
        backend = self
        self.requests = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                backend.requests.append((self.path, self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.send_header("Content-Type", "application/x-protobuf")
                self.end_headers()

            def log_message(self, *_):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.endpoint = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _received(self, path, kind):
        for sent, body in list(self.requests):
            if sent == path:
                request = kind()
                request.ParseFromString(body)
                yield request

    def spans(self) -> list:
        """(the resource's attributes, the span) of each span received."""
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

        return [(_attributes(group.resource), span) for request in self._received("/v1/traces", ExportTraceServiceRequest)
                for group in request.resource_spans for scope in group.scope_spans for span in scope.spans]

    def logs(self) -> list:
        """(the resource's attributes, the log record) of each one received."""
        from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest

        return [(_attributes(group.resource), log) for request in self._received("/v1/logs", ExportLogsServiceRequest)
                for group in request.resource_logs for scope in group.scope_logs for log in scope.log_records]

    def gauges(self) -> dict:
        """{a gauge's name: its last value received}."""
        from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest

        return {metric.name: metric.gauge.data_points[-1].as_double
                for request in self._received("/v1/metrics", ExportMetricsServiceRequest)
                for group in request.resource_metrics for scope in group.scope_metrics for metric in scope.metrics}


def _attributes(resource) -> dict:
    return {attribute.key: attribute.value.string_value for attribute in resource.attributes}


@pytest.fixture
def otlp_backend():
    backend = OtlpBackend()
    yield backend
    backend.server.shutdown()
