# The public depot web app's image (compose.yaml beside it): the repository's
# code, its Python libraries and the browser interface, built here so the
# container never builds anything. The build context is the repository's root,
# and Dockerfile.dockerignore beside this file names what enters it: never a
# .env file. The design system's submodule must be checked out
# (`git submodule update --init`).
# The images are pinned by digest: a new version is a change to their
# lines, and their licenses are checked again (docs/licenses.md).
FROM oven/bun:1.4.4-debian@sha256:93d5032fe06d163427822dc1598f420363759bb18155c9cc90eb1be05a64a19e AS bun

FROM python:3.14-slim-trixie@sha256:35caba83af2fe6da9a5d922f9b0156f21d1246a48278c58a08bd253303ba42e8 AS build
COPY --from=bun /usr/local/bin/bun /usr/local/bin/bun
# the design system's build runs bunx, which is the same program under that name
RUN ln -s bun /usr/local/bin/bunx
# rustup and a C toolchain build the tokenizer from source, with the Rust version its rust-toolchain.toml names
RUN apt-get update \
    && apt-get install -y --no-install-recommends rustup build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.13.5
WORKDIR /app
COPY . .
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
RUN uv sync --frozen --no-default-groups --package wpt
# the interface, built by the same code that serves it; then what only the build reads
RUN .venv/bin/python -c "from wpt.depot.web import ui; ui.built()" \
    && find . \( -name node_modules -o -name target \) -prune -exec rm -rf {} +

FROM python:3.14-slim-trixie@sha256:35caba83af2fe6da9a5d922f9b0156f21d1246a48278c58a08bd253303ba42e8
COPY --from=build /app /app
WORKDIR /app
ENV PATH=/app/.venv/bin:$PATH
# Bun is not in this image: the build fails here if the interface counts as stale,
# which would make the server try to build it when it starts
RUN python -c "from wpt.depot.web import ui; ui.built()"
CMD ["wpt", "depot", "web", "--port", "8780"]
