# ⚡ Open Terminal

A lightweight, self-hosted terminal that gives AI agents and automation tools a dedicated environment to run commands, manage files, and execute code — all through a simple API.

## Why Open Terminal?

AI assistants are great at writing code, but they need somewhere to *run* it. Open Terminal is that place — a remote shell with file management, search, and more, accessible over a simple REST API.

You can run it two ways:

- **Docker (sandboxed)** — runs in an isolated container with a full toolkit pre-installed: Python, Node.js, git, build tools, data science libraries, ffmpeg, and more. Great for giving AI agents a safe playground without touching your host system.
- **Bare metal** — install it with `pip` and run it anywhere Python runs. Commands run directly on your machine with access to your real files, your real tools, and your real environment, perfect for local development, personal automation, or giving an AI assistant full access to your actual projects.

## Getting Started

### Docker (recommended)

```bash
docker run -d --name open-terminal --restart unless-stopped -p 8000:8000 -v open-terminal:/home/user -e OPEN_TERMINAL_API_KEY=your-secret-key ghcr.io/open-webui/open-terminal
```

That's it — you're up and running at `http://localhost:8000`.

> [!TIP]
> If you don't set an API key, one is generated automatically. Grab it with `docker logs open-terminal`.

#### Image Variants

| | `latest` | `slim` | `alpine` | `openshift` |
|---|---|---|---|---|
| **Best for** | AI agent sandboxes | Production / hardened | Edge / CI / minimal footprint | OpenShift restricted SCC |
| **Size** | ~4 GB | ~430 MB | ~230 MB | ~430 MB |
| **Bundled tooling** | Node.js, gcc, ffmpeg, LibreOffice, LaTeX, Docker CLI, data science libs | git, curl, jq | git, curl, jq | git, curl, jq |
| **Install packages at runtime** | ✔ (has `sudo`) | ✘ | ✘ | ✘ |
| **Multi-user mode** | ✔ | ✘ | ✘ | ✘ |
| **Egress firewall** | ✔ | ✔ | ✔ | ✘ |

**`slim`** and **`alpine`** have the same feature set. Slim uses Debian (glibc) for broader binary compatibility; Alpine uses musl libc and is smaller, but some C-extension pip packages may need to compile from source.

```bash
docker run -d -p 8000:8000 -e OPEN_TERMINAL_API_KEY=secret ghcr.io/open-webui/open-terminal:slim
docker run -d -p 8000:8000 -e OPEN_TERMINAL_API_KEY=secret ghcr.io/open-webui/open-terminal:alpine
docker run -d -p 8000:8000 -e OPEN_TERMINAL_API_KEY=secret ghcr.io/open-webui/open-terminal:openshift
```

> [!NOTE]
> Slim and Alpine don't support `OPEN_TERMINAL_PACKAGES` / `OPEN_TERMINAL_PIP_PACKAGES` / `OPEN_TERMINAL_NPM_PACKAGES`. To add packages, extend [Dockerfile.slim](Dockerfile.slim) or [Dockerfile.alpine](Dockerfile.alpine).

> [!NOTE]
> The default `latest` image includes LibreOffice for Word, Excel, and PowerPoint to PDF conversions. The `slim`, `alpine`, and `openshift` images stay minimal; build a custom image if those variants need Office conversion tools.

> [!NOTE]
> The OpenShift image is for restricted non-root pod policies. It does not support runtime package installs, Docker socket access, the iptables egress firewall, or `OPEN_TERMINAL_MULTI_USER=true`. Build a custom image ahead of time when OpenShift users need extra tools.

#### Updating

```bash
docker pull ghcr.io/open-webui/open-terminal
docker rm -f open-terminal
```

Then re-run the `docker run` command above.

### Bare Metal

No Docker? No problem. Open Terminal is a standard Python package:

```bash
# One-liner with uvx (no install needed)
uvx open-terminal run --host 0.0.0.0 --port 8000 --api-key your-secret-key

# Or install globally with pip
pip install open-terminal
open-terminal run --host 0.0.0.0 --port 8000 --api-key your-secret-key
```

> [!CAUTION]
> On bare metal, commands run directly on your machine with your user's permissions. Use Docker if you want sandboxed execution.

#### Customizing the Docker Environment

The easiest way to add extra packages is with environment variables — no fork needed:

```bash
docker run -d --name open-terminal -p 8000:8000 \
  -e OPEN_TERMINAL_PACKAGES="cowsay figlet" \
  -e OPEN_TERMINAL_PIP_PACKAGES="httpx polars" \
  -e OPEN_TERMINAL_NPM_PACKAGES="typescript tsx" \
  ghcr.io/open-webui/open-terminal
```

| Variable | Description |
|---|---|
| `OPEN_TERMINAL_PACKAGES` | Space-separated list of **apt** packages to install at startup |
| `OPEN_TERMINAL_PIP_PACKAGES` | Space-separated list of **pip** packages to install at startup |
| `OPEN_TERMINAL_NPM_PACKAGES` | Space-separated list of **npm** packages to install globally at startup |

> [!NOTE]
> Packages are installed each time the container starts, so startup will take longer with large package lists. For heavy customization, build a custom image instead.

#### Docker Access

The image includes the Docker CLI, Compose, and Buildx. To let agents build images, run containers, etc., mount the host's Docker socket:

```bash
docker run -d --name open-terminal -p 8000:8000 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v open-terminal:/home/user \
  ghcr.io/open-webui/open-terminal
```

> [!CAUTION]
> Mounting the Docker socket gives the container **full control over the host's Docker daemon**, which is effectively root access on the host machine. Anyone with access to the terminal can pull/run arbitrary containers (including `--privileged` ones), mount host directories, access host networking, and manage all containers on the host. Only do this in fully trusted environments.

For full control, fork the repo, edit the [Dockerfile](Dockerfile), and build your own image:

```bash
docker build -t my-terminal .
docker run -d --name open-terminal -p 8000:8000 my-terminal
```


## Configuration

Open Terminal can be configured via a TOML config file, environment variables, and CLI flags. Settings are resolved in this order (highest priority wins):

1. **CLI flags** (`--host`, `--port`, `--api-key`, etc.)
2. **Environment variables** (`OPEN_TERMINAL_API_KEY`, etc.)
3. **User config** — `$XDG_CONFIG_HOME/open-terminal/config.toml` (defaults to `~/.config/open-terminal/config.toml`)
4. **System config** — `/etc/open-terminal/config.toml`
5. **Built-in defaults**

Create a config file at either location with any of these keys (all optional):

```toml
host = "0.0.0.0"
port = 8000
api_key = "sk-my-secret-key"
cors_allowed_origins = "*"
log_dir = "/var/log/open-terminal"
binary_mime_prefixes = "image,audio"
execute_timeout = 5  # seconds to wait for command output (unset by default)
file_browser_root = "home"
```

> [!TIP]
> Use the system config at `/etc/open-terminal/config.toml` to set site-wide defaults for host and port, and the user config for personal settings like the API key — this keeps the key out of `ps` / `htop`.

You can also point to a specific config file:

```bash
open-terminal run --config /path/to/my-config.toml
```

### Resource Limits

The settings above also accept environment variables using the `OPEN_TERMINAL_` prefix and uppercase key names. File helpers share a bounded pool per service process; a full queue or wait timeout returns `503` with `Retry-After: 1`. Pool sizing and cleanup timings use internal defaults. This does not limit HTTP body parsing or total request size.

Unset OpenMP, OpenBLAS, MKL, NumExpr, VECLIB and BLIS thread variables default to 1 across commands, terminals and notebooks. Explicit environment or kernelspec values take precedence. These are library defaults; hard CPU, memory and thread limits require cgroup mode.

Set `OPEN_TERMINAL_EXECUTION_MODE=cgroup` and `OPEN_TERMINAL_EXECUTION_POLICY_FILE=/path/to/policy.json` (TOML: `execution_mode`, `execution_policy_file`). The default `legacy` mode provides no compute resource hard limits or total runtime deadline.

Cgroup mode requires Linux cgroup v2 with CPU, memory, PID, Swap and `cgroup.kill` support; a root, multi-user service with one worker; and an administrator-provided, writable, root-owned dedicated subtree. Policy, application code and interpreter paths must not be writable by compute users. Startup uses the `nobody` account for an actual launch probe and fails if requirements are unmet or untracked tasks remain; it does not silently fall back to legacy mode. Task recovery after restart is outside the current scope; residual tasks must be reconciled before startup.

Example policy (size budgets for your deployment):

```json
{
  "cgroup_root": "/sys/fs/cgroup/open-terminal",
  "service": {"cpu_millis": 500, "memory_bytes": 536870912, "pids": 128},
  "helpers": {"cpu_millis": 500, "memory_bytes": 536870912, "pids": 128},
  "compute": {"cpu_millis": 2000, "memory_bytes": 2147483648, "pids": 256},
  "user": {"cpu_millis": 1000, "memory_bytes": 1073741824, "pids": 128},
  "max_tasks": 8,
  "max_user_tasks": 4,
  "max_runtime": 3600
}
```

`cpu_millis=1000` is one CPU of time. Resource values must be positive integers, memory page-aligned, and `user <= compute`. Swap is always disabled for service, helpers and compute, including every user. The dedicated subtree must have finite limits; it and its visible ancestors must accommodate `service + helpers + compute`, without ancestor `memory.oom.group=1`. User processes must not have writable cgroup controls or access to privileged services such as a Docker socket. This is resource control for trusted users, not complete hostile-tenant, disk, network or device isolation.

`max_runtime` is optional (default 3600 seconds) and must be finite and positive. Task counts must be positive and `max_user_tasks <= max_tasks`. These are the only accepted policy fields; unknown or duplicate fields are rejected. Queue bounds, queue wait, startup/cleanup timeouts and termination grace use internal defaults.

When migrating an older policy, remove `task`, each `swap_bytes`, and the former queue/timing fields (`max_queue`, `max_user_queue`, `queue_timeout`, `start_timeout`, `cleanup_timeout`, `terminate_grace`). The former `compute_threads`, `file_helper_*` and `idempotency_*` TOML/environment settings are no longer read.

Commands, PTYs and notebook kernels share one user budget across sessions, including idle kernels. Tasks have no separate resource quota; their cgroups only identify descendants for stopping and cleanup. Each active user reserves a full user budget until their last task is drained. Waiting is bounded and scheduled fairly by user; a full queue or timeout returns `429` with `Retry-After: 1`. Even `wait=0` waits for admission; disconnecting the last creation waiter cancels pending work. There is no persistent queue or separate queued-task API.

`max_runtime` starts at actual launch, includes idle time and cannot be renewed by a request. Expiry sends SIGTERM, then kills remaining descendants after the termination grace period; quota is returned only after the task cgroup is empty. API `wait` and `EXECUTE_TIMEOUT` only limit response waiting. Managed responses expose `execution` timestamps and reasons (`completed`, `cancelled`, `timed_out`, `oom`, `start_failed`, `shutdown`); stopped notebook kernels reject execution with `409`.

### Creation Request Retries

`POST /execute`, `POST /api/terminals` and `POST /notebooks` accept an optional `Idempotency-Key` header. Use the same key and creation body for retries, a new key for a new task. Keys contain 1–128 ASCII letters, digits, `.`, `_`, `:` or `-`. Scope includes the full user ID, session ID and endpoint; authentication still runs on every request. Invalid/repeated headers return `400`, different parameters with the same key return `409`, and normalized parameters over 1 MiB return `413`.

Concurrent retries share creation; one disconnected waiter does not cancel other waiters. `/execute` reuses the process ID but rereads output (`wait` and `tail` are not creation parameters); terminals and notebooks replay the initial creation response. Notebook file contents are not fingerprinted: use a new key to start a kernel from changed content.

Creation failures, including queue `429`, release their key after cleanup completes and the current waiters finish. Concurrent waiters receive the same failure; a subsequent request can retry with the same key. Cancelled creation returns `409` to attached waiters and follows the same retry rule. Unexpected errors with unconfirmed cleanup retain their key, even past the TTL, until service restart; retries replay `500` rather than create another resource. Deleted resources return `410` while their records remain. Record-capacity or retry-concurrency `429` can be retried with the same key. Records expire after an internal TTL from creation completion, but pending/running/cleaning resources remain pinned; full capacity rejects new keys rather than evicting records. Record counts and concurrent waiters remain bounded by internal limits. Deduplication is in-memory, per process, and does not survive restarts or span workers.

### Personal Files and Tests

This fork provides `/home-files` and `/workspace-files` browsing/content APIs plus home-file mkdir, move, upload, text/save, trash and restore operations. They require API-key authentication and a multi-user `X-User-Id`, and are excluded from OpenAPI/MCP discovery. Paths are relative to the user's home or `w/` workspace; symlinks, traversal and special files are rejected. Writes run as the target OS user. Uploads support conflict handling, text saves use version checks (up to 2 MiB), and restore never overwrites existing files. These protections apply to file APIs, not arbitrary shell commands.

Run regression tests with `uv run --group dev python -m pytest -q`. Disposable Docker acceptance scripts remain in [tests/](tests/): `execution_docker_bootstrap.py`, `execution_kernel_acceptance.py`, `execution_http_acceptance.py` and `compute_container_smoke.py`. The cgroup suites require a private container cgroup namespace, writable delegation and explicit `OT_EXECUTION_DISPOSABLE_TEST=1`; use Tini to reap descendants. The bootstrap checks outer limits of at most 2 CPUs, 2 GiB RAM and 512 PIDs and imposes a 180-second backstop. Test acceptance covered one worker and zero Swap; it does not enable limits in an existing production deployment.

### File Browser Root

Open Terminal reports file-browser root metadata from `GET /files/cwd` so clients can hide parent navigation above a friendly starting point.

Set `OPEN_TERMINAL_FILE_BROWSER_ROOT` to:

| Value | Behavior |
|---|---|
| `home` | Default. Report the current user's home directory as `Home` |
| `/workspace` | Report an explicit path as the visual root |
| `{{home}}/project` | Report a path under the current user's home |
| `filesystem` | Opt out and report no visual root metadata |

This is a UI hint for clients. It does not restrict terminal commands or file APIs.

### External Workspace Storage

Open Terminal uses the filesystem it is mounted on. To store workspace files
externally, mount that storage into the container:

- Single-user: `/home/user`
- Multi-user: `/home`

This can be a Docker volume, Kubernetes persistent volume, NFS/Azure Files
mount, or FUSE mount such as blobfuse, s3fs, or rclone.

### Office Previews

`GET /files/view?path=...` always returns the original file by default. Clients
that want a rendered preview can add `preview=true`; when LibreOffice is
available the server can return DOCX and PPTX previews as PDF. When rendered
preview support is not available, the same endpoint falls back to returning the
original file bytes.

## Using with Open WebUI

Open Terminal integrates with [Open WebUI](https://github.com/open-webui/open-webui), giving your AI assistants the ability to run commands, manage files, and interact with a terminal right from the AI interface. Make sure to add it under **Open Terminal** in the integrations settings, not as a tool server. Adding it as an Open Terminal connection gives you a built-in file navigation sidebar where you can browse directories, upload, download, and edit files. There are two ways to connect:

### Direct Connection

Users can connect their own Open Terminal instance from their user settings. This is useful when the terminal is running on their local machine or a network only they can reach, since requests go directly from the **browser**.

1. Go to **User Settings → Integrations → Open Terminal**
2. Add the terminal **URL** and **API key**
3. Enable the connection

### System-Level Connection (Multi-User)

Admins can configure Open Terminal connections for all their users from the admin panel. No additional services required. Multiple terminals can be set up with access controlled at the user or group level. Requests are proxied through the Open WebUI **backend**, so the terminal only needs to be reachable from the server.

1. Go to **Admin Settings → Integrations → Open Terminal**
2. Add the terminal **URL** and **API key**
3. Enable the connection

#### Built-in Multi-User Isolation

> [!CAUTION]
> Single-container multi-user mode is **not designed for production multi-user deployments**. All users share the same kernel, network, and system resources with no hard isolation boundaries between them. If one user's process misbehaves, it can affect every other user on the system. This mode exists as a lightweight convenience for small, trusted groups — not as a security model you should rely on.

For small, trusted deployments you can enable per-user isolation inside a single container:

```bash
docker run -d --name open-terminal -p 8000:8000 \
  -v open-terminal:/home \
  -e OPEN_TERMINAL_MULTI_USER=true \
  -e OPEN_TERMINAL_API_KEY=your-secret-key \
  ghcr.io/open-webui/open-terminal
```

Each user automatically gets a dedicated Linux account with its own home directory. Files, commands, and terminals are isolated between users via standard Unix permissions.

Notebook sessions belong to the `X-User-Id` and `X-Session-Id` values supplied when they are created. Use the same values to execute cells, check status, or stop a session. In multi-user mode, `X-User-Id` is required, notebook paths must stay within the user's home directory, and symbolic links are rejected. Kernels and notebook file operations run as that Linux user, with the notebook's directory as the kernel's working directory. In single-user mode, kernels and file operations use the service account and retain the existing filesystem access scope.

## API Docs

Full interactive API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs) once your instance is running.

## Star History

<a href="https://star-history.com/#open-webui/open-terminal&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=open-webui/open-terminal&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=open-webui/open-terminal&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=open-webui/open-terminal&type=Date" />
  </picture>
</a>

> [!TIP]
> **Need container-per-user isolation?** Check out **[Terminals](https://github.com/open-webui/terminals)**, which provisions and manages separate Open Terminal containers per user. For lighter deployments, built-in multi-user mode (`OPEN_TERMINAL_MULTI_USER=true`) provides per-user isolation inside a single container.

## License

MIT — see [LICENSE](LICENSE) for details.
