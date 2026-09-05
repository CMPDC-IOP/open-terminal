# Personal file management

This fork includes the Linux personal-file routes used by Open WebUI's **My files** page. They are registered by `open_terminal.main` directly, without patching an installed package during the image build.

## Build and test

From this repository:

```bash
uv run python -m pytest -q tests/test_personal_files.py
docker build -t open-terminal:0.12.5-local .
```

The existing Dockerfile packages `open_terminal/workspace.py` and `open_terminal/file_operations.py` along with the rest of the application. The Python package version follows upstream at 0.12.5; the local image tag distinguishes this build from upstream.

Open WebUI's Compose configuration uses this sibling repository as its terminal build context (`../open-terminal` by default). Set `OPEN_TERMINAL_SOURCE_DIR` to another checkout path if necessary. Rebuilding the image does not replace a running container; deploying the updated Compose service applies it while retaining the existing `/home` volume.

## API contract

- `/home-files` and `/home-files/content` browse and download paths relative to the authenticated terminal user's home.
- `/workspace-files` and `/workspace-files/content` retain the workspace-relative read API under that user's `w/` directory.
- `/home-files/mkdir`, `/home-files/move`, and `/home-files/upload` manage files and directories. Upload conflicts support error, replace, or keep-both; move never overwrites a destination.
- `/home-files/text` and `/home-files/save` edit UTF-8 text up to 2 MiB with content-version checks.
- `/home-files/trash` and `/home-files/restore` move items into per-user trash and restore their original paths without overwriting existing items.

These routes require the existing API-key authentication and an `X-User-Id` resolved through multi-user mode. They reject the single-user filesystem and are excluded from OpenAPI and MCP tool discovery. Browsing does not create workspace directories. Symlinks, path traversal, special files, and direct access to the internal `.webui-trash` directory are rejected or hidden.

Mutations run through a fixed, isolated helper process as the provisioned operating-system user. The helper preserves native file permissions, atomic writes, conflict checks, and per-user trash isolation. Its environment does not inherit API credentials or application configuration.

Open WebUI remains responsible for protecting paths referenced by conversations and projects because it owns those database records. This file-management protection does not restrict arbitrary shell commands. The WebUI terminal prompt also asks the model to preserve bound workspace roots; that application-specific prompt remains in WebUI.
