"""W-ModelForge client for Google's hosted Colab MCP endpoint; credentials stay local.

Copyright (c) 2026 W-World-Labs contributors. SPDX-License-Identifier: MIT
"""

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

ENDPOINT = "https://colaboratory.googleapis.com/mcp"
ACCOUNT = os.environ.get("COLAB_MCP_ACCOUNT", "")
CONFIG = Path.home() / ".config/colab-mcp"
TOKEN = CONFIG / (ACCOUNT + ".json") if ACCOUNT else CONFIG / "credentials.json"
STATE = Path(os.environ.get("COLAB_MCP_STATE", ".colab-state.json")).resolve()
SCOPES = ["openid", "https://www.googleapis.com/auth/userinfo.email",
          "https://www.googleapis.com/auth/colaboratory"]


def private_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as out:
        json.dump(value, out, indent=2)


def request(url, payload=None, headers=None, timeout=90, method=None):
    data = None if payload is None else json.dumps(payload).encode()
    h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    h.update(headers or {})
    req = urllib.request.Request(url, data, h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode()[:2000]}") from None
    if body.lstrip().startswith("data:") or body.lstrip().startswith("event:"):
        events = [json.loads(line[5:].strip()) for line in body.splitlines() if line.startswith("data:")]
        return next(item for item in reversed(events) if "result" in item or "error" in item)
    return json.loads(body) if body.strip() else {}


def credentials():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    if not ACCOUNT:
        raise RuntimeError("Set COLAB_MCP_ACCOUNT to your Google account email.")
    if not TOKEN.exists():
        raise RuntimeError("Run login first for COLAB_MCP_ACCOUNT.")
    creds = Credentials.from_authorized_user_file(TOKEN, SCOPES)
    if not creds.valid:
        creds.refresh(Request())
        private_json(TOKEN, json.loads(creds.to_json()))
    return creds


def login():
    if not ACCOUNT:
        raise RuntimeError("Set COLAB_MCP_ACCOUNT to your Google account email.")
    from google_auth_oauthlib.flow import InstalledAppFlow
    # Public installed-application client shipped by the official Google Colab CLI.
    import importlib.resources
    client = json.loads(importlib.resources.files("colab_cli").joinpath("oauth_config.json").read_text())
    flow = InstalledAppFlow.from_client_config(client, SCOPES, autogenerate_code_verifier=True)
    flow.redirect_uri = "https://sdk.cloud.google.com/applicationdefaultauthcode.html"
    url, _ = flow.authorization_url(prompt="consent", token_usage="remote", login_hint=ACCOUNT,
                                    access_type="offline")
    print(url, flush=True)
    import getpass
    code = getpass.getpass("Paste Google authorization code (hidden): ").strip()
    flow.fetch_token(code=code)
    info = request("https://openidconnect.googleapis.com/v1/userinfo",
                   headers={"Authorization": "Bearer " + flow.credentials.token})
    if info.get("email", "").lower() != ACCOUNT:
        raise RuntimeError(f"Expected {ACCOUNT}; credentials were not saved.")
    private_json(TOKEN, json.loads(flow.credentials.to_json()))
    print("Authenticated:", ACCOUNT)


def rpc(method, params=None, *, authenticated=True, timeout=90):
    headers = {"Authorization": "Bearer " + credentials().token} if authenticated else {}
    project = os.environ.get("GOOGLE_CLOUD_QUOTA_PROJECT")
    if authenticated and project:
        headers["x-goog-user-project"] = project
    reply = request(ENDPOINT, {"jsonrpc": "2.0", "id": 1, "method": method,
                              **({"params": params} if params is not None else {})}, headers, timeout)
    if "error" in reply:
        raise RuntimeError(json.dumps(reply["error"]))
    result = reply["result"]
    if result.get("isError"):
        raise RuntimeError(json.dumps(result))
    return result


def call(name, arguments):
    return rpc("tools/call", {"name": name, "arguments": arguments}, timeout=650)


def redact(value):
    if isinstance(value, dict):
        return {key: ("<redacted>" if key in ("token", "access_token", "refresh_token") else redact(val))
                for key, val in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            return value
        return json.dumps(redact(decoded), ensure_ascii=False)
    return value


def result_data(result):
    if result.get("structuredContent") is not None:
        return result["structuredContent"]
    for block in result.get("content", []):
        if block.get("type") == "text":
            try:
                return json.loads(block["text"])
            except ValueError:
                continue
    return result


def upload(runtime, local, remote):
    runtime_info = result_data(call("get_runtime", {"name": runtime}))
    connection = runtime_info["connectionInfo"]
    headers = {"X-Colab-Runtime-Proxy-Token": connection["token"]}
    base = connection["url"].rstrip("/") + "/api/contents/"
    import base64
    path = Path(remote)
    for parent in reversed(path.parents):
        if str(parent) != "/":
            try:
                request(base + str(parent).lstrip("/"), headers=headers)
            except RuntimeError as error:
                if "HTTP 404" not in str(error):
                    raise
                request(base + str(parent).lstrip("/"), {"type": "directory"}, headers, method="PUT")
    with Path(local).open("rb") as source:
        chunk = source.read(8 << 20)
        index = 1
        while chunk:
            following = source.read(8 << 20)
            payload = {"type": "file", "format": "base64", "content": base64.b64encode(chunk).decode()}
            if index > 1 or following:
                payload["chunk"] = index if following else -1
            request(base + str(path).lstrip("/"),
                    payload, headers, method="PUT")
            chunk, index = following, index + 1
    print("Uploaded:", remote)


def download(runtime, remote, local):
    info = result_data(call("get_runtime", {"name": runtime}))["connectionInfo"]
    req = urllib.request.Request(info["url"].rstrip("/") + "/files/" + remote.lstrip("/"),
                                 headers={"X-Colab-Runtime-Proxy-Token": info["token"]})
    with urllib.request.urlopen(req, timeout=90) as source, Path(local).open("wb") as out:
        import shutil
        shutil.copyfileobj(source, out)
    print("Downloaded:", local)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("login")
    sub.add_parser("tools")
    tool = sub.add_parser("call")
    tool.add_argument("name")
    tool.add_argument("arguments", nargs="?", default="{}")
    tool.add_argument("--save-state", action="store_true")
    up = sub.add_parser("upload")
    up.add_argument("runtime")
    up.add_argument("local")
    up.add_argument("remote")
    down = sub.add_parser("download")
    down.add_argument("runtime")
    down.add_argument("remote")
    down.add_argument("local")
    args = parser.parse_args()
    if args.command == "login":
        login()
    elif args.command == "tools":
        print(json.dumps(rpc("tools/list", authenticated=False), indent=2))
    elif args.command == "upload":
        upload(args.runtime, args.local, args.remote)
    elif args.command == "download":
        download(args.runtime, args.remote, args.local)
    else:
        result = call(args.name, json.loads(args.arguments))
        if args.save_state:
            private_json(STATE, result_data(result))
        print(json.dumps(redact(result_data(result)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, urllib.error.URLError) as error:
        sys.exit(str(error))
