"""Run the web app for others to use (the container's entry point), refusing to start without a spend guard.

    python -m app.serve            # listens on 0.0.0.0:$PORT (default 8000)

Every question spends API credit, so a server reachable by others must either require API_TOKEN or set both
ASK_LIMIT_PER_HOUR and ASK_LIMIT_PER_DAY. ALLOWED_HOSTS must name the public hostname.
"""
import os
import sys

from app.config import settings

DEFAULT_HOSTS = ["localhost", "127.0.0.1", "[::1]"]


def problems() -> list[str]:
    out = []
    if settings.api_token is None and not (settings.ask_limit_per_hour and settings.ask_limit_per_day):
        out.append("set API_TOKEN, or both ASK_LIMIT_PER_HOUR and ASK_LIMIT_PER_DAY, so strangers cannot spend "
                   "your API credit without limit")
    if settings.allowed_hosts == DEFAULT_HOSTS:
        out.append("set ALLOWED_HOSTS to the public hostname (requests for any other Host are refused)")
    return out


def main() -> int:
    found = problems()
    if found:
        print("Not starting:\n" + "\n".join(f"- {p}" for p in found), file=sys.stderr)
        return 1
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")),
                proxy_headers=True, forwarded_allow_ips=os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
