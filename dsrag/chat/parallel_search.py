"""Anonymous Parallel Search MCP source for chat retrieval."""

import json
from urllib.parse import urlsplit

import requests


PARALLEL_SEARCH_ID = "__parallel_web__"
MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "dsrag (+https://github.com/D-Star-AI/dsRAG)"
PROTOCOL_VERSION = "2024-11-05"


class ParallelSearchError(RuntimeError):
    """The web source could not return usable search results."""


def _http_url(value):
    """Accept only absolute HTTP(S) URLs without embedded credentials."""
    try:
        url = urlsplit(value)
        return (url.scheme in ("http", "https") and bool(url.hostname)
                and not url.username and not url.password
                and not any(c.isspace() for c in value))
    except (TypeError, ValueError):
        return False


class ParallelSearchSource:
    """A reusable chat source; conversation identity is supplied per query.

    Uses the Streamable HTTP MCP transport with a bounded timeout per request.
    No credentials are read from the environment or user configuration. Errors
    propagate to the caller; there is no automatic retry or provider fallback.
    """

    kb_metadata = {
        "title": "Web search (Parallel)",
        "description": (
            "Search the public web for current information. For a specific page, "
            "use its exact HTTP(S) URL as the query to fetch relevant excerpts."
        ),
    }
    file_system = None

    def __init__(self, timeout=30):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout

    def _request(self, session, headers, method, params, request_id):
        message = {"jsonrpc": "2.0", "method": method, "params": params}
        if request_id is not None:
            message["id"] = request_id
        with session.post(MCP_URL, json=message, headers=headers,
                          timeout=self.timeout, allow_redirects=False,
                          stream=True) as response:
            response.raise_for_status()
            if response.is_redirect:
                raise ParallelSearchError("Parallel MCP returned a redirect")
            if request_id is None:
                return None
            if response.headers.get("Mcp-Session-Id"):
                headers["Mcp-Session-Id"] = response.headers["Mcp-Session-Id"]
            if "text/event-stream" in response.headers.get("Content-Type", ""):
                response.encoding = "utf-8"
                reply, data = {}, []
                # Stop at the matching response rather than waiting for the
                # server to close a potentially long-lived SSE connection.
                for line in response.iter_lines(decode_unicode=True):
                    if line.startswith("data:"):
                        data.append(line[5:].lstrip(" "))
                    elif not line and data:
                        event = json.loads("\n".join(data))
                        data = []
                        if isinstance(event, dict) and event.get("id") == request_id:
                            reply = event
                            break
            else:
                reply = response.json()
            if not isinstance(reply, dict) or reply.get("id") != request_id:
                raise ParallelSearchError("Invalid Parallel MCP response")
            if "error" in reply:
                raise ParallelSearchError("Parallel MCP error: " + str(reply["error"]))
            if not isinstance(reply.get("result"), dict):
                raise ParallelSearchError("Missing Parallel MCP result")
            return reply["result"]

    def _call(self, name, arguments):
        # Transport state is local to this call, never shared between chats.
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        with requests.Session() as session:
            # Avoid ambient .netrc credentials on the anonymous path. Keep proxy
            # and CA configuration available through requests' normal settings.
            session.auth = lambda request: request
            initialized = self._request(session, headers, "initialize", {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "dsrag", "version": "0.5.6"},
            }, 1)
            headers["MCP-Protocol-Version"] = initialized["protocolVersion"]
            self._request(session, headers, "notifications/initialized", {}, None)
            tools = self._request(session, headers, "tools/list", {}, 2)
            if name not in {tool["name"] for tool in tools.get("tools", [])}:
                raise ParallelSearchError("Parallel MCP does not expose " + name)
            result = self._request(session, headers, "tools/call", {
                "name": name, "arguments": arguments,
            }, 3)
            if result.get("isError"):
                raise ParallelSearchError("Parallel tool error: " + str(result.get("content")))
            payload = result.get("structuredContent")
            if payload is None:
                text = "\n".join(block["text"] for block in result.get("content", [])
                                 if block.get("type") == "text")
                payload = json.loads(text)
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                raise ParallelSearchError("Invalid Parallel tool results")
            if payload.get("errors"):
                raise ParallelSearchError("Parallel fetch error: " + str(payload["errors"]))
            return payload["results"]

    def query(self, search_queries, rse_params=None, metadata_filter=None, *,
              session_id, objective="Find information relevant to these queries"):
        """Return citation-ready excerpts; exact URL queries use web_fetch.

        session_id must be the chat thread's opaque UUID, including on retries
        and follow-up turns. Metadata filters apply to local corpora and cannot
        restrict the web; reject them before making any external request.
        """
        if metadata_filter is not None:
            raise ValueError("Metadata filters are not supported by Parallel web search")
        if not session_id or len(session_id) > 100:
            raise ValueError("A conversation session_id of at most 100 characters is required")
        queries = [query.strip() for query in search_queries if query.strip()]
        urls = [query for query in queries if _http_url(query)]
        keywords = [query for query in queries if query not in urls]
        results = []
        if keywords:
            results.extend(self._call("web_search", {
                "objective": objective, "search_queries": keywords,
                "session_id": session_id,
            }))
        if urls:
            results.extend(self._call("web_fetch", {
                "urls": urls[:20], "objective": objective[:200],
                "session_id": session_id,
            }))
        segments = []
        seen = set()
        for result in results:
            url = result.get("url")
            excerpts = result.get("excerpts", [])
            if not _http_url(url) or url in seen or not excerpts:
                continue
            seen.add(url)
            title = result.get("title") or url
            segments.append({
                "doc_id": url, "url": url, "title": title,
                "content": title + "\n" + url + "\n\n" + "\n\n".join(excerpts),
            })
        return segments
