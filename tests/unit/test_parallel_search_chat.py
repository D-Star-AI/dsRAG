"""Scripted model tests of the public chat caller and its HTTP MCP requests."""

import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import requests

from dsrag.chat import chat
from dsrag.chat.auto_query import Queries, Query
from dsrag.chat.chat_types import ChatResponseInput
from dsrag.chat.citations import Citation, ResponseWithCitations
from dsrag.chat.parallel_search import (
    MCP_URL, PARALLEL_SEARCH_ID, USER_AGENT, ParallelSearchError, ParallelSearchSource,
)
from dsrag.database.chat_thread.basic_db import BasicChatThreadDB


URL = "https://example.org/research?a=one&b=two#results"
EXCERPT = "A source excerpt used in the answer."


class MCPFixture:
    """Respond at the requests adapter boundary, recording prepared requests."""

    def __init__(self):
        self.requests = []
        self.failure = None
        self.sse = False
        self.text_only = False
        self.results = [{"url": URL, "title": "Research", "excerpts": [EXCERPT]}]

    def send(self, request, **kwargs):
        message = json.loads(request.body)
        self.requests.append((request, message, kwargs))
        status = 200
        if message["method"] == "initialize":
            result = {"protocolVersion": "2024-11-05"}
        elif message["method"] == "notifications/initialized":
            status, result = 202, {}
        elif message["method"] == "tools/list":
            result = {"tools": [{"name": "web_search"}, {"name": "web_fetch"}]}
        else:
            if isinstance(self.failure, BaseException):
                raise self.failure
            result = {"structuredContent": {"results": self.results}, "content": []}
            if self.text_only:
                result = {"content": [{"type": "text", "text": json.dumps({"results": self.results})}]}
            if self.failure == "http":
                status = 503
            elif self.failure == "quota":
                result = {"isError": True, "content": [{"type": "text", "text": "Free quota exceeded"}]}
            elif self.failure == "rpc":
                result = None
            elif self.failure == "fetch":
                result["structuredContent"]["errors"] = [{"url": URL, "error_type": "not_found"}]
        reply = {"jsonrpc": "2.0", "id": message.get("id"), "result": result}
        if self.failure == "rpc" and message["method"] == "tools/call":
            reply = {"jsonrpc": "2.0", "id": message["id"], "error": {"code": -1}}
        response = requests.Response()
        response.status_code = status
        response.request = request
        response.headers["Mcp-Session-Id"] = "transport-only"
        response.headers["Content-Type"] = "text/event-stream" if self.sse else "application/json"
        data = json.dumps(reply)
        if self.sse:
            data = ": heartbeat\r\n\r\nevent: message\r\ndata: " + data + "\r\n\r\n"
        response._content = data.encode()
        response._content_consumed = True
        return response

    def calls(self):
        return [m["params"] for _, m, _ in self.requests if m["method"] == "tools/call"]


class TestParallelChat(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        previous_directory = os.getcwd()
        os.chdir(self.directory.name)
        self.addCleanup(os.chdir, previous_directory)
        self.db = BasicChatThreadDB()
        self.http = MCPFixture()
        self.addCleanup(patch.stopall)
        patch("requests.adapters.HTTPAdapter.send", side_effect=self.http.send).start()
        patch.dict(os.environ, {}, clear=True).start()
        patch("requests.sessions.get_netrc_auth", return_value=("ambient", "secret")).start()
        self.query_model = patch("dsrag.chat.auto_query.get_response", side_effect=self.queries).start()
        self.answer_model = patch("dsrag.chat.chat.get_response", side_effect=self.answer).start()
        # Token counting is orthogonal to retrieval; avoid downloading a tokenizer
        # while all HTTP requests are routed to this MCP fixture.
        patch("dsrag.chat.chat.count_tokens", side_effect=len).start()
        self.contexts = []

    def queries(self, **kwargs):
        content = kwargs["messages"][-1]["content"]
        kb_id = "local" if "kb_id: local" in kwargs["messages"][0]["content"] else PARALLEL_SEARCH_ID
        return Queries(queries=[Query(query=content, knowledge_base_id=kb_id)])

    def answer(self, **kwargs):
        self.contexts.append(kwargs["messages"][0]["content"])
        response = ResponseWithCitations(response="Answer from the source", citations=[
            Citation(source_index=0, page_number=None, cited_text=EXCERPT),
        ])
        return iter([response]) if kwargs.get("stream") else response

    def new_thread(self, params=None):
        return chat.create_new_chat_thread({} if params is None else params, self.db)

    def respond(self, thread, text="current research findings", sources=None, stream=False, **kwargs):
        result = chat.get_chat_thread_response(
            thread, ChatResponseInput(user_input=text, **kwargs), self.db,
            {} if sources is None else sources, stream=stream,
        )
        return list(result)[-1] if stream else result

    def test_default_caller_consumes_results_and_persists_citations(self):
        thread = self.new_thread()
        result = self.respond(thread)
        self.assertEqual(result["model_response"]["citations"][0]["doc_id"], URL)
        self.assertEqual(result["relevant_segments"][0]["document_type"], "web")
        self.assertEqual(result["relevant_segments"][0]["file_name"], "Research")
        self.assertIn(EXCERPT, self.contexts[-1])
        self.assertIn(URL, self.contexts[-1])
        self.assertNotIn("You DO NOT have access to the internet", self.contexts[-1])
        saved = self.db.get_chat_thread(thread)
        self.assertEqual(saved["interactions"][0]["model_response"]["citations"][0]["doc_id"], URL)
        for request, message, options in self.http.requests:
            self.assertEqual(request.url, MCP_URL)
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.headers["User-Agent"], USER_AGENT)
            self.assertNotIn("Authorization", request.headers)
            self.assertEqual(options["timeout"], 30)
            if message["method"] != "initialize":
                self.assertEqual(request.headers["Mcp-Session-Id"], "transport-only")
        self.assertEqual(self.http.calls()[0]["arguments"]["session_id"], thread)

    def test_search_fetch_followup_and_new_chat_identity(self):
        source = ParallelSearchSource()
        shared = {PARALLEL_SEARCH_ID: source}
        a, b = self.new_thread(), self.new_thread()
        self.respond(a, sources=shared)
        self.respond(a, text=URL, sources=shared)
        self.respond(b, sources=shared)
        self.respond(a, sources=shared)
        calls = self.http.calls()
        self.assertEqual([c["name"] for c in calls], ["web_search", "web_fetch", "web_search", "web_search"])
        self.assertEqual(calls[1]["arguments"]["urls"], [URL])
        self.assertEqual([c["arguments"]["session_id"] for c in calls], [a, a, b, a])
        self.assertNotEqual(a, b)

    def test_concurrent_threads_share_source_without_identity_leak(self):
        shared = {PARALLEL_SEARCH_ID: ParallelSearchSource()}
        threads = [self.new_thread(), self.new_thread()]
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda thread: self.respond(thread, sources=shared), threads))
        self.assertCountEqual([c["arguments"]["session_id"] for c in self.http.calls()], threads)

    def test_empty_selection_and_legacy_threads_do_not_search(self):
        for params in ({"kb_ids": []}, {"kb_ids": [], "model": "gpt-4o-mini"}):
            thread = self.new_thread(params)
            self.respond(thread)
            self.assertEqual(self.db.get_chat_thread(thread)["params"]["kb_ids"], [])
        self.assertEqual(self.http.calls(), [])
        self.query_model.assert_not_called()

    def test_configured_missing_source_never_falls_back(self):
        result = self.respond(self.new_thread({"kb_ids": ["missing"]}))
        self.assertEqual(result, {"message": "Missing knowledge bases: missing"})
        self.assertEqual(self.http.requests, [])

    def test_per_turn_disable_and_identity_override(self):
        thread = self.new_thread()
        self.respond(thread, chat_thread_params=chat._set_chat_thread_params({"kb_ids": []}))
        self.assertEqual(self.http.requests, [])
        self.respond(thread, chat_thread_params=chat._set_chat_thread_params(
            {"kb_ids": [PARALLEL_SEARCH_ID], "thread_id": "wrong"}))
        self.assertEqual(self.http.calls()[0]["arguments"]["session_id"], thread)
        self.assertEqual(self.db.get_chat_thread(thread)["params"]["kb_ids"], [PARALLEL_SEARCH_ID])

    def test_caller_supplied_source_is_preserved(self):
        class LocalSource:
            kb_metadata = {"title": "Local", "description": "Configured corpus"}
            file_system = None

            def query(self, **kwargs):
                self.arguments = kwargs
                return []

        local = LocalSource()
        self.respond(self.new_thread({"kb_ids": ["local"]}), sources={"local": local})
        self.assertEqual(local.arguments["search_queries"], ["current research findings"])
        self.assertEqual(self.http.requests, [])
        # A caller may also override the built-in source's identifier.
        self.respond(self.new_thread(), sources={PARALLEL_SEARCH_ID: local})
        self.assertEqual(self.http.requests, [])

    def test_metadata_filter_fails_before_external_requests(self):
        with self.assertRaisesRegex(ValueError, "Metadata filters"):
            self.respond(self.new_thread(), metadata_filter={"field": "doc_id", "operator": "equals", "value": "private"})
        self.assertEqual(self.http.requests, [])

    def test_streaming_uses_same_caller_and_citations(self):
        thread = self.new_thread()
        result = self.respond(thread, stream=True)
        self.assertEqual(result["model_response"]["citations"][0]["doc_id"], URL)
        self.assertEqual(result["relevant_segments"][0]["url"], URL)
        self.assertEqual(self.http.calls()[0]["arguments"]["session_id"], thread)
        self.assertEqual(self.db.get_chat_thread(thread)["interactions"][0]["model_response"]["status"], "finished")
        self.assertEqual(BasicChatThreadDB().get_chat_thread(thread)["interactions"][0]
                         ["relevant_segments"][0]["url"], URL)

    def test_sse_and_text_content_results(self):
        self.http.sse, self.http.text_only = True, True
        self.respond(self.new_thread())
        self.assertIn(EXCERPT, self.contexts[-1])

    def test_errors_and_retry_preserve_conversation(self):
        for failure, error in [("quota", ParallelSearchError), ("http", requests.HTTPError),
                               ("rpc", ParallelSearchError), ("fetch", ParallelSearchError),
                               (requests.Timeout("timeout"), requests.Timeout),
                               (KeyboardInterrupt(), KeyboardInterrupt)]:
            with self.subTest(failure=repr(failure)):
                self.http.failure = failure
                thread = self.new_thread()
                with self.assertRaises(error):
                    self.respond(thread)
                self.assertEqual(self.db.get_chat_thread(thread)["interactions"], [])
                self.http.failure = None
                self.respond(thread)
                self.assertEqual([c["arguments"]["session_id"] for c in self.http.calls()[-2:]], [thread, thread])

    def test_timeout_and_cancellation_close_transport(self):
        for error in (requests.Timeout("timeout"), KeyboardInterrupt()):
            self.http.failure = error
            with patch.object(requests.Session, "close", autospec=True) as close:
                with self.assertRaises(type(error)):
                    self.respond(self.new_thread())
                close.assert_called_once()

    def test_invalid_result_urls_cannot_become_citations(self):
        self.http.results = [
            {"url": "javascript:alert(1)", "excerpts": ["bad"]},
            {"url": "https://user:password@example.org/", "excerpts": ["bad"]},
            {"url": URL, "excerpts": [EXCERPT]},
            {"url": URL, "excerpts": ["duplicate"]},
        ]
        result = self.respond(self.new_thread())
        self.assertEqual(len(result["relevant_segments"]), 1)
        self.assertEqual(result["relevant_segments"][0]["doc_id"], URL)


if __name__ == "__main__":
    unittest.main()
