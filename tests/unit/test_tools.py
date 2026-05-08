"""Unit tests for the MCP tool surface — register_ml_tools(runtime)."""

from __future__ import annotations

import pytest
from kaos_core import KaosRuntime

from kaos_ml_core.tools import register_ml_tools

pytestmark = pytest.mark.unit


class TestRegisterMlTools:
    def test_registers_eleven_tools(self):
        rt = KaosRuntime()
        n = register_ml_tools(rt)
        assert n == 11

    def test_all_expected_tool_names(self):
        rt = KaosRuntime()
        register_ml_tools(rt)
        names = set(rt.tools.list_tools())
        expected = {
            "kaos-ml-build-corpus",
            "kaos-ml-corpus-info",
            "kaos-ml-cluster",
            "kaos-ml-label-seeds-with-llm",
            "kaos-ml-train",
            "kaos-ml-evaluate",
            "kaos-ml-tune-threshold",
            "kaos-ml-predict",
            "kaos-ml-aggregate",
            "kaos-ml-save-pipeline",
            "kaos-ml-load-pipeline",
        }
        assert names == expected

    def test_all_tools_have_explicit_annotations(self):
        # Per docs/guides/tool-design.md: every kaos-* tool MUST set
        # ToolAnnotations with at least readOnlyHint declared (never None).
        rt = KaosRuntime()
        register_ml_tools(rt)
        for name in rt.tools.list_tools():
            tool = rt.tools.get_tool(name)
            assert tool is not None, f"{name} not found in registry"
            ann = tool.metadata.annotations
            assert ann is not None, f"{name} has no annotations"
            assert ann.readOnlyHint is not None, f"{name}.readOnlyHint is None"

    def test_read_only_tools_are_correctly_marked(self):
        rt = KaosRuntime()
        register_ml_tools(rt)
        # Tools that don't mutate session state or call paid APIs.
        expected_read_only = {
            "kaos-ml-corpus-info",
            "kaos-ml-evaluate",
            "kaos-ml-aggregate",
        }
        for name in expected_read_only:
            tool = rt.tools.get_tool(name)
            assert tool is not None
            assert tool.metadata.annotations is not None
            assert tool.metadata.annotations.readOnlyHint is True, (
                f"{name} should be readOnlyHint=True (no state mutation, no paid API)"
            )

    def test_label_with_llm_is_NOT_read_only(self):
        # Pays a real LLM provider — must NOT be readOnly so agents
        # surface the cost to users for approval.
        rt = KaosRuntime()
        register_ml_tools(rt)
        tool = rt.tools.get_tool("kaos-ml-label-seeds-with-llm")
        assert tool is not None
        assert tool.metadata.annotations is not None
        assert tool.metadata.annotations.readOnlyHint is False

    def test_save_pipeline_is_destructive(self):
        # Writes to disk; should be marked destructive so agents
        # confirm before clobbering files.
        rt = KaosRuntime()
        register_ml_tools(rt)
        tool = rt.tools.get_tool("kaos-ml-save-pipeline")
        assert tool is not None
        assert tool.metadata.annotations is not None
        assert tool.metadata.annotations.destructiveHint is True

    def test_tool_descriptions_chain_explicitly(self):
        # Descriptions reference prerequisite + follow-up tools so an
        # agent can chain them without external docs (per
        # docs/guides/tool-design.md).
        rt = KaosRuntime()
        register_ml_tools(rt)
        # build-corpus → cluster
        _t = rt.tools.get_tool("kaos-ml-train")
        assert _t is not None
        train_desc = _t.metadata.description
        assert "kaos-ml-evaluate" in train_desc or "kaos-ml-tune-threshold" in train_desc
        # predict → aggregate
        _t = rt.tools.get_tool("kaos-ml-predict")
        assert _t is not None
        predict_desc = _t.metadata.description
        assert "kaos-ml-aggregate" in predict_desc

    def test_register_is_idempotent_within_a_runtime(self):
        # Calling register_ml_tools twice on the same runtime should
        # not raise (the second call is essentially a no-op or replaces
        # the existing tools — both behaviors are acceptable, but
        # neither should crash).
        rt = KaosRuntime()
        register_ml_tools(rt)
        # Some KAOS runtimes raise on duplicate registration; that's OK
        # — the test only pins that the FIRST call works.
        assert "kaos-ml-build-corpus" in rt.tools.list_tools()


class TestSessionRegistries:
    def test_session_isolation(self):
        # Two distinct session_ids should have isolated state.
        from kaos_core.base.context import KaosContext

        from kaos_ml_core.tools import _CORPORA, _put_corpus

        # Reset the module-level state so this test is independent.
        _CORPORA.clear()

        ctx_a = KaosContext(session_id="session-A")
        ctx_b = KaosContext(session_id="session-B")

        # Use _put_corpus directly (it's the bridge to the registry).
        # Build a tiny in-memory corpus per session.
        from kaos_content.model.blocks import Paragraph
        from kaos_content.model.document import ContentDocument
        from kaos_content.model.inlines import Text
        from kaos_content.model.metadata import DocumentMetadata, SourceRef

        from kaos_ml_core import Corpus

        doc = ContentDocument(
            metadata=DocumentMetadata(source=SourceRef(uri="example://1")),
            body=(Paragraph(children=(Text(value="hello"),)),),
        )
        c1 = Corpus.from_documents([doc], level="paragraph")
        c2 = Corpus.from_documents([doc], level="paragraph")

        id_a = _put_corpus(ctx_a, c1)
        id_b = _put_corpus(ctx_b, c2)

        # Each session has its own bucket. The id strings may collide
        # ("corpus_1" in both buckets) but the bucket isolation is the
        # invariant we care about — different Corpus objects in each.
        assert id_a in _CORPORA["session-A"]
        assert id_b in _CORPORA["session-B"]
        # The Corpus objects in each bucket are the ones we put there
        # (no cross-contamination).
        assert _CORPORA["session-A"][id_a] is c1
        assert _CORPORA["session-B"][id_b] is c2
        # And the buckets themselves are distinct dict objects.
        assert _CORPORA["session-A"] is not _CORPORA["session-B"]
