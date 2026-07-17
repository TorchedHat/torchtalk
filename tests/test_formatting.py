"""Tests for response formatters."""

from torchtalk.formatting import (
    CompactText,
    Markdown,
    coverage_note,
    create_formatter,
    relative_path,
    set_formatter_mode,
)


class _FakeExtractor:
    def __init__(self, cov):
        self._cov = cov

    def coverage_summary(self):
        return self._cov


class TestCompactText:
    def test_h2_uses_brackets(self):
        md = CompactText()
        md.h2("Title")
        assert md.build() == "[Title]"

    def test_h3_plain_text(self):
        md = CompactText()
        md.h3("Subtitle")
        assert md.build() == "Subtitle"

    def test_blank_deduplicates(self):
        md = CompactText()
        md.text("a").blank().blank().blank().text("b")
        assert md.build() == "a\n\nb"

    def test_table_skips_headers(self):
        md = CompactText()
        md.table(["A", "B"], [["1", "2"], ["3", "4"]])
        result = md.build()
        assert "A" not in result
        assert "1  2" in result

    def test_item_indentation(self):
        md = CompactText()
        md.item("top").item("nested", indent=1)
        result = md.build()
        assert "- top" in result
        assert "  - nested" in result


class TestMarkdown:
    def test_h2_uses_hashes(self):
        md = Markdown()
        md.h2("Title")
        assert "## Title" in md.build()

    def test_table_has_pipes(self):
        md = Markdown()
        md.table(["A", "B"], [["1", "2"]])
        result = md.build()
        assert "| A | B |" in result
        assert "| 1 | 2 |" in result

    def test_bold(self):
        md = Markdown()
        md.bold("Label", "value")
        assert "**Label:** value" in md.build()

    def test_codeblock(self):
        md = Markdown()
        md.codeblock("print('hi')", "python")
        result = md.build()
        assert "```python" in result
        assert "print('hi')" in result


class TestFormatterFactory:
    def test_default_is_compact(self):
        f = create_formatter()
        assert isinstance(f, CompactText)

    def test_switch_to_markdown(self):
        set_formatter_mode("markdown")
        f = create_formatter()
        assert isinstance(f, Markdown)
        set_formatter_mode("compact")

    def test_invalid_mode_raises(self):
        import pytest

        with pytest.raises(ValueError):
            set_formatter_mode("invalid")


class TestCoverageNote:
    def test_none_extractor_returns_empty(self):
        assert coverage_note(None) == ""

    def test_empty_coverage_returns_empty(self):
        assert coverage_note(_FakeExtractor({})) == ""

    def test_ok_only_omits_unindexed(self):
        assert (
            coverage_note(_FakeExtractor({"ok": 100}))
            == "Coverage: 100 of 100 C++ TUs indexed."
        )

    def test_unsupported_language_appears(self):
        note = coverage_note(_FakeExtractor({"ok": 1216, "unsupported_language": 5147}))
        assert "unindexed: 5,147 unsupported" in note
        assert "1,216 of 6,363" in note

    def test_cu_cc_composition_appears(self):
        cov = {
            "ok": 1200,
            "unsupported_language": 400,
            "parse_failed": 60,
            "cu_unindexed": 397,
            "cc_unindexed": 50,
        }
        note = coverage_note(_FakeExtractor(cov))
        assert "(incl. 397 .cu, 50 .cc)" in note
        assert "1,200 of 1,660" in note

    def test_parse_failed_appears(self):
        note = coverage_note(_FakeExtractor({"ok": 1200, "parse_failed": 260}))
        assert "260 parse-failed" in note

    def test_all_buckets_combine(self):
        cov = {
            "ok": 1216,
            "unsupported_language": 5147,
            "parse_failed": 200,
            "filtered": 1320,
        }
        note = coverage_note(_FakeExtractor(cov))
        assert "1,216 of 7,883" in note
        assert "5,147 unsupported" in note
        assert "200 parse-failed" in note

    def test_filtered_counted_in_total(self):
        note = coverage_note(_FakeExtractor({"ok": 10, "filtered": 5}))
        assert "10 of 15" in note


class TestRelativePath:
    def test_strips_base_prefix(self):
        assert relative_path("/src/pytorch/aten/x.cpp", "/src/pytorch") == "aten/x.cpp"

    def test_strips_pytorch_fallback_without_base(self):
        assert relative_path("pytorch/aten/x.cpp") == "aten/x.cpp"

    def test_strips_checkout_name_for_other_repos(self):
        assert relative_path("vllm/csrc/ops.cpp", "/scratch/vllm") == "csrc/ops.cpp"

    def test_unrelated_path_unchanged(self):
        assert relative_path("other/x.cpp", "/src/pytorch") == "other/x.cpp"
