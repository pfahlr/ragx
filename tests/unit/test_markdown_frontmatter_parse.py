import textwrap
import importlib
import pytest

MODULE_PATH = "pkgs.ingest.markdown_frontmatter"  # implement here per spec
FUNC_NAME = "parse_front_matter"

@pytest.fixture
def parse():
    try:
        mod = importlib.import_module(MODULE_PATH)
    except ModuleNotFoundError:
        pytest.skip(f"{MODULE_PATH} not implemented; create it to enable this test")
    fn = getattr(mod, FUNC_NAME, None)
    if fn is None:
        pytest.skip(f"{MODULE_PATH}.{FUNC_NAME} not implemented yet")
    return fn

def test_parses_header_block_simple(parse):
    md = textwrap.dedent(
        """\
        title: Neuroplasticity Overview
        author: Jane Doe
        tags: neuroscience,plasticity
        ---
        # Neuroplasticity
        The brain changes with experience.
        """
    )
    meta, body = parse(md)
    assert meta == {
        "title": "Neuroplasticity Overview",
        "author": "Jane Doe",
        "tags": "neuroscience,plasticity",
    }
    assert body.lstrip().startswith("# Neuroplasticity")

def test_handles_no_front_matter(parse):
    md = "# Just content\nNo header here."
    meta, body = parse(md)
    assert meta == {}
    assert body.startswith("# Just content")

def test_ignores_blank_lines_above_header(parse):
    md = "\n\nkey: value\n---\nContent"
    meta, body = parse(md)
    assert meta == {"key": "value"}
    assert body == "Content"

def test_stops_header_on_first_rule(parse):
    md = "a: 1\n---\n---\nbody"
    meta, body = parse(md)
    assert meta == {"a": "1"}
    assert body == "---\nbody"

def test_trims_values_and_keys(parse):
    md = " title :  Hello World  \n author:  A  \n---\nbody"
    meta, body = parse(md)
    assert meta == {"title": "Hello World", "author": "A"}
    assert body == "body"

def test_allows_colons_in_value(parse):
    md = "summary: Part 1: Basics\n---\nbody"
    meta, body = parse(md)
    assert meta == {"summary": "Part 1: Basics"}
    assert body == "body"

def test_keeps_markdown_intact(parse):
    md = "k: v\n---\n# H1\n* list\n\n**bold**"
    meta, body = parse(md)
    assert meta == {"k": "v"}
    assert "# H1" in body and "**bold**" in body
