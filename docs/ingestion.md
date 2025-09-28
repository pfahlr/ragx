# Ingestion Pipeline

The vector database builder ingests documents from a corpus directory before
handing content to embedding and indexing backends. This task adds first-class
support for Markdown sources alongside existing PDF handling.

## Supported Formats

| Format | Description |
| --- | --- |
| `md` | Markdown with optional front-matter. |
| `pdf` | Portable Document Format (requires `pypdf` for extraction). |

Use the CLI to select formats:

```bash
python -m ragcore.cli build --corpus-dir ./corpus --accept-format md --out ./index
```

If `--accept-format` is omitted, both `pdf` and `md` are scanned.

## Markdown Front-Matter

Markdown files may begin with a block of metadata located **above** a line that
contains only `---`. The parser treats the block as YAML, so rich structures are
allowed. A minimal example:

```markdown
title: Sample Playbook
slug: sample-playbook
summary: Preferred description for search results
tags:
  - onboarding
  - automation
---
# Sample Playbook

Document body starts here.
```

Parsing rules:

1. The block is interpreted with `yaml.safe_load`. Plain `key: value` headers
   are valid because they are also valid YAML mappings.
2. Front-matter keys override derived metadata (for example, `title` derived
   from the first `#` heading). The derived title is still exposed as
   `derived_title` for diagnostics.
3. Body text excludes the metadata block and leading blank lines, making it
   ready for chunking and embedding.

If the block is absent, the parser falls back to the first Markdown heading.
When no headings are present the file stem is used as the title.

## Directory Scanner

The scanner walks `--corpus-dir` recursively, normalising records into:

* `text`: cleaned document contents.
* `metadata`: base keys (`source_path`, `source_relpath`, `source_format`) plus
  any front-matter entries.

The resulting records feed the builder CLI, which writes a `docmap.json`
artifact describing ingested documents. This docmap is the contract consumed by
downstream embedding and indexing stages.
