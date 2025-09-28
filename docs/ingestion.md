# Ingestion Pipeline

RAGX supports corpus ingestion for the vector database builder through the
`vectordb-builder build` CLI command. The command walks the directory supplied
via `--corpus-dir`, parses each supported document type, and emits normalized
records (text + metadata) to `docmap.json` in the build output directory.

This task adds first-class support for Markdown sources alongside existing PDF handling.

## Supported Formats


| Format | Description |
| --- | --- |
| `md` | Markdown with optional front-matter. |
| `pdf` | Portable Document Format (requires `pypdf` for extraction). |


The accepted formats are controlled by the repeatable `--accept-format` flag.
By default the builder scans for both Markdown (`md`) and PDF (`pdf`) files. A
single format can be selected by supplying the flag one or more times, e.g.:

```bash
vectordb-builder build \
  --corpus-dir ./corpus \
  --out ./build \
  --accept-format md
```

## Markdown Front-Matter

Markdown ingestion is front-matter aware. If a document begins with front
matter it is parsed and merged into the record metadata. Two front-matter
flavours are recognised:

1. **YAML front-matter** using the classic `---` fenced block.

   ```markdown
   ---
   id: doc-one
   title: Document One
   tags:
     - alpha
     - beta
   ---
   # Document One
   Body text...
   ```

2. **Key-value header** lines preceding a `---` delimiter.

   ```markdown
   title: Document Two
   tags: overview, getting-started
   ---
   Document content...
   ```

Front-matter keys override any existing metadata supplied by upstream systems
(per the `markdown_front_matter_precedence` open decision). Content below the
delimiter is preserved verbatim.


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

### PDF (`.pdf`)

PDF ingestion relies on `pypdf` for text extraction. When the package is not
available the builder raises a clear error prompting installation. Extracted
document metadata is merged with the base record while preserving any fields
already populated.


## Directory Scanner

The scanner walks `--corpus-dir` recursively, normalising records into:

* `text`: cleaned document contents.
* `metadata`: base keys (`source_path`, `source_relpath`, `source_format`) plus
  any front-matter entries.

The resulting records feed the builder CLI, which writes a `docmap.json`
artifact describing ingested documents. This docmap is the contract consumed by
downstream embedding and indexing stages.

## Output

The CLI writes `docmap.json` into the directory supplied via `--out`. Each
entry contains:

```json
{
  "id": "doc-one",
  "path": "guides/doc1.md",
  "format": "md",
  "metadata": {"title": "Document One", "tags": ["alpha", "beta"]},
  "text": "# Document One\nBody text..."
}
```
Document identifiers default to the Markdown front-matter `id`. When not
present the filename stem is used and deduplicated automatically.
