# pdfclaw.parsers

PDF parsing for the rest of the project. One Protocol, one return type,
one string-keyed registry — pick an engine, get a `ParseResult`.

```python
from pdfclaw.parsers import parse

r = parse("paper.pdf", parser="docling")
r.body_text     # str — reading-order text
r.references    # list[str] — one entry per bib item
r.tables        # list[str] — markdown per detected table
r.n_pages       # int
r.metadata      # dict[str, Any]
r.parser_used   # "docling"
```

## Engine matrix

| name | install | quality | best for |
| --- | --- | --- | --- |
| `pymupdf` | `pip install citeclaw[pdf]` (always available) | OK text, weak on multi-column / tables / references | tests, CI, fast paths |
| `docling` | `pip install citeclaw[docling]` (single dep, CPU works, GPU optional) | good text, **good tables**, section-aware references | recommended default for production runs |
| `grobid` | requires a running [GROBID](https://github.com/kermitt2/grobid) server (see `modal_grobid_server.py` at repo root) | clean text, **structured references** (TEI XML), weak tables | when reference-resolution accuracy matters most |

No silent fallbacks across engines — pick explicitly, get explicit `ParserError` on failure.
A caller that wants a fallback chain composes one itself.

## CLI flags

```bash
python -m pdfclaw parse <pdf>                      --parser docling
python -m pdfclaw fetch-doi <doi>...               --parser docling
python -m citeclaw extract-info <pdf-or-text> ...  --parser docling
python -m citeclaw fetch-pdfs <data_dir>           --parser docling
```

Engine kwargs go through `--parser-kwarg KEY=VALUE` (repeatable).
Example: `--parser-kwarg base_url=https://you--citeclaw-grobid-serve.modal.run`.

## YAML (CiteClaw runs)

```yaml
- step: ExpandByPDF
  parser: docling           # default pymupdf
  parser_kwargs:
    do_ocr: false
```

## Adding a new engine

Three pieces:

1. **A class** in `src/pdfclaw/parsers/<name>.py` implementing the `Parser` Protocol from `base.py`:

   ```python
   class MyParser:
       name = "mineru"
       def parse(self, pdf_bytes: bytes) -> ParseResult: ...
   ```

   Lazy-import any heavy library inside the method body so importing the module stays cheap.
   Wrap engine errors as `ParserError`.

2. **Register it** in `src/pdfclaw/parsers/__init__.py`:

   ```python
   from pdfclaw.parsers.mineru import MyParser as MinerUParser
   PARSER_REGISTRY["mineru"] = MinerUParser
   ```

3. **Add the install extra** to `pyproject.toml` if the new engine has heavy dependencies, mirroring the `docling` block.

That's it. Every CLI / YAML knob picks up the new name automatically because they all consult the registry.

## Files

```
parsers/
├── README.md       this file
├── __init__.py     PARSER_REGISTRY, parse(), get_parser(), list_parsers()
├── base.py         Parser Protocol, ParseResult dataclass, ParserError
├── pymupdf.py      PyMuPDFParser
├── docling.py      DoclingParser
└── grobid.py       GrobidParser (HTTP client)
```
