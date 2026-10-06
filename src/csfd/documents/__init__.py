"""Supporting documents: manuals, KB articles, parts lists and the like for a run's cases.

A document is stored once as a typed tree (:mod:`csfd.documents.ir`) and rendered
at export time to DOCX (python-docx), PDF (Typst), a Markdown and a JSON sidecar,
plus BEIR-style retrieval files (:mod:`csfd.documents.export`). Documents come
from builders (:mod:`csfd.documents.build`) run over a finished run by
``csfd documents <run_id>``. Part numbers, error codes and document numbers
come from the run's identifier registry (:mod:`csfd.documents.registry`). The
whole feature is off unless ``documents.enabled`` is true.
"""
