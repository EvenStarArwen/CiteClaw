"""Client for the self-hosted S2ORC full-text mirror.

Given an accepted :class:`~citeclaw.models.PaperRecord`, fetch its parsed
open-access full text if (and only if) the paper has an S2ORC record,
writing the body through to the shared ``paper_full_text`` cache so chat
and ``full_text``-scope screening both benefit. A paper absent from
S2ORC returns ``None`` and — deliberately — does NOT poison the cache
with an error row (that would block the separate PDF-parse path).
"""

from __future__ import annotations

from citeclaw.clients.s2orc.client import (
    S2orcClient,
    S2orcResult,
    build_s2orc_client,
    mirror_id_for,
)

__all__ = ["S2orcClient", "S2orcResult", "build_s2orc_client", "mirror_id_for"]
