"""Curated venue-name presets for common journal families.

Designed to be used from YAML configs via the ``VenuePreset`` Route
predicate::

    - if: {VenuePreset: ["nature", "science", "cell"]}
      pass_to: ...
    - if: {VenuePreset: ["preprint"]}
      pass_to: ...

Matching is normalized (lowercase + whitespace-collapsed) exact match.
Exact match is deliberate: for comprehensive family lists a substring
match would false-positive on venues like "Nature-Inspired Computing"
from the ``nature`` preset.

Add new presets by inserting a constant list and registering it in
``VENUE_PRESETS``.
"""

from __future__ import annotations

NATURE: list[str] = [
    "Nature",
    "Nature Africa",
    "Nature Aging",
    "Nature Astronomy",
    "Nature Biomedical Engineering",
    "Nature Biotechnology",
    "Nature Cancer",
    "Nature Cardiovascular Research",
    "Nature Catalysis",
    "Nature Cell Biology",
    "Nature Chemical Biology",
    "Nature Chemical Engineering",
    "Nature Chemistry",
    "Nature Cities",
    "Nature Climate Change",
    "Nature Communications",
    "Nature Computational Science",
    "Nature Digest",
    "Nature Ecology & Evolution",
    "Nature Electronics",
    "Nature Energy",
    "Nature Food",
    "Nature Genetics",
    "Nature Geoscience",
    "Nature Health",
    "Nature Human Behaviour",
    "Nature Immunology",
    "Nature India",
    "Nature Italy",
    "Nature Machine Intelligence",
    "Nature Materials",
    "Nature Mechanical Engineering",
    "Nature Medicine",
    "Nature Mental Health",
    "Nature Metabolism",
    "Nature Methods",
    "Nature Microbiology",
    "Nature Nanotechnology",
    "Nature Neuroscience",
    "Nature Photonics",
    "Nature Physics",
    "Nature Plants",
    "Nature Progress Brain Health",
    "Nature Progress Oncology",
    "Nature Protocols",
    "Nature Reviews Biodiversity",
    "Nature Reviews Bioengineering",
    "Nature Reviews Cancer",
    "Nature Reviews Cardiology",
    "Nature Reviews Chemistry",
    "Nature Reviews Clean Technology",
    "Nature Reviews Clinical Oncology",
    "Nature Reviews Computing",
    "Nature Reviews Disease Primers",
    "Nature Reviews Drug Discovery",
    "Nature Reviews Earth & Environment",
    "Nature Reviews Electrical Engineering",
    "Nature Reviews Endocrinology",
    "Nature Reviews Gastroenterology & Hepatology",
    "Nature Reviews Genetics",
    "Nature Reviews Immunology",
    "Nature Reviews Materials",
    "Nature Reviews Methods Primers",
    "Nature Reviews Microbiology",
    "Nature Reviews Molecular Cell Biology",
    "Nature Reviews Nephrology",
    "Nature Reviews Neurology",
    "Nature Reviews Neuroscience",
    "Nature Reviews Physics",
    "Nature Reviews Psychology",
    "Nature Reviews Rheumatology",
    "Nature Reviews Urology",
    "Nature Sensors",
    "Nature Structural & Molecular Biology",
    "Nature Sustainability",
    "Nature Synthesis",
    "Nature Water",
]

SCIENCE: list[str] = [
    "Science",
    "Science Advances",
    "Science Immunology",
    "Science Robotics",
    "Science Signaling",
    "Science Translational Medicine",
]

CELL: list[str] = [
    "Cell",
    "Cancer Cell",
    "Cell Biomaterials",
    "Cell Chemical Biology",
    "Cell Genomics",
    "Cell Host & Microbe",
    "Cell Metabolism",
    "Cell Press Blue",
    "Cell Reports",
    "Cell Reports Medicine",
    "Cell Reports Methods",
    "Cell Reports Physical Science",
    "Cell Reports Sustainability",
    "Cell Stem Cell",
    "Cell Systems",
]

# Common preprint servers. S2 ``venue`` strings vary in formatting
# (e.g. "arXiv" vs "arXiv.org"), so multiple variants per server are
# listed here.
PREPRINT: list[str] = [
    "arXiv",
    "arXiv.org",
    "bioRxiv",
    "medRxiv",
    "ChemRxiv",
    "SSRN",
    "TechRxiv",
    "Research Square",
    "Preprints",
    "Preprints.org",
    "OSF Preprints",
]

VENUE_PRESETS: dict[str, list[str]] = {
    "nature": NATURE,
    "science": SCIENCE,
    "cell": CELL,
    "preprint": PREPRINT,
}


def resolve_presets(names: list[str]) -> list[str]:
    """Expand preset names to a deduplicated list of venue strings.

    Raises ``ValueError`` on an unknown preset name.
    """
    resolved: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = name.strip().lower()
        if key not in VENUE_PRESETS:
            known = sorted(VENUE_PRESETS)
            raise ValueError(
                f"Unknown venue preset {name!r}. Known presets: {known}"
            )
        for v in VENUE_PRESETS[key]:
            if v not in seen:
                seen.add(v)
                resolved.append(v)
    return resolved
