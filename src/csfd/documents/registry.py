"""The identifier registry: part numbers, error codes and document numbers per machine model.

Documents, problems and (later) conversations must name the same parts, error
codes and documents, so a run draws them once, before Phase 1, into one
:class:`ProductFacts` record per machine model of the seed's ``### Assets``
table. The sources are mixed:

* the seed's optional ``### Parts``, ``### Error codes`` and ``### Documents``
  tables (under ``## Case facts``) win wherever they are present;
* the model (agent ``registry_writer``, :class:`RegistryNamesOutput`) names
  what the seed lacks: components, error meanings and actions, and menu paths;
* code assigns every number and date the seed does not give: part numbers,
  error codes, document numbers, revisions, issue dates and firmware versions,
  each from its own ``derive_rng(run_seed, "registry:...")`` stream, like
  serial numbers (:func:`csfd.facts.draw_case_facts`).

Phase 1 sees each model's parts and error codes (:func:`phase1_view`) and cites
parts by number; :func:`unknown_identifiers` fails text that names a part number
or error code the registry does not hold, and :func:`register_parts` gives a
part that a plan needs but the registry lacks a fresh number.
"""

from __future__ import annotations

import re
import string
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta
from typing import Any, Literal

from faker import Faker
from pydantic import BaseModel, Field

from csfd.documents.ir import DocType
from csfd.facts import format_pattern
from csfd.seeds.company import CaseFactsCatalogue, CompanyProfile
from csfd.utils.rng import derive_rng

Source = Literal["seed", "generated", "mixed"]
NameKind = Literal["parts", "error_codes", "menus"]

# Per-model library documents and the type code printed in their number.
LIBRARY_DOCUMENTS: dict[DocType, str] = {
    "operating_manual": "OM",
    "service_manual": "SM",
    "datasheet": "DS",
    "parts_list": "PL",
}
# Documents are English only (decision DOC-LANGUAGE).
LANGUAGE_CODE = "EN"
# Error codes by kind; '#' is a digit.
_CODE_FORMATS = {"error": "E-##", "warning": "W-##"}
_MAX_DRAWS = 10_000


class Part(BaseModel):
    part_number: str  # "KD-60-2204"
    name: str  # "Thermocouple, type S"

    @property
    def entry(self) -> str:
        """How a diagnosis plan lists the part: number, then name."""
        return f"{self.part_number} {self.name}"


class ErrorCode(BaseModel):
    code: str  # "E-21"
    meaning: str
    action: str = ""


class DocRef(BaseModel):
    """A library document of one model: its printed number and revision history."""

    doc_number: str  # "KD-SM-CF600-EN"
    doc_type: DocType
    revisions: list[str]  # oldest first; the last is current
    issued: list[date]  # one per revision, all before the run's first contact


class ProductFacts(BaseModel):
    """The registry entry of one machine model."""

    model: str
    # Where the names came from: the seed tables, the model, or both.
    source: Source
    # Format of code-assigned part numbers ('#' digit), kept for parts registered later.
    part_number_format: str
    parts: list[Part] = Field(default_factory=list)
    error_codes: list[ErrorCode] = Field(default_factory=list)
    firmware: list[str] = Field(default_factory=list)  # oldest first; the last is current
    menus: list[str] = Field(default_factory=list)
    documents: list[DocRef] = Field(default_factory=list)


class NamedErrorCode(BaseModel):
    kind: Literal["error", "warning"] = "error"
    meaning: str
    action: str = ""


class RegistryNamesOutput(BaseModel):
    """What the ``registry_writer`` names for one machine model; code adds every number."""

    parts: list[str] = Field(default_factory=list)  # spare-part names
    error_codes: list[NamedErrorCode] = Field(default_factory=list)
    menus: list[str] = Field(default_factory=list)  # "Service > Calibration > Silver test"


def _key(text: str) -> str:
    """Names compared loosely: lower case, punctuation and extra spaces removed."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _normalise(identifier: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", identifier.upper())


def company_prefix(company_name: str) -> str:
    """Initials printed on part and document numbers: "Kalvora Dental" -> "KD"."""
    words = re.findall(r"[A-Za-z]+", company_name)
    if len(words) > 1:
        return "".join(w[0] for w in words).upper()
    return words[0][:2].upper() if words else "XX"


def _stream(seed: int, label: str) -> Faker:
    """The seeded number stream ``label``; replaying it skips what is already taken."""
    fake = Faker()
    fake.seed_instance(derive_rng(seed, label).getrandbits(64))
    return fake


def _fresh(fake: Faker, fmt: str, taken: set[str]) -> str:
    """The stream's next value of ``fmt`` that is not ``taken`` (and take it)."""
    for _ in range(_MAX_DRAWS):
        value = fake.bothify(fmt, letters=string.ascii_uppercase)
        if value not in taken:
            taken.add(value)
            return value
    raise ValueError(f"no free identifier left for format {fmt!r}")


def missing_names(catalogue: CaseFactsCatalogue, model: str) -> list[NameKind]:
    """What the seed tables do not name for ``model`` (menus never come from seeds)."""
    need: list[NameKind] = []
    if not any(p.model == model for p in catalogue.parts):
        need.append("parts")
    if not any(c.model == model for c in catalogue.error_codes):
        need.append("error_codes")
    return [*need, "menus"]


def registry_names_inputs(company: CompanyProfile, model: str) -> dict[str, Any]:
    """Prompt inputs for one model's ``registry_writer`` call."""
    catalogue = company.case_facts
    asset = next(a for a in catalogue.assets if a.model == model)
    return {
        "company_name": company.name,
        "company_overview": company.raw_markdown[:2000],
        "model": asset.model,
        "description": asset.description,
        "products": [{"model": a.model, "description": a.description} for a in catalogue.assets],
        "need": missing_names(catalogue, model),
        "seed_parts": [p.name for p in catalogue.parts if p.model == model],
        "seed_error_codes": [c.meaning for c in catalogue.error_codes if c.model == model],
    }


def _check_seed_tables(catalogue: CaseFactsCatalogue) -> None:
    models = {a.model for a in catalogue.assets}
    named = [r.model for r in catalogue.parts] + [r.model for r in catalogue.error_codes]
    named += [r.model for r in catalogue.documents]
    unknown = sorted({m for m in named if m not in models})
    if unknown:
        raise ValueError(f"seed tables name unknown model(s): {', '.join(unknown)}")
    bad_types = sorted({d.doc_type for d in catalogue.documents} - set(LIBRARY_DOCUMENTS))
    if bad_types:
        raise ValueError(
            f"seed ### Documents has unknown type(s) {', '.join(bad_types)} "
            f"(known: {', '.join(LIBRARY_DOCUMENTS)})"
        )


def _firmware(seed: int, model: str) -> list[str]:
    rng = derive_rng(seed, f"registry:{model}:firmware")
    major, minor, patch = rng.randint(1, 4), rng.randint(0, 6), rng.randint(0, 9)
    versions = [f"{major}.{minor}.{patch}"]
    for _ in range(rng.randint(1, 3)):
        if rng.random() < 0.5:
            patch += rng.randint(1, 3)
        else:
            minor, patch = minor + 1, 0
        versions.append(f"{major}.{minor}.{patch}")
    return versions


def _documents(
    catalogue: CaseFactsCatalogue,
    model: str,
    *,
    prefix: str,
    seed: int,
    calendar_start: datetime,
) -> list[DocRef]:
    seeded = {d.doc_type: d for d in catalogue.documents if d.model == model}
    refs: list[DocRef] = []
    for doc_type, code in LIBRARY_DOCUMENTS.items():
        rng = derive_rng(seed, f"registry:{model}:doc:{doc_type}")
        row = seeded.get(doc_type)
        number = (row.doc_number if row else "") or (
            f"{prefix}-{code}-{_normalise(model)}-{LANGUAGE_CODE}"
        )
        revisions = list(row.revisions) if row and row.revisions else []
        count = len(revisions) or rng.randint(1, 4)
        revisions = revisions or list(string.ascii_uppercase[:count])
        # Drawn backwards from the first contact: the current revision is the newest.
        issued = [calendar_start.date() - timedelta(days=rng.randint(30, 365))]
        for _ in range(count - 1):
            issued.insert(0, issued[0] - timedelta(days=rng.randint(120, 720)))
        refs.append(
            DocRef(
                doc_number=number,
                doc_type=doc_type,
                revisions=revisions,
                issued=issued,
            )
        )
    return refs


def build_registry(
    company: CompanyProfile,
    *,
    seed: int,
    calendar_start: datetime,
    names: Mapping[str, RegistryNamesOutput],
) -> dict[str, ProductFacts]:
    """The run's registry, one entry per catalogue model, in catalogue order.

    ``names`` holds the ``registry_writer`` output per model; only what the
    seed tables lack is taken from it (see :func:`missing_names`). Numbers are
    drawn seed rows first, then named entries, so a seed row keeps its number
    whatever the model returns.
    """
    catalogue = company.case_facts
    _check_seed_tables(catalogue)
    prefix = company_prefix(company.name)
    families = derive_rng(seed, "registry:families").sample(range(10, 100), len(catalogue.assets))
    taken_parts = {p.part_number for p in catalogue.parts if p.part_number}
    registry: dict[str, ProductFacts] = {}
    for asset, family in zip(catalogue.assets, families, strict=True):
        model = asset.model
        named = names.get(model, RegistryNamesOutput())
        need = missing_names(catalogue, model)
        part_format = f"{prefix}-{family}-####"

        part_names = [(p.name, p.part_number) for p in catalogue.parts if p.model == model]
        if "parts" in need:
            part_names = [(n.strip(), "") for n in dict.fromkeys(named.parts) if n.strip()]
        fake = _stream(seed, f"registry:{model}:parts")
        parts: list[Part] = []
        seen: set[str] = set()
        for name, number in part_names:
            if _key(name) in seen:
                continue
            seen.add(_key(name))
            parts.append(
                Part(part_number=number or _fresh(fake, part_format, taken_parts), name=name)
            )

        seeded_codes = [c for c in catalogue.error_codes if c.model == model]
        code_rows: list[tuple[str, str, str, str]] = [
            ("error", c.code, c.meaning, c.action) for c in seeded_codes
        ]
        if "error_codes" in need:
            code_rows = [
                (c.kind, "", c.meaning.strip(), c.action.strip()) for c in named.error_codes
            ]
        taken_codes = {code for _, code, _, _ in code_rows if code}
        fake = _stream(seed, f"registry:{model}:codes")
        codes: list[ErrorCode] = []
        seen = set()
        for kind, code, meaning, action in code_rows:
            if not meaning or _key(meaning) in seen:
                continue
            seen.add(_key(meaning))
            fmt = _CODE_FORMATS[kind]
            codes.append(
                ErrorCode(
                    code=code or _fresh(fake, fmt, taken_codes), meaning=meaning, action=action
                )
            )

        from_seed = bool(
            seeded_codes
            or "parts" not in need
            or any(d.model == model for d in catalogue.documents)
        )
        from_model = bool(
            named.menus or ("parts" in need and parts) or ("error_codes" in need and codes)
        )
        registry[model] = ProductFacts(
            model=model,
            source="mixed" if from_seed and from_model else "seed" if from_seed else "generated",
            part_number_format=part_format,
            parts=parts,
            error_codes=codes,
            firmware=_firmware(seed, model),
            menus=[m.strip() for m in dict.fromkeys(named.menus) if m.strip()],
            documents=_documents(
                catalogue, model, prefix=prefix, seed=seed, calendar_start=calendar_start
            ),
        )
    return registry


def phase1_view(registry: Mapping[str, ProductFacts]) -> list[dict[str, Any]]:
    """What the Phase 1 prompt sees: each model's parts and error codes."""
    return [
        {
            "model": f.model,
            "parts": [{"part_number": p.part_number, "name": p.name} for p in f.parts],
            "error_codes": [{"code": c.code, "meaning": c.meaning} for c in f.error_codes],
        }
        for f in registry.values()
    ]


def _patterns(identifiers: Iterable[str]) -> list[re.Pattern[str]]:
    """Patterns for every shape the identifiers have (digits generalised).

    An all-digit shape would also match quantities and dates, so it is left
    unchecked: a seed that numbers its parts without letters gets no backstop.
    """
    shapes = dict.fromkeys(re.sub(r"\d", "#", i) for i in identifiers)
    return [format_pattern(s) for s in shapes if re.search(r"[A-Za-z]", s)]


def _mask_serials(text: str, catalogue: CaseFactsCatalogue) -> str:
    for fmt in dict.fromkeys(a.serial_format for a in catalogue.assets if a.serial_format):
        text = format_pattern(fmt).sub(" ", text)
    return text


def unknown_identifiers(
    texts: Iterable[str],
    registry: Mapping[str, ProductFacts],
    model: str | None,
    catalogue: CaseFactsCatalogue,
) -> list[str]:
    """Part numbers and error codes in ``texts`` that ``model``'s registry entry lacks.

    Only identifiers shaped like the registry's own are recognised, so a miss
    is possible. Without a known ``model`` any model's identifiers are allowed.
    Serial numbers are masked first, so they are never read as part numbers.
    Returns one short message per distinct unknown identifier.
    """
    if not registry:
        return []
    text = _mask_serials("\n".join(texts), catalogue)
    facts = registry.get(model) if model is not None else None
    scope = [facts] if facts is not None else list(registry.values())
    where = f"the {model}" if facts is not None else "any machine"
    kinds = {
        "part number": [(p.part_number, f.model) for f in registry.values() for p in f.parts],
        "error code": [(c.code, f.model) for f in registry.values() for c in f.error_codes],
    }
    allowed = {_normalise(p.part_number) for f in scope for p in f.parts}
    allowed |= {_normalise(c.code) for f in scope for c in f.error_codes}
    issues: list[str] = []
    for kind, ids in kinds.items():
        owners = {_normalise(i): m for i, m in ids}
        seen: set[str] = set()
        for pattern in _patterns(i for i, _ in ids):
            for m in pattern.finditer(text):
                found = _normalise(m.group())
                if found in allowed or found in seen:
                    continue
                seen.add(found)
                owner = owners.get(found)
                issues.append(
                    f"{m.group()!r} is not a registry {kind} of {where}"
                    + (f" (it belongs to the {owner})." if owner else "; do not invent one.")
                )
    return issues


def register_parts(
    registry: Mapping[str, ProductFacts],
    model: str | None,
    entries: Sequence[str],
    *,
    seed: int,
) -> tuple[dict[str, ProductFacts], list[str]]:
    """Canonical part entries for one fix, registering the parts the registry lacks.

    An entry naming one of ``model``'s part numbers becomes "<number> <name>".
    An entry with no part number is a part the fix needs: it is matched to a
    registered part by name, or registered with the next free number of the
    model's stream. Any other entry (a number the registry does not hold for
    ``model``, which :func:`unknown_identifiers` reports) is kept as written.
    Without a known ``model`` the entries are returned unchanged.
    """
    out = dict(registry)
    if model is None or model not in out:
        return out, list(entries)
    facts = out[model]
    patterns = _patterns(p.part_number for f in out.values() for p in f.parts)
    taken = {p.part_number for f in out.values() for p in f.parts}
    fake = _stream(seed, f"registry:{model}:parts")
    canonical: list[str] = []
    for entry in entries:
        by_number = {_normalise(p.part_number): p for p in facts.parts}
        numbers = [_normalise(m.group()) for p in patterns for m in p.finditer(entry)]
        if numbers:
            known = next((by_number[n] for n in numbers if n in by_number), None)
            canonical.append(known.entry if known else entry)
            continue
        name = entry.strip()
        known = next((p for p in facts.parts if _key(p.name) == _key(name)), None)
        if known is None and name:
            known = Part(part_number=_fresh(fake, facts.part_number_format, taken), name=name)
            facts = facts.model_copy(
                update={
                    "parts": [*facts.parts, known],
                    "source": "mixed" if facts.source == "seed" else facts.source,
                }
            )
        canonical.append(known.entry if known else entry)
    out[model] = facts
    return out, canonical
