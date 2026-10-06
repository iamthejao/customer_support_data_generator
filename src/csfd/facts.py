"""Seeded, structured facts about one case, shared by both speakers.

Every case (allocation slot) gets one :class:`CaseFacts` record before any
dialogue is generated: the customer's machine (a model and a serial number
from the company seed's ``## Case facts`` catalogue), the customer site
(company, city, country, drawn with Faker) and the caller's job role. The
customer is told what a customer knows, the agent what the CRM shows, and the
consistency check compares the finished dialogue against the record (see
:func:`identifier_mismatches`), so nobody has to make identifiers up.
"""

from __future__ import annotations

import re
import string
from collections.abc import Iterable
from typing import Any

from faker import Faker
from pydantic import BaseModel

from csfd.seeds.company import AssetModel, CaseFactsCatalogue
from csfd.utils.rng import derive_rng

DEFAULT_SITE_LOCALE = "en_US"

# A model identifier shaped like "CF-600": letters, optional separator, digits.
_MODEL_FAMILY = re.compile(r"([A-Za-z]+)[-\s]?(\d+)[A-Za-z]*")


class CaseFacts(BaseModel):
    """The ground-truth facts of one case, fixed before the conversation starts."""

    asset_model: str | None = None
    asset_description: str | None = None
    asset_serial: str | None = None
    customer_company: str
    site_city: str
    site_country: str
    caller_role: str | None = None


def named_asset(catalogue: CaseFactsCatalogue, problem_text: str) -> AssetModel | None:
    """The first catalogue model the problem record names, if any.

    Uses the same matching as :func:`identifier_mismatches`, so "CF600" or
    "CF 600" in the problem picks the CF-600 the check will then expect.
    """
    by_name = {_normalise(a.model): a for a in catalogue.assets}
    mentions = [m for p in _model_patterns(catalogue) for m in p.finditer(problem_text)]
    for m in sorted(mentions, key=lambda m: m.start()):
        if (asset := by_name.get(_normalise(m.group()))) is not None:
            return asset
    return None


def draw_case_facts(
    catalogue: CaseFactsCatalogue,
    *,
    seed: int,
    slot_index: int,
    problem_text: str = "",
    asset_model: str | None = None,
) -> CaseFacts:
    """Draw one case's facts deterministically from ``(seed, slot_index)``.

    ``asset_model`` (the model a problem is about, when known otherwise), or a
    model the problem record names, wins, so the facts never contradict the
    problem; otherwise the model is a seeded pick from the catalogue.
    """
    rng = derive_rng(seed, f"case:{slot_index}:facts")
    locale = rng.choice(catalogue.site_locales or [DEFAULT_SITE_LOCALE])
    fake = Faker(locale)
    fake.seed_instance(rng.getrandbits(64))
    picked = rng.choice(catalogue.assets) if catalogue.assets else None
    known = next((a for a in catalogue.assets if a.model == asset_model), None)
    asset = known or named_asset(catalogue, problem_text) or picked
    serial = (
        fake.bothify(asset.serial_format, letters=string.ascii_uppercase)
        if asset is not None and asset.serial_format
        else None
    )
    return CaseFacts(
        asset_model=asset.model if asset else None,
        asset_description=(asset.description or None) if asset else None,
        asset_serial=serial,
        customer_company=fake.company(),
        site_city=fake.city(),
        site_country=fake.current_country(),
        caller_role=rng.choice(catalogue.caller_roles) if catalogue.caller_roles else None,
    )


def customer_view(facts: CaseFacts) -> dict[str, Any]:
    """What the customer knows: who they are, where, and what the nameplate says."""
    return {
        "role": facts.caller_role,
        "company": facts.customer_company,
        "site": f"{facts.site_city}, {facts.site_country}",
        "asset_model": facts.asset_model,
        "asset_serial": facts.asset_serial,
    }


def crm_view(facts: CaseFacts) -> dict[str, Any]:
    """What the agent's CRM shows for this customer: account, contact and installed asset."""
    return {
        "account": facts.customer_company,
        "site": f"{facts.site_city}, {facts.site_country}",
        "contact_role": facts.caller_role,
        "asset_model": facts.asset_model,
        "asset_description": facts.asset_description,
        "asset_serial": facts.asset_serial,
    }


def _normalise(identifier: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", identifier.upper())


def format_pattern(fmt: str) -> re.Pattern[str]:
    """A whole-token pattern for identifiers shaped like ``fmt`` (``#`` digit, ``?`` letter).

    Separators (``-`` or a space) are optional, so ``CF600-1234-AB`` also
    matches ``CF600 1234AB``. Used for serial numbers here and for the
    identifier registry's part numbers and error codes.
    """
    parts = []
    for ch in fmt:
        if ch == "#":
            parts.append(r"\d")
        elif ch == "?":
            parts.append("[A-Za-z]")
        elif ch in "- ":
            parts.append(r"[-\s]?")
        else:
            parts.append(re.escape(ch))
    return re.compile(rf"(?<![A-Za-z0-9]){''.join(parts)}(?![A-Za-z0-9])", re.I)


def _model_patterns(catalogue: CaseFactsCatalogue) -> list[re.Pattern[str]]:
    """One pattern per model family ("CF" + digits) plus the literal non-family names.

    A family token needs at least as many digits as the family's shortest
    catalogue model, so component tags such as "CF1" or "CF 15" are not read
    as machines. Trailing letters count only when upper-case, so "CF-600s"
    reads as the CF-600 while an invented "CF-6600X" is caught whole.
    """
    min_digits: dict[str, int] = {}
    patterns: list[re.Pattern[str]] = []
    for asset in catalogue.assets:
        family = _MODEL_FAMILY.fullmatch(asset.model)
        if family is None:
            patterns.append(
                re.compile(rf"(?<![A-Za-z0-9]){re.escape(asset.model)}(?![A-Za-z0-9])", re.I)
            )
            continue
        prefix, digits = family.group(1).upper(), len(family.group(2))
        min_digits[prefix] = min(digits, min_digits.get(prefix, digits))
    for prefix, digits in min_digits.items():
        patterns.append(
            re.compile(
                rf"(?<![A-Za-z0-9])(?i:{re.escape(prefix)})[-\s]?\d{{{digits},}}[A-Z]*(?![0-9])"
            )
        )
    return patterns


def identifier_mismatches(
    texts: Iterable[str], facts: CaseFacts, catalogue: CaseFactsCatalogue
) -> list[str]:
    """Model and serial numbers in ``texts`` that differ from the case record.

    Only identifiers shaped like the company catalogue's are recognised (see
    :func:`_model_patterns`), so a miss is possible. Returns one short
    message per distinct wrong identifier (empty when everything matches).
    """
    if facts.asset_model is None:
        return []
    text = "\n".join(texts)
    issues: list[str] = []
    seen: set[str] = set()
    # Serials first, then masked out, so a serial's model-like prefix is not
    # read again as a model name.
    for fmt in dict.fromkeys(a.serial_format for a in catalogue.assets if a.serial_format):
        pattern = format_pattern(fmt)
        for m in pattern.finditer(text):
            found = _normalise(m.group())
            if found in seen or (facts.asset_serial and found == _normalise(facts.asset_serial)):
                continue
            seen.add(found)
            issues.append(
                f"The dialogue gives serial number {m.group()!r}, but the case record's "
                f"serial number is {facts.asset_serial or 'not set'}."
            )
        text = pattern.sub(" ", text)
    expected = _normalise(facts.asset_model)
    for pattern in _model_patterns(catalogue):
        for m in pattern.finditer(text):
            found = _normalise(m.group())
            if found == expected or found in seen:
                continue
            seen.add(found)
            issues.append(
                f"The dialogue names the machine {m.group()!r}, but the case record's "
                f"machine is the {facts.asset_model}."
            )
    return issues
