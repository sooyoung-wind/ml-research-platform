"""GROBID TEI XML parser for the ML Research Platform.

Parses TEI XML output produced by GROBID into structured data that can be
merged into the unified Paper model.  Uses defusedxml for safe XML
processing and degrades gracefully on malformed input (partial extraction).
"""

from __future__ import annotations

import logging
from typing import Any

import defusedxml.ElementTree as DefusedET
from pydantic import BaseModel, Field

from ml_platform.models import Author, Paper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TEI namespace helpers
# ---------------------------------------------------------------------------

TEI_NS = "http://www.tei-c.org/ns/1.0"
XML_NS = "http://www.w3.org/XML/1998/namespace"

# ElementTree tag format: {namespace}localname
def _tei(local: str) -> str:
    return f"{{{TEI_NS}}}{local}"

def _xml(local: str) -> str:
    return f"{{{XML_NS}}}{local}"


# ---------------------------------------------------------------------------
# Pydantic result models
# ---------------------------------------------------------------------------

class TeiSection(BaseModel):
    """A single section of the parsed paper body."""

    heading: str | None = None
    number: str | None = None
    paragraphs: list[str] = Field(default_factory=list)
    subsections: list[TeiSection] = Field(default_factory=list)

    @property
    def text(self) -> str:
        """Flattened text of paragraphs (no subsection text included)."""
        return "\n\n".join(self.paragraphs)


class TeiReference(BaseModel):
    """A bibliographic reference extracted from the paper."""

    ref_id: str | None = None
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    journal: str | None = None
    volume: str | None = None
    pages: str | None = None
    publisher: str | None = None


class TeiFigure(BaseModel):
    """A figure or table description."""

    fig_id: str | None = None
    fig_type: str = "figure"  # "figure" or "table"
    label: str | None = None
    caption: str | None = None
    description: str | None = None
    file_name: str | None = None


class TeiParseResult(BaseModel):
    """Complete result of parsing a GROBID TEI XML document."""

    title: str | None = None
    abstract: str | None = None
    authors: list[Author] = Field(default_factory=list)
    sections: list[TeiSection] = Field(default_factory=list)
    references: list[TeiReference] = Field(default_factory=list)
    figures: list[TeiFigure] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    doi: str | None = None
    publication_date: str | None = None

    # Metadata about the parse itself
    parse_errors: list[str] = Field(default_factory=list)
    partial: bool = False


# ---------------------------------------------------------------------------
# Internal helper text extraction
# ---------------------------------------------------------------------------

def _get_text(element: Any | None) -> str | None:
    """Extract all text content from an element and its children."""
    if element is None:
        return None
    # itertext() yields all text nodes in document order
    parts = "".join(element.itertext()).strip()
    return parts if parts else None


def _get_child_text(parent: Any, tag: str) -> str | None:
    """Get text of a direct child element, or None."""
    child = parent.find(tag)
    return _get_text(child) if child is not None else None


# ---------------------------------------------------------------------------
# Section extraction (recursive)
# ---------------------------------------------------------------------------

def _parse_div(div_elem: Any) -> TeiSection:
    """Parse a <div> element into a TeiSection, recursing into child divs."""
    heading_el = div_elem.find(_tei("head"))
    heading = _get_text(heading_el)
    number = heading_el.get("n") if heading_el is not None else None

    paragraphs: list[str] = []
    subsections: list[TeiSection] = []

    for child in div_elem:
        tag = child.tag
        if tag == _tei("p"):
            text = _get_text(child)
            if text:
                paragraphs.append(text)
        elif tag == _tei("div"):
            subsections.append(_parse_div(child))
        # skip <head> — already handled above

    return TeiSection(
        heading=heading,
        number=number,
        paragraphs=paragraphs,
        subsections=subsections,
    )


# ---------------------------------------------------------------------------
# Author extraction
# ---------------------------------------------------------------------------

def _parse_author(author_el: Any) -> Author:
    """Parse a TEI <author> element into an Author model."""
    # Name parts
    pers_name = author_el.find(_tei("persName"))
    if pers_name is not None:
        forename_el = pers_name.find(_tei("forename"))
        surname_el = pers_name.find(_tei("surname"))
        forename = _get_text(forename_el) or ""
        surname = _get_text(surname_el) or ""
        name = f"{forename} {surname}".strip()
    else:
        # Fallback: try <name> or raw text
        name_el = author_el.find(_tei("name"))
        if name_el is not None:
            name = _get_text(name_el) or ""
        else:
            name = ""

    # Affiliation — take the first <affiliation>
    affiliation: str | None = None
    aff_el = author_el.find(_tei("affiliation"))
    if aff_el is not None:
        org_parts: list[str] = []
        for org_name in aff_el.findall(_tei("orgName")):
            t = _get_text(org_name)
            if t:
                org_parts.append(t)
        addr = aff_el.find(_tei("address"))
        if addr is not None:
            settlement = _get_child_text(addr, _tei("settlement"))
            country = _get_child_text(addr, _tei("country"))
            if settlement:
                org_parts.append(settlement)
            if country:
                org_parts.append(country)
        affiliation = ", ".join(org_parts) if org_parts else None

    # Semantic Scholar / other IDs from <idno>
    semantic_scholar_id: str | None = None
    for idno_el in author_el.findall(_tei("idno")):
        idno_type = idno_el.get("type", "")
        idno_text = (_get_text(idno_el) or "").strip()
        if idno_type.lower() in ("semantic_scholar", "s2") and idno_text:
            semantic_scholar_id = idno_text
            break  # take first match

    return Author(
        name=name,
        affiliation=affiliation,
        semantic_scholar_id=semantic_scholar_id,
    )


# ---------------------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------------------

def _parse_bibl_struct(bibl: Any) -> TeiReference | None:
    """Parse a <biblStruct> element into a TeiReference."""
    ref_id = bibl.get(_xml("id")) or bibl.get("id")

    # Title — <title level="a"> for article, <title> without level for fallback
    title: str | None = None
    for title_el in bibl.findall(_tei("title")):
        if title_el.get("level") == "a" or title is None:
            t = _get_text(title_el)
            if t:
                title = t
                if title_el.get("level") == "a":
                    break  # article title takes priority

    # Journal / book title (level="j" or level="m")
    journal: str | None = None
    for title_el in bibl.findall(_tei("title")):
        lvl = title_el.get("level")
        if lvl in ("j", "m"):
            journal = _get_text(title_el)
            break

    # Authors
    authors: list[str] = []
    for author_el in bibl.findall(_tei("author")):
        pers = author_el.find(_tei("persName"))
        if pers is not None:
            forename = _get_child_text(pers, _tei("forename")) or ""
            surname = _get_child_text(pers, _tei("surname")) or ""
            auth_name = f"{forename} {surname}".strip()
        else:
            auth_name = _get_child_text(author_el, _tei("name")) or ""
        if auth_name:
            authors.append(auth_name)

    # Date — <date type="published">
    year: int | None = None
    for date_el in bibl.findall(_tei("date")):
        if date_el.get("type") == "published" or year is None:
            when = date_el.get("when") or date_el.get("when-custom")
            if when:
                try:
                    year = int(str(when)[:4])
                except (ValueError, TypeError):
                    pass

    # DOI
    doi: str | None = None
    for idno_el in bibl.findall(_tei("idno")):
        idno_type = (idno_el.get("type") or "").lower()
        if idno_type == "doi":
            doi = _get_text(idno_el)
            break

    # Volume / pages / publisher from <analytic> or <monogr>
    volume: str | None = None
    pages: str | None = None
    publisher: str | None = None
    for scope_el in bibl.iter(_tei("biblScope")):
        unit = scope_el.get("unit", "")
        if unit == "volume":
            volume = _get_text(scope_el) or scope_el.get("n")
        elif unit == "page":
            from_pg = scope_el.get("from")
            to_pg = scope_el.get("to")
            if from_pg and to_pg:
                pages = f"{from_pg}–{to_pg}"
            else:
                pages = _get_text(scope_el) or scope_el.get("n")

    for pub_el in bibl.iter(_tei("publisher")):
        pub_text = _get_text(pub_el)
        if pub_text:
            publisher = pub_text
            break

    # Only return if we have at least a title or an author
    if not title and not authors:
        return None

    return TeiReference(
        ref_id=ref_id,
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        journal=journal,
        volume=volume,
        pages=pages,
        publisher=publisher,
    )


# ---------------------------------------------------------------------------
# Figure / table extraction
# ---------------------------------------------------------------------------

def _parse_figure(fig_el: Any) -> TeiFigure | None:
    """Parse a <figure> element into a TeiFigure."""
    fig_id = fig_el.get(_xml("id")) or fig_el.get("id")
    fig_type = fig_el.get("type", "figure")

    label_el = fig_el.find(_tei("label"))
    label = _get_text(label_el)

    # <figDesc> or <head> as caption
    caption_el = fig_el.find(_tei("figDesc"))
    caption = _get_text(caption_el)
    if not caption:
        head_el = fig_el.find(_tei("head"))
        caption = _get_text(head_el)

    # Description (text inside <p> children)
    desc_parts: list[str] = []
    for p_el in fig_el.findall(_tei("p")):
        t = _get_text(p_el)
        if t:
            desc_parts.append(t)
    description = "\n".join(desc_parts) if desc_parts else None

    # File reference from graphic
    file_name: str | None = None
    graphic_el = fig_el.find(_tei("graphic"))
    if graphic_el is not None:
        file_name = graphic_el.get("url") or graphic_el.get("target")

    return TeiFigure(
        fig_id=fig_id,
        fig_type=fig_type,
        label=label,
        caption=caption,
        description=description,
        file_name=file_name,
    )


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_tei_xml(xml_content: str) -> TeiParseResult:
    """Parse a GROBID TEI XML string into a TeiParseResult.

    Extracts as much data as possible.  If the XML is malformed, returns a
    partial result with parse_errors populated.
    """
    result = TeiParseResult()
    parse_errors: list[str] = []

    # --- Parse XML safely ---
    try:
        root = DefusedET.fromstring(xml_content)
    except DefusedET.ParseError as exc:
        logger.warning("Failed to parse TEI XML: %s", exc)
        result.parse_errors.append(f"XML parse error: {exc}")
        result.partial = True
        return result

    # The root should be <TEI>
    if root.tag != _tei("TEI"):
        result.parse_errors.append(
            f"Expected root tag <TEI>, got <{root.tag.split('}')[-1]}>"
        )
        result.partial = True
        # Try to continue anyway — maybe the namespace is different

    # --- Title ---
    try:
        title_stmt = root.find(f".//{_tei('titleStmt')}")
        if title_stmt is not None:
            title_el = title_stmt.find(_tei("title"))
            if title_el is not None:
                result.title = _get_text(title_el)
    except Exception as exc:
        parse_errors.append(f"Error extracting title: {exc}")

    # --- Abstract ---
    try:
        abstract_el = root.find(f".//{_tei('profileDesc')}/{_tei('abstract')}")
        if abstract_el is not None:
            paragraphs: list[str] = []
            for p_el in abstract_el.findall(_tei("p")):
                t = _get_text(p_el)
                if t:
                    paragraphs.append(t)
            if paragraphs:
                result.abstract = "\n\n".join(paragraphs)
            elif abstract_el.text and abstract_el.text.strip():
                result.abstract = abstract_el.text.strip()
    except Exception as exc:
        parse_errors.append(f"Error extracting abstract: {exc}")

    # --- Authors ---
    try:
        source_desc = root.find(f".//{_tei('sourceDesc')}")
        if source_desc is not None:
            for author_el in source_desc.findall(f".//{_tei('author')}"):
                try:
                    result.authors.append(_parse_author(author_el))
                except Exception as exc:
                    parse_errors.append(f"Error parsing one author: {exc}")
    except Exception as exc:
        parse_errors.append(f"Error extracting authors: {exc}")

    # --- Sections (body) ---
    try:
        body = root.find(f".//{_tei('body')}")
        if body is not None:
            for div_el in body.findall(_tei("div")):
                try:
                    result.sections.append(_parse_div(div_el))
                except Exception as exc:
                    parse_errors.append(f"Error parsing one body div: {exc}")
    except Exception as exc:
        parse_errors.append(f"Error extracting body: {exc}")

    # --- References ---
    try:
        back = root.find(f".//{_tei('back')}")
        if back is not None:
            for bibl_el in back.findall(f".//{_tei('biblStruct')}"):
                try:
                    ref = _parse_bibl_struct(bibl_el)
                    if ref is not None:
                        result.references.append(ref)
                except Exception as exc:
                    parse_errors.append(f"Error parsing one reference: {exc}")
    except Exception as exc:
        parse_errors.append(f"Error extracting references: {exc}")

    # --- Figures / Tables ---
    try:
        # Figures can appear inside <body> or <figure> elements
        for fig_el in root.findall(f".//{_tei('figure')}"):
            try:
                fig = _parse_figure(fig_el)
                if fig is not None:
                    result.figures.append(fig)
            except Exception as exc:
                parse_errors.append(f"Error parsing one figure: {exc}")
    except Exception as exc:
        parse_errors.append(f"Error extracting figures: {exc}")

    # --- Keywords ---
    try:
        profile_desc = root.find(f".//{_tei('profileDesc')}")
        if profile_desc is not None:
            for kw_el in profile_desc.findall(f".//{_tei('keyword')}"):
                text = _get_text(kw_el)
                if text:
                    result.keywords.append(text)
            # Also check <term> elements inside <textClass>
            text_class = profile_desc.find(_tei("textClass"))
            if text_class is not None:
                for term_el in text_class.findall(f".//{_tei('term')}"):
                    text = _get_text(term_el)
                    if text and text not in result.keywords:
                        result.keywords.append(text)
    except Exception as exc:
        parse_errors.append(f"Error extracting keywords: {exc}")

    # --- DOI ---
    try:
        # DOI can appear in <publicationStmt><idno type="doi"> or in fileDesc
        for idno_el in root.findall(f".//{_tei('idno')}"):
            idno_type = (idno_el.get("type") or "").lower()
            if idno_type == "doi":
                doi_text = _get_text(idno_el)
                if doi_text:
                    result.doi = doi_text
                    break
    except Exception as exc:
        parse_errors.append(f"Error extracting DOI: {exc}")

    # --- Publication date ---
    try:
        for date_el in root.findall(
            f".//{_tei('sourceDesc')}//{_tei('date')}"
        ):
            when = date_el.get("when") or date_el.get("when-custom")
            if when:
                result.publication_date = str(when)
                break
    except Exception as exc:
        parse_errors.append(f"Error extracting publication date: {exc}")

    # Finalise
    result.parse_errors = parse_errors
    if parse_errors:
        result.partial = True
        logger.info(
            "TEI parse completed with %d error(s), partial=%s",
            len(parse_errors),
            result.partial,
        )

    return result


# ---------------------------------------------------------------------------
# Paper model updater
# ---------------------------------------------------------------------------

def update_paper(paper: Paper, parse_result: TeiParseResult) -> Paper:
    """Merge a TeiParseResult into an existing Paper model.

    Returns a **new** Paper instance (immutable-style update via ``model_copy``).
    Fields already set on *paper* are preserved unless the parse result
    provides richer data.
    """
    updates: dict[str, Any] = {}

    # Title — prefer existing (from source metadata) if already set
    if parse_result.title and not paper.title:
        updates["title"] = parse_result.title

    # Abstract — prefer the parsed abstract (usually more complete)
    if parse_result.abstract:
        updates["abstract"] = parse_result.abstract

    # Authors — only overwrite if paper has none and parse result does
    if parse_result.authors and not paper.authors:
        updates["authors"] = parse_result.authors

    # Keywords — merge without duplicates
    merged_kw = list(paper.keywords)
    for kw in parse_result.keywords:
        if kw not in merged_kw:
            merged_kw.append(kw)
    if merged_kw != paper.keywords:
        updates["keywords"] = merged_kw

    # DOI
    if parse_result.doi and not paper.doi:
        updates["doi"] = parse_result.doi

    # Build structured parsed_content dict
    parsed_content: dict[str, Any] = {
        "sections": [s.model_dump() for s in parse_result.sections],
        "references": [r.model_dump() for r in parse_result.references],
        "figures": [f.model_dump() for f in parse_result.figures],
        "parse_errors": parse_result.parse_errors,
        "partial": parse_result.partial,
    }
    if parse_result.publication_date:
        parsed_content["publication_date"] = parse_result.publication_date

    updates["parsed_content"] = parsed_content

    # Use model_copy for an immutable-style update (Pydantic v2)
    return paper.model_copy(update=updates)
