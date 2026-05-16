"""GROBID TEI XML parser for the ML Research Platform.

Parses TEI XML output produced by GROBID into structured data that can be
merged into the unified Paper model. Uses defusedxml for safe XML
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
    """Return a fully-qualified TEI namespace tag.

    Args:
        local: Local XML tag name.

    Returns:
        Fully-qualified tag string.
    """
    return f"{{{TEI_NS}}}{local}"

def _xml(local: str) -> str:
    """Return a fully-qualified XML namespace tag.

    Args:
        local: Local XML tag name.

    Returns:
        Fully-qualified tag string.
    """
    return f"{{{XML_NS}}}{local}"


# ---------------------------------------------------------------------------
# Pydantic result models
# ---------------------------------------------------------------------------

class TeiSection(BaseModel):
    """A single section of the parsed paper body.

    Attributes:
        heading: Section heading text.
        number: Section number (e.g. "1", "2.1").
        paragraphs: List of paragraph text strings.
        subsections: Nested child sections.
    """

    heading: str | None = None
    number: str | None = None
    paragraphs: list[str] = Field(default_factory=list)
    subsections: list[TeiSection] = Field(default_factory=list)

    @property
    def text(self) -> str:
        """Flattened text of paragraphs (no subsection text included).

        Returns:
            Joined paragraph text separated by double newlines.
        """
        return "\n\n".join(self.paragraphs)

class TeiReference(BaseModel):
    """A bibliographic reference extracted from the paper.

    Attributes:
        ref_id: Reference identifier from the TEI XML.
        title: Title of the referenced work.
        authors: List of author names.
        year: Publication year.
        doi: Digital Object Identifier.
        journal: Journal or book title.
        volume: Volume number.
        pages: Page range.
        publisher: Publisher name.
    """

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
    """A figure or table description.

    Attributes:
        fig_id: Figure identifier from the TEI XML.
        fig_type: Type of figure (``"figure"`` or ``"table"``).
        label: Figure label (e.g. "Figure 1").
        caption: Figure caption text.
        description: Description text from paragraph children.
        file_name: Referenced file name from graphic element.
    """

    fig_id: str | None = None
    fig_type: str = "figure"  # "figure" or "table"
    label: str | None = None
    caption: str | None = None
    description: str | None = None
    file_name: str | None = None


class TeiParseResult(BaseModel):
    """Complete result of parsing a GROBID TEI XML document.

    Attributes:
        title: Paper title.
        abstract: Paper abstract.
        authors: List of Author objects.
        sections: List of parsed body sections.
        references: List of bibliographic references.
        figures: List of figure/table descriptions.
        keywords: Extracted keywords.
        doi: Digital Object Identifier.
        publication_date: Publication date string.
        parse_errors: List of errors encountered during parsing.
        partial: Whether the parse result is incomplete.
    """

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
    """Extract all text content from an element and its children.

    Args:
        element: An XML element, or None.

    Returns:
        The combined text content, or None if empty.
    """
    if element is None:
        return None
    # itertext() yields all text nodes in document order
    parts = "".join(element.itertext()).strip()
    return parts if parts else None


def _get_child_text(parent: Any, tag: str) -> str | None:
    """Get text of a direct child element, or None.

    Args:
        parent: The parent XML element.
        tag: The child tag to search for.

    Returns:
        The text content of the first matching child, or None.
    """
    child = parent.find(tag)
    return _get_text(child) if child is not None else None


# ---------------------------------------------------------------------------
# Section extraction (recursive)
# ---------------------------------------------------------------------------

def _parse_div(div_elem: Any) -> TeiSection:
    """Parse a <div> element into a TeiSection, recursing into child divs.

    Args:
        div_elem: A TEI <div> XML element.

    Returns:
        A TeiSection with paragraphs and subsections populated.
    """
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
# Author extraction helpers
# ---------------------------------------------------------------------------

def _extract_author_name(author_el: Any) -> str:
    """Extract the author's name from a TEI <author> element.

    Tries <persName> first, then falls back to <name>.

    Args:
        author_el: A TEI <author> XML element.

    Returns:
        The author's name as a string, or empty string if not found.
    """
    pers_name = author_el.find(_tei("persName"))
    if pers_name is not None:
        forename = _get_text(pers_name.find(_tei("forename"))) or ""
        surname = _get_text(pers_name.find(_tei("surname"))) or ""
        return f"{forename} {surname}".strip()
    name_el = author_el.find(_tei("name"))
    if name_el is not None:
        return _get_text(name_el) or ""
    return ""


def _extract_affiliation_text(aff_el: Any) -> str | None:
    """Extract affiliation text from a TEI <affiliation> element.

    Joins orgName children and address settlement/country into a
    single comma-separated string.

    Args:
        aff_el: A TEI <affiliation> XML element.

    Returns:
        Joined affiliation string, or None if no parts found.
    """
    org_parts: list[str] = []
    for org_name in aff_el.findall(_tei("orgName")):
        t = _get_text(org_name)
        if t:
            org_parts.append(t)
    addr = aff_el.find(_tei("address"))
    if addr is not None:
        for addr_tag in (_tei("settlement"), _tei("country")):
            val = _get_child_text(addr, addr_tag)
            if val:
                org_parts.append(val)
    return ", ".join(org_parts) if org_parts else None


def _extract_s2_author_id(author_el: Any) -> str | None:
    """Extract Semantic Scholar author ID from <idno> elements.

    Args:
        author_el: A TEI <author> XML element.

    Returns:
        The Semantic Scholar author ID, or None if not found.
    """
    for idno_el in author_el.findall(_tei("idno")):
        idno_type = idno_el.get("type", "")
        idno_text = (_get_text(idno_el) or "").strip()
        if idno_type.lower() in ("semantic_scholar", "s2") and idno_text:
            return idno_text
    return None


def _parse_author(author_el: Any) -> Author:
    """Parse a TEI <author> element into an Author model.

    Args:
        author_el: A TEI <author> XML element.

    Returns:
        An Author instance with name, affiliation, and optional IDs.
    """
    name = _extract_author_name(author_el)

    affiliation: str | None = None
    aff_el = author_el.find(_tei("affiliation"))
    if aff_el is not None:
        affiliation = _extract_affiliation_text(aff_el)

    semantic_scholar_id = _extract_s2_author_id(author_el)

    return Author(
        name=name,
        affiliation=affiliation,
        semantic_scholar_id=semantic_scholar_id,
    )


# ---------------------------------------------------------------------------
# Reference extraction helpers
# ---------------------------------------------------------------------------

def _extract_bibl_title(bibl: Any) -> str | None:
    """Extract the article title from a <biblStruct> element.

    Prefers <title level="a"> (article-level); falls back to the
    first available title.

    Args:
        bibl: A TEI <biblStruct> XML element.

    Returns:
        The article title string, or None.
    """
    title: str | None = None
    for title_el in bibl.findall(_tei("title")):
        if title_el.get("level") == "a" or title is None:
            t = _get_text(title_el)
            if t:
                title = t
                if title_el.get("level") == "a":
                    break  # article title takes priority
    return title


def _extract_bibl_journal(bibl: Any) -> str | None:
    """Extract the journal or book title from a <biblStruct> element.

    Looks for <title level="j"> or <title level="m">.

    Args:
        bibl: A TEI <biblStruct> XML element.

    Returns:
        The journal/book title string, or None.
    """
    for title_el in bibl.findall(_tei("title")):
        lvl = title_el.get("level")
        if lvl in ("j", "m"):
            return _get_text(title_el)
    return None


def _extract_bibl_author_names(bibl: Any) -> list[str]:
    """Extract author name strings from a <biblStruct> element.

    Args:
        bibl: A TEI <biblStruct> XML element.

    Returns:
        List of author name strings.
    """
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
    return authors


def _extract_bibl_year(bibl: Any) -> int | None:
    """Extract the publication year from a <biblStruct> element.

    Args:
        bibl: A TEI <biblStruct> XML element.

    Returns:
        The publication year as an integer, or None.
    """
    year: int | None = None
    for date_el in bibl.findall(_tei("date")):
        if date_el.get("type") == "published" or year is None:
            when = date_el.get("when") or date_el.get("when-custom")
            if when:
                try:
                    year = int(str(when)[:4])
                except (ValueError, TypeError):
                    pass
    return year


def _extract_bibl_doi(bibl: Any) -> str | None:
    """Extract the DOI from a <biblStruct> element.

    Args:
        bibl: A TEI <biblStruct> XML element.

    Returns:
        The DOI string, or None.
    """
    for idno_el in bibl.findall(_tei("idno")):
        idno_type = (idno_el.get("type") or "").lower()
        if idno_type == "doi":
            return _get_text(idno_el)
    return None


def _extract_bibl_scope(bibl: Any) -> tuple[str | None, str | None, str | None]:
    """Extract volume, pages, and publisher from a <biblStruct> element.

    Args:
        bibl: A TEI <biblStruct> XML element.

    Returns:
        Tuple of (volume, pages, publisher), each possibly None.
    """
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
                pages = f"{from_pg}\u2013{to_pg}"
            else:
                pages = _get_text(scope_el) or scope_el.get("n")

    for pub_el in bibl.iter(_tei("publisher")):
        pub_text = _get_text(pub_el)
        if pub_text:
            publisher = pub_text
            break

    return volume, pages, publisher


def _parse_bibl_struct(bibl: Any) -> TeiReference | None:
    """Parse a <biblStruct> element into a TeiReference.

    Args:
        bibl: A TEI <biblStruct> XML element.

    Returns:
        A TeiReference, or None if the reference lacks both title and authors.
    """
    ref_id = bibl.get(_xml("id")) or bibl.get("id")
    title = _extract_bibl_title(bibl)
    journal = _extract_bibl_journal(bibl)
    authors = _extract_bibl_author_names(bibl)
    year = _extract_bibl_year(bibl)
    doi = _extract_bibl_doi(bibl)
    volume, pages, publisher = _extract_bibl_scope(bibl)

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
    """Parse a <figure> element into a TeiFigure.

    Args:
        fig_el: A TEI <figure> XML element.

    Returns:
        A TeiFigure instance, or None if parsing fails.
    """
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
# TEI section extraction helpers (for parse_tei_xml)
# ---------------------------------------------------------------------------

def _extract_title_from_xml(root: Any, errors: list[str]) -> str | None:
    """Extract paper title from TEI XML root element.

    Args:
        root: The root XML element of the TEI document.
        errors: List to append error messages to.

    Returns:
        The paper title string, or None.
    """
    try:
        title_stmt = root.find(f".//{_tei('titleStmt')}")
        if title_stmt is not None:
            title_el = title_stmt.find(_tei("title"))
            if title_el is not None:
                return _get_text(title_el)
    except Exception as exc:
        errors.append(f"Error extracting title: {exc}")
    return None


def _extract_abstract_from_xml(root: Any, errors: list[str]) -> str | None:
    """Extract abstract text from TEI XML root element.

    Args:
        root: The root XML element of the TEI document.
        errors: List to append error messages to.

    Returns:
        The abstract text, or None.
    """
    try:
        abstract_el = root.find(
            f".//{_tei('profileDesc')}/{_tei('abstract')}"
        )
        if abstract_el is not None:
            paragraphs: list[str] = []
            for p_el in abstract_el.findall(_tei("p")):
                t = _get_text(p_el)
                if t:
                    paragraphs.append(t)
            if paragraphs:
                return "\n\n".join(paragraphs)
            if abstract_el.text and abstract_el.text.strip():
                return abstract_el.text.strip()
    except Exception as exc:
        errors.append(f"Error extracting abstract: {exc}")
    return None


def _extract_authors_from_xml(root: Any, errors: list[str]) -> list[Author]:
    """Extract author list from TEI XML root element.

    Args:
        root: The root XML element of the TEI document.
        errors: List to append error messages to.

    Returns:
        List of Author objects.
    """
    authors: list[Author] = []
    try:
        source_desc = root.find(f".//{_tei('sourceDesc')}")
        if source_desc is not None:
            for author_el in source_desc.findall(f".//{_tei('author')}"):
                try:
                    authors.append(_parse_author(author_el))
                except Exception as exc:
                    errors.append(f"Error parsing one author: {exc}")
    except Exception as exc:
        errors.append(f"Error extracting authors: {exc}")
    return authors


def _extract_sections_from_xml(
    root: Any, errors: list[str],
) -> list[TeiSection]:
    """Extract body sections from TEI XML root element.

    Args:
        root: The root XML element of the TEI document.
        errors: List to append error messages to.

    Returns:
        List of TeiSection objects.
    """
    sections: list[TeiSection] = []
    try:
        body = root.find(f".//{_tei('body')}")
        if body is not None:
            for div_el in body.findall(_tei("div")):
                try:
                    sections.append(_parse_div(div_el))
                except Exception as exc:
                    errors.append(f"Error parsing one body div: {exc}")
    except Exception as exc:
        errors.append(f"Error extracting body: {exc}")
    return sections


def _extract_references_from_xml(
    root: Any, errors: list[str],
) -> list[TeiReference]:
    """Extract bibliographic references from TEI XML root element.

    Args:
        root: The root XML element of the TEI document.
        errors: List to append error messages to.

    Returns:
        List of TeiReference objects.
    """
    refs: list[TeiReference] = []
    try:
        back = root.find(f".//{_tei('back')}")
        if back is not None:
            for bibl_el in back.findall(f".//{_tei('biblStruct')}"):
                try:
                    ref = _parse_bibl_struct(bibl_el)
                    if ref is not None:
                        refs.append(ref)
                except Exception as exc:
                    errors.append(f"Error parsing one reference: {exc}")
    except Exception as exc:
        errors.append(f"Error extracting references: {exc}")
    return refs


def _extract_figures_from_xml(root: Any, errors: list[str]) -> list[TeiFigure]:
    """Extract figures and tables from TEI XML root element.

    Args:
        root: The root XML element of the TEI document.
        errors: List to append error messages to.

    Returns:
        List of TeiFigure objects.
    """
    figures: list[TeiFigure] = []
    try:
        for fig_el in root.findall(f".//{_tei('figure')}"):
            try:
                fig = _parse_figure(fig_el)
                if fig is not None:
                    figures.append(fig)
            except Exception as exc:
                errors.append(f"Error parsing one figure: {exc}")
    except Exception as exc:
        errors.append(f"Error extracting figures: {exc}")
    return figures


def _extract_keywords_from_xml(root: Any, errors: list[str]) -> list[str]:
    """Extract keywords from TEI XML root element.

    Args:
        root: The root XML element of the TEI document.
        errors: List to append error messages to.

    Returns:
        List of keyword strings.
    """
    keywords: list[str] = []
    try:
        profile_desc = root.find(f".//{_tei('profileDesc')}")
        if profile_desc is not None:
            for kw_el in profile_desc.findall(f".//{_tei('keyword')}"):
                text = _get_text(kw_el)
                if text:
                    keywords.append(text)
            # Also check <term> elements inside <textClass>
            text_class = profile_desc.find(_tei("textClass"))
            if text_class is not None:
                for term_el in text_class.findall(f".//{_tei('term')}"):
                    text = _get_text(term_el)
                    if text and text not in keywords:
                        keywords.append(text)
    except Exception as exc:
        errors.append(f"Error extracting keywords: {exc}")
    return keywords


def _extract_doi_from_xml(root: Any, errors: list[str]) -> str | None:
    """Extract DOI from TEI XML root element.

    Args:
        root: The root XML element of the TEI document.
        errors: List to append error messages to.

    Returns:
        The DOI string, or None.
    """
    try:
        for idno_el in root.findall(f".//{_tei('idno')}"):
            idno_type = (idno_el.get("type") or "").lower()
            if idno_type == "doi":
                doi_text = _get_text(idno_el)
                if doi_text:
                    return doi_text
    except Exception as exc:
        errors.append(f"Error extracting DOI: {exc}")
    return None


def _extract_pub_date_from_xml(root: Any, errors: list[str]) -> str | None:
    """Extract publication date from TEI XML root element.

    Args:
        root: The root XML element of the TEI document.
        errors: List to append error messages to.

    Returns:
        The publication date string, or None.
    """
    try:
        for date_el in root.findall(
            f".//{_tei('sourceDesc')}//{_tei('date')}"
        ):
            when = date_el.get("when") or date_el.get("when-custom")
            if when:
                return str(when)
    except Exception as exc:
        errors.append(f"Error extracting publication date: {exc}")
    return None


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_tei_xml(xml_content: str) -> TeiParseResult:
    """Parse a GROBID TEI XML string into a TeiParseResult.

    Extracts as much data as possible. If the XML is malformed, returns a
    partial result with parse_errors populated.

    Args:
        xml_content: Raw TEI XML string from GROBID.

    Returns:
        A TeiParseResult with extracted data and any parse errors.

    Raises:
        defusedxml.ParseError: If XML parsing fails (caught internally
            and returned as a partial result).
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

    # --- Extract all sections via helpers ---
    result.title = _extract_title_from_xml(root, parse_errors)
    result.abstract = _extract_abstract_from_xml(root, parse_errors)
    result.authors = _extract_authors_from_xml(root, parse_errors)
    result.sections = _extract_sections_from_xml(root, parse_errors)
    result.references = _extract_references_from_xml(root, parse_errors)
    result.figures = _extract_figures_from_xml(root, parse_errors)
    result.keywords = _extract_keywords_from_xml(root, parse_errors)
    result.doi = _extract_doi_from_xml(root, parse_errors)
    result.publication_date = _extract_pub_date_from_xml(root, parse_errors)

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
# Paper model updater helpers
# ---------------------------------------------------------------------------

def _build_parsed_content(parse_result: TeiParseResult) -> dict[str, Any]:
    """Build structured parsed_content dict from a TeiParseResult.

    Args:
        parse_result: The TEI parse result.

    Returns:
        Dictionary with sections, references, figures, and metadata.
    """
    parsed_content: dict[str, Any] = {
        "sections": [s.model_dump() for s in parse_result.sections],
        "references": [r.model_dump() for r in parse_result.references],
        "figures": [f.model_dump() for f in parse_result.figures],
        "parse_errors": parse_result.parse_errors,
        "partial": parse_result.partial,
    }
    if parse_result.publication_date:
        parsed_content["publication_date"] = parse_result.publication_date
    return parsed_content


def _collect_simple_updates(
    paper: Paper, parse_result: TeiParseResult,
) -> dict[str, Any]:
    """Collect simple field updates from a TEI parse result.

    Updates fields only when the paper lacks them (except abstract,
    which is always overwritten when present).

    Args:
        paper: The existing Paper model.
        parse_result: The TEI parse result.

    Returns:
        Dictionary of field name to new value.
    """
    updates: dict[str, Any] = {}
    for field in ("title", "authors", "doi"):
        val = getattr(parse_result, field)
        if val and not getattr(paper, field):
            updates[field] = val
    if parse_result.abstract:
        updates["abstract"] = parse_result.abstract
    return updates


def update_paper(paper: Paper, parse_result: TeiParseResult) -> Paper:
    """Merge a TeiParseResult into an existing Paper model.

    Returns a new Paper instance (immutable-style update via ``model_copy``).
    Fields already set on *paper* are preserved unless the parse result
    provides richer data.

    Args:
        paper: The existing Paper model to update.
        parse_result: The TEI parse result to merge.

    Returns:
        A new Paper instance with merged data.
    """
    updates = _collect_simple_updates(paper, parse_result)

    # Keywords — merge without duplicates
    merged_kw = list(paper.keywords)
    for kw in parse_result.keywords:
        if kw not in merged_kw:
            merged_kw.append(kw)
    if merged_kw != paper.keywords:
        updates["keywords"] = merged_kw

    updates["parsed_content"] = _build_parsed_content(parse_result)

    # Use model_copy for an immutable-style update (Pydantic v2)
    return paper.model_copy(update=updates)
