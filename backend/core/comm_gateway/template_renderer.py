"""Dynamic payload templating.

Placeholders look like ``{slot_id}``.  Rendering is JSON aware:

* inside a JSON string literal (``"ST_{slot_id}"``) the value is inserted as
  escaped string content, so quotes/newlines in a camera name cannot break
  the document;
* outside a string (``"occupied": {is_occupied}``) the value is emitted as a
  native JSON token (``true``/``false``/number/``null``/quoted string).

Only names from :data:`TEMPLATE_VARIABLE_NAMES` (plus any extra context key)
are substituted; JSON object braces such as ``{"a": 1}`` are never matched
because a placeholder must be ``{`` immediately followed by an identifier.
"""

import json
import re
from typing import Any, Dict, Iterable, Optional

from .constants import TEMPLATE_VARIABLE_NAMES

_IDENT = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


class TemplateError(ValueError):
    """Raised when a template cannot produce a valid payload."""


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _json_token(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(str(value), ensure_ascii=False)


def find_placeholders(template: str, known: Optional[Iterable[str]] = None) -> list:
    names = set(known or TEMPLATE_VARIABLE_NAMES)
    return [m.group(1) for m in _IDENT.finditer(template or "") if m.group(1) in names]


def _substitute_json(template: str, context: Dict[str, Any], names: set) -> str:
    out = []
    i, n = 0, len(template)
    in_string = False
    while i < n:
        ch = template[i]
        if ch == "{":
            match = _IDENT.match(template, i)
            if match and match.group(1) in names:
                value = context.get(match.group(1))
                if in_string:
                    out.append(json.dumps(_text_value(value), ensure_ascii=False)[1:-1])
                else:
                    out.append(_json_token(value))
                i = match.end()
                continue
        if in_string and ch == "\\" and i + 1 < n:
            out.append(template[i:i + 2])
            i += 2
            continue
        if ch == '"':
            in_string = not in_string
        out.append(ch)
        i += 1
    return "".join(out)


def _json_error(rendered: str, exc: json.JSONDecodeError) -> TemplateError:
    lines = rendered.splitlines()
    line = lines[exc.lineno - 1] if 0 < exc.lineno <= len(lines) else ""
    return TemplateError(f"JSON không hợp lệ (dòng {exc.lineno}, cột {exc.colno}): {exc.msg}. Đoạn lỗi: {line.strip()[:80]}")


def render_template(template: str, context: Dict[str, Any], payload_type: str = "json") -> str:
    """Render ``template`` with ``context``.

    For ``payload_type == "json"`` the result is validated with ``json.loads``
    and re-serialised on a single line (required by line-framed transports).
    Raises :class:`TemplateError` with an operator-readable message.
    """
    if template is None or not str(template).strip():
        raise TemplateError("Template rỗng.")
    names = set(TEMPLATE_VARIABLE_NAMES) | set(context.keys())
    if payload_type == "text":
        return _IDENT.sub(lambda m: _text_value(context.get(m.group(1))) if m.group(1) in names else m.group(0), template)
    rendered = _substitute_json(template, context, names)
    try:
        document = json.loads(rendered)
    except json.JSONDecodeError as exc:
        raise _json_error(rendered, exc) from None
    return json.dumps(document, ensure_ascii=False)


def validate_template(template: str, payload_type: str = "json") -> Optional[str]:
    """Return an error message, or ``None`` when the template renders cleanly."""
    from .context import sample_context

    try:
        render_template(template, sample_context(), payload_type)
    except TemplateError as exc:
        return str(exc)
    return None
