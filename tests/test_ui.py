"""
The web UI against common web-development and accessibility guidelines (WCAG 2.2 AA and general good practice).
These are static checks of the shipped files; the interactive behaviour is covered by the browser tests.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "verbadb" / "static"
HTML = (STATIC / "index.html").read_text(encoding="utf-8")
CSS = (STATIC / "style.css").read_text(encoding="utf-8")


# ── a tiny DOM, enough to ask structural questions ────────────────────────

class Page(HTMLParser):
    VOID = {"meta", "link", "input", "br", "hr", "img"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.elements: list = []          # dicts: tag, attrs, text, parent, in_head
        self._stack: list = []
        self._in_head = False

    def handle_starttag(self, tag, attrs):
        if tag == "head":
            self._in_head = True
        el = {"tag": tag, "attrs": dict(attrs), "text": "", "parent": self._stack[-1] if self._stack else None,
              "in_head": self._in_head, "index": len(self.elements)}
        self.elements.append(el)
        if tag not in self.VOID:
            self._stack.append(el)

    def handle_endtag(self, tag):
        if tag == "head":
            self._in_head = False
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i]["tag"] == tag:
                del self._stack[i:]
                break

    def handle_data(self, data):
        for el in self._stack:
            el["text"] += data

    def find(self, tag=None, **attrs):
        return [e for e in self.elements
                if (tag is None or e["tag"] == tag) and all(e["attrs"].get(k.replace("_", "-")) == v for k, v in attrs.items())]


PAGE = Page()
PAGE.feed(HTML)
IDS = [e["attrs"]["id"] for e in PAGE.elements if "id" in e["attrs"]]


def text_of(el) -> str:
    return " ".join(el["text"].split())


# ── document ──────────────────────────────────────────────────────────────

def test_document_basics():
    assert HTML.lower().startswith("<!doctype html>")
    assert PAGE.find("html")[0]["attrs"]["lang"] == "en"
    assert "VerbaDB" in text_of(PAGE.find("title")[0])
    assert PAGE.find("meta", name="description")[0]["attrs"]["content"].strip()
    assert PAGE.find("meta", charset="utf-8")


def test_the_viewport_does_not_block_zooming():
    content = PAGE.find("meta", name="viewport")[0]["attrs"]["content"].replace(" ", "")
    assert "width=device-width" in content and "initial-scale=1" in content
    assert "maximum-scale" not in content and "user-scalable=no" not in content


def test_both_colour_schemes_are_declared_to_the_browser():
    assert PAGE.find("meta", name="color-scheme")[0]["attrs"]["content"] == "light dark"
    assert PAGE.find("meta", name="theme-color")


def test_there_is_exactly_one_h1_and_it_is_the_brand():
    (h1,) = PAGE.find("h1")
    assert text_of(h1) == "VerbaDB"


def test_landmarks():
    assert len(PAGE.find("header")) == 1 and len(PAGE.find("main")) == 1
    (aside,) = PAGE.find("aside")
    assert aside["attrs"].get("aria-label")
    assert PAGE.find("noscript"), "something must be shown when JavaScript is off"


def test_skip_link_comes_first_and_lands_on_main():
    first_focusable = next(e for e in PAGE.elements if e["tag"] in ("a", "button", "input", "textarea", "select"))
    assert first_focusable["tag"] == "a" and "skip" in first_focusable["attrs"].get("class", "")
    target = first_focusable["attrs"]["href"].lstrip("#")
    (main,) = PAGE.find("main")
    assert main["attrs"].get("id") == target and main["attrs"].get("tabindex") == "-1"


def test_the_notices_live_inside_the_header_landmark():
    (header,) = PAGE.find("header")

    def inside(el, ancestor):
        while el is not None:
            if el is ancestor:
                return True
            el = el["parent"]
        return False

    for notice_id in ("setup", "privacy"):
        assert inside(PAGE.find(id=notice_id)[0], header), f"#{notice_id} must belong to a landmark"


def test_a_visible_label_is_part_of_the_accessible_name():
    """WCAG 2.5.3 (Label in Name): voice-control users say what they see, so an aria-label must contain the visible text."""
    for el in PAGE.elements:
        label, visible = el["attrs"].get("aria-label"), text_of(el)
        if el["tag"] in ("button", "a") and label and visible:
            assert visible.lower() in label.lower(), f"<{el['tag']}> shows {visible!r} but is named {label!r}"
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    assert not re.search(r'class:\s*"pick"[^}]*aria-label', app), "the database buttons must be named by their visible text"


def test_ids_are_unique_and_every_aria_reference_resolves():
    assert len(IDS) == len(set(IDS))
    for el in PAGE.elements:
        for attr in ("aria-controls", "aria-labelledby", "aria-describedby", "for"):
            for ref in el["attrs"].get(attr, "").split():
                assert ref in IDS, f"<{el['tag']}> {attr}={ref!r} points at nothing"


# ── scripts and styles: the CSP forbids inline code, and the theme must not flash ──

def test_no_inline_script_style_or_event_handlers():
    assert all(s["attrs"].get("src") for s in PAGE.find("script")), "inline <script> blocks are blocked by the CSP"
    assert not PAGE.find("style")
    for el in PAGE.elements:
        assert "style" not in el["attrs"], f"inline style on <{el['tag']}>"
        assert not [a for a in el["attrs"] if a.startswith("on")], f"inline handler on <{el['tag']}>"


def test_the_theme_script_is_blocking_and_runs_before_styles_paint():
    scripts = PAGE.find("script")
    theme = next(s for s in scripts if s["attrs"]["src"].endswith("theme.js"))
    app = next(s for s in scripts if s["attrs"]["src"].endswith("app.js"))
    stylesheet = PAGE.find("link", rel="stylesheet")[0]
    assert theme["in_head"] and "defer" not in theme["attrs"] and "async" not in theme["attrs"]
    assert theme["index"] < stylesheet["index"], "the theme must be applied before the stylesheet is used"
    assert "defer" in app["attrs"]
    assert "verbadb.theme" in (STATIC / "theme.js").read_text(encoding="utf-8")


# ── the dark-mode switch ──────────────────────────────────────────────────

def test_the_dark_mode_control_is_a_labelled_switch():
    (switch,) = PAGE.find("button", id="theme-toggle")
    assert switch["attrs"]["role"] == "switch" and switch["attrs"]["aria-checked"] in ("true", "false")
    assert switch["attrs"]["type"] == "button"
    visible_label = next(e for e in PAGE.elements if e["parent"] is switch and "switch-text" in e["attrs"].get("class", ""))
    assert text_of(visible_label) == "Dark mode"                 # the accessible name is real, visible text
    assert PAGE.find("button", id="theme-auto")[0]["attrs"].get("aria-label")


# ── controls ──────────────────────────────────────────────────────────────

def test_every_button_declares_its_type():
    assert all(b["attrs"].get("type") in ("button", "submit") for b in PAGE.find("button"))


def test_every_form_control_has_an_accessible_name():
    labelled_for = {e["attrs"]["for"] for e in PAGE.find("label") if "for" in e["attrs"]}
    for el in PAGE.elements:
        if el["tag"] not in ("input", "textarea", "select"):
            continue
        if el["attrs"].get("type") == "hidden":
            continue
        wrapped = False
        parent = el["parent"]
        while parent is not None:
            if parent["tag"] == "label":
                wrapped = True
            parent = parent["parent"]
        named = el["attrs"].get("id") in labelled_for or wrapped or el["attrs"].get("aria-label") or el["attrs"].get("aria-labelledby")
        assert named, f"<{el['tag']} {el['attrs']}> has no label"


def test_tabs_follow_the_aria_pattern():
    (tablist,) = PAGE.find(role="tablist")
    tabs = PAGE.find(role="tab")
    assert len(tabs) == 3 and tablist["attrs"].get("aria-label")
    assert sum(1 for t in tabs if t["attrs"].get("aria-selected") == "true") == 1
    assert [t["attrs"]["tabindex"] for t in tabs].count("0") == 1                # roving tabindex
    for tab in tabs:
        panel = PAGE.find(id=tab["attrs"]["aria-controls"])[0]
        assert panel["attrs"]["role"] == "tabpanel" and panel["attrs"]["aria-labelledby"] == tab["attrs"]["id"]


def test_dialogs_are_native_and_named():
    dialogs = PAGE.find("dialog")
    assert len(dialogs) == 2
    for d in dialogs:
        assert d["attrs"]["aria-labelledby"] in IDS
    for error in (e for e in PAGE.elements if e["attrs"].get("id") in ("new-error", "connect-error")):
        assert error["attrs"].get("role") == "alert"


def test_the_page_does_not_steal_focus_on_load_and_cards_are_not_landmarks():
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    init_body = app[app.index("async function init()"):]
    assert ".focus()" not in init_body.split("loadHistory();")[-1], "init() must not move focus when the page loads"
    assert "autofocus" not in HTML
    assert 'h("section"' not in app, "result / change cards repeat, so they must not be named landmarks"
    assert 'role: "region"' not in app


def test_loaders_ignore_out_of_order_responses_and_chips_do_not_flicker():
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    for guard, loader in (("historyRequest", "loadHistory"), ("databasesRequest", "loadDatabases")):
        body = app[app.index(f"async function {loader}"):]
        body = body[:body.index("\n}\n")]
        assert f"++{guard}" in body and f"mine !== {guard}" in body, f"{loader} must drop superseded responses"
    chips = app[app.index("function chipsFor"):app.index("function buildEmpty")]
    assert "schemaNames == null" in chips and "return []" in chips, "no chips until the schema is known"


def test_the_conversation_is_a_live_log_and_the_composer_is_described():
    (feed,) = PAGE.find(id="feed")
    assert feed["attrs"]["role"] == "log" and feed["attrs"]["aria-live"] == "polite"
    (question,) = PAGE.find("textarea", id="question")
    assert "composer-hint" in question["attrs"]["aria-describedby"]


# ── CSS: tokens, contrast, units, motion, focus ───────────────────────────

def _block(selector_regex: str) -> str:
    match = re.search(selector_regex + r"\s*\{([^{}]*)\}", CSS)
    assert match, f"CSS block not found: {selector_regex}"
    return match.group(1)


def _tokens(block: str) -> dict:
    return {name: value.strip() for name, value in re.findall(r"--([a-z0-9-]+)\s*:\s*([^;]+);", block)}


LIGHT = _tokens(_block(r":root"))
DARK = _tokens(_block(r':root\[data-theme="dark"\]'))
DARK_BY_SYSTEM = _tokens(_block(r':root:not\(\[data-theme="light"\]\)'))


def _luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    channels = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_the_two_dark_blocks_stay_in_sync():
    assert DARK == DARK_BY_SYSTEM, "dark tokens applied by the switch and by the OS setting must be identical"
    assert set(DARK) >= {k for k in LIGHT if k not in ("fs-xs", "fs-sm", "fs-base", "fs-md", "fs-lg", "fs-xl", "radius",
                                                         "radius-sm", "mono") and not k.startswith("sp-")}


TEXT_PAIRS = [   # (foreground, background): normal text needs 4.5:1
    ("text", "bg"), ("text", "surface"), ("text", "surface-2"), ("text", "code-bg"), ("text", "accent-soft"),
    ("text", "warn-bg"), ("text", "ins-bg"), ("text", "del-bg"), ("text", "upd-bg"),
    ("muted", "bg"), ("muted", "surface"), ("muted", "surface-2"), ("muted", "accent-soft"),
    ("accent-ink", "accent"), ("accent", "surface"), ("accent", "surface-2"), ("accent", "accent-soft"),
    ("ok", "surface"), ("ok", "surface-2"), ("bad", "surface"), ("bad", "bad-bg"),
    ("ins", "ins-bg"), ("del", "del-bg"), ("upd", "upd-bg"), ("ins", "surface"), ("del", "surface"), ("upd", "surface"),
]
NON_TEXT_PAIRS = [   # focus rings, control edges, switch icons: 3:1
    ("control-border", "surface"), ("control-border", "bg"), ("accent", "bg"), ("accent", "surface"), ("muted", "surface-2"),
]


@pytest.mark.parametrize("theme,tokens", [("light", LIGHT), ("dark", DARK)])
@pytest.mark.parametrize("fg,bg", TEXT_PAIRS)
def test_text_meets_wcag_aa_contrast(theme, tokens, fg, bg):
    ratio = contrast(tokens[fg], tokens[bg])
    assert ratio >= 4.5, f"{theme}: {fg} on {bg} is {ratio:.2f}:1 (needs 4.5:1)"


@pytest.mark.parametrize("theme,tokens", [("light", LIGHT), ("dark", DARK)])
@pytest.mark.parametrize("fg,bg", NON_TEXT_PAIRS)
def test_controls_and_focus_meet_non_text_contrast(theme, tokens, fg, bg):
    ratio = contrast(tokens[fg], tokens[bg])
    assert ratio >= 3.0, f"{theme}: {fg} on {bg} is {ratio:.2f}:1 (needs 3:1)"


def test_sizes_use_rem_and_media_queries_use_em():
    declared = re.findall(r"font-size\s*:\s*([^;]+);", CSS) + re.findall(r"(?<![-\w])font\s*:\s*([^;]+);", CSS)
    assert declared and not [d for d in declared if re.search(r"\d\s*px", d)], "font sizes must not be fixed in px"
    for query in re.findall(r"@media\s*\(([^)]*(?:width)[^)]*)\)", CSS):
        assert "px" not in query, f"media query {query!r} should use em so it follows the user's font size"


def test_motion_and_contrast_preferences_are_respected():
    assert "prefers-reduced-motion: no-preference" in CSS and "prefers-reduced-motion: reduce" in CSS
    assert "prefers-contrast: more" in CSS and "forced-colors: active" in CSS
    assert "pointer: coarse" in CSS
    transitions = [m.start() for m in re.finditer(r"\btransition\s*:", CSS)]
    gate = CSS.index("prefers-reduced-motion: no-preference")
    assert all(t > gate for t in transitions), "every transition must sit inside the reduced-motion guard"


def test_keyboard_focus_is_never_removed():
    assert ":focus-visible" in CSS
    for match in re.finditer(r"([^{}]+)\{[^{}]*outline\s*:\s*(none|0)\b[^{}]*\}", CSS):
        assert match.group(1).strip() == "main:focus", f"outline removed on {match.group(1).strip()!r}"


def test_small_targets_are_at_least_24px_and_touch_targets_44px():
    small = _block(r"\.btn\.small")
    assert float(re.search(r"min-height\s*:\s*([\d.]+)rem", small).group(1)) >= 1.5            # WCAG 2.2 target size (minimum)
    coarse = CSS[CSS.index("pointer: coarse"):]
    assert float(re.search(r"\.btn[^{]*\{[^}]*min-height\s*:\s*([\d.]+)rem", coarse).group(1)) >= 2.75   # 44px


def test_the_switch_has_a_forced_colors_fallback():
    forced = CSS[CSS.index("forced-colors: active"):]
    assert ".switch-track" in forced and "Highlight" in forced
