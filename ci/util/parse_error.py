#!/usr/bin/env python3
"""Extract errors from configure/build/test logs.

Reads one or more logs (or stdin) and reports the first N lines matching
known error patterns. Output can be JSON or Markdown; JSON is used as the
canonical representation and the Markdown is derived from it to keep both
formats in sync.
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import sys
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# Regex patterns
# -----------------------------------------------------------------------------

@dataclass
class PatternSpec:
    """Specification for a single error pattern.

    name: human‑readable identifier (e.g., "cmake error", "nvcc error").
    type: category string: 'config' | 'build' | 'test' | 'lit'.
    error: compiled regex that matches an error line and provides 'file',
      'line', and 'msg' groups when available.
    context_begin/context_end: optional anchors that delimit a multi‑line
      context block. See iter_matches for capture rules.
    refine_error_by_type: optional list of PatternSpec types to run against the
      captured context; when a refinement pattern matches, replace all fields
      from the parent match (except 'context') with the refinement's fields.
    """
    name: str
    type: str
    error: re.Pattern[str]
    context_begin: Optional[re.Pattern[str]] = None
    context_end: Optional[re.Pattern[str]] = None
    context_begin_inclusive: bool = True
    context_end_inclusive: bool = True
    refine_error_by_type: Optional[List[str]] = None
    # Optional regex to extract a target/test name. The regex should contain
    # a named group 'target' or a first positional group with the desired name.
    target_name: Optional[re.Pattern[str]] = None


CONFIGURE_SPECS: List[PatternSpec] = [
    # CMake configure errors
    # example: "CMake Error at CMakeLists.txt:5 (message):"
    PatternSpec(
        name="cmake error",
        type="config",
        error=re.compile(
            r"^\s*CMake (Error|Fatal) at (?P<file>[^:\n]+):(?P<line>\d+) \((?P<msg>[^)]+)\):",
            re.IGNORECASE,
        ),
        context_end=re.compile(
            r"(Configuring incomplete, errors occurred!|See also )",
            re.IGNORECASE,
        ),
    ),
]

# --- build ------------------------------------------------------------------
BUILD_SPECS: List[PatternSpec] = [
    # C/C++ compiler diagnostics (clang, GCC)
    # example: "foo.cpp:3:5: error: expected ';' after expression"
    PatternSpec(
        name="gcc/clang error",
        type="build",
        error=re.compile(
            r"^(?P<file>[^:\n]+):(?P<line>\d+):(?:\d+:)?\s*(?P<msg>.*\b(error|fatal)\b.*)$",
            re.IGNORECASE,
        ),
        # Capture context from the preceding Ninja failure banner to the
        # final clang summary. For CUDA compilation via clang, the tool emits
        # lines like "8 errors generated when compiling for sm_80."; include
        # those as the end of context when present.
        context_begin=re.compile(r"^\s*FAILED:\s+"),
        context_end=re.compile(
            r"^\s*\d+\s+errors?\s+generated\s+when\s+compiling\s+for\s+sm_\d+\.?\s*$",
            re.IGNORECASE,
        ),
        target_name=re.compile(r"CMakeFiles/(?P<target>[^/\s]+)\.dir/"),
    ),
    # NVCC diagnostics
    # example: "foo.cu(10): error: identifier 'bar' is undefined"
    PatternSpec(
        name="nvcc error",
        type="build",
        error=re.compile(
            r"^(?P<file>[^:(\n]+)\((?P<line>\d+)\):\s*(?P<msg>.*\b(error|fatal)\b.*)$",
            re.IGNORECASE,
        ),
        # Context capture for nvcc in Ninja output
        context_begin=re.compile(r"^\s*FAILED:\s+"),
        context_end=re.compile(r"^\s*\d+\s+error detected in the compilation of "),
        target_name=re.compile(r"CMakeFiles/(?P<target>[^/\s]+)\.dir/"),
    ),
]

# --- LIT --------------------------------------------------------------------
LIT_SPECS: List[PatternSpec] = [
    # lit result lines (unexpected failures/passes)
    # example: "FAIL: suite :: test (1 of 2)"
    PatternSpec(
        name="lit error",
        type="lit",
        error=re.compile(
            r"^\s*(?P<msg>FAIL|XPASS|ERROR):\s+(?:.+?\s+::\s+)?(?P<file>.+?) \(\d+ of \d+\)$",
            re.IGNORECASE,
        ),
        # lit context: between a FAILED banner and the closing banner
        # Example begin: ******************** TEST 'suite :: test' FAILED ********************
        # Example end:   ********************
        context_begin=re.compile(r"^\s*\*{6,}\s*TEST '.*' FAILED\s*\*{6,}\s*$"),
        context_end=re.compile(r"^\s*\*{6,}\s*$"),
        # Update error info with any build errors within the full context match.
        refine_error_by_type=["build"],
        target_name=re.compile(r"TEST '.*?::\s*(?P<target>.*?)'\s*FAILED", re.IGNORECASE),
    ),
    # lit diagnostics
    # example: "lit: /path/format.py:130: fatal: Unsupported RUN line"
    PatternSpec(
        name="lit diagnostic",
        type="lit",
        error=re.compile(
            r"^lit:\s+(?P<file>[^:\n]+):(?P<line>\d+):\s*(?P<msg>.*\b(error|fatal)\b.*)$",
            re.IGNORECASE,
        ),
    ),
]

# --- test -------------------------------------------------------------------
TEST_SPECS: List[PatternSpec] = [
    # CTest summary lines
    # example: "1 - fail (Failed)"
    PatternSpec(
        name="ctest failure",
        type="test",
        error=re.compile(
            r"^\s*\d+/\d+\s+Test\s+#(?P<line>\d+):\s+[^\s]+\s+\.\.+\*\*\*(?P<msg>Failed|Timeout|Not Run|Skipped|Passed)\b.*$",
            re.IGNORECASE,
        ),
        # Capture epilogue from the error header line down to the next test start or final summary
        # End examples:  "Start 2: fail2" OR "0% tests passed, 3 tests failed out of 3"
        context_end=re.compile(
            r"^(\s*Start\s+\d+:|\s*\d+%\s+tests\s+passed,\s+\d+\s+tests\s+failed\s+out\s+of\s+\d+)", re.IGNORECASE),
        context_end_inclusive=False,
        # Sometimes build errors end up in these:
        refine_error_by_type=["build"],
        # Extract test name from the header line itself
        target_name=re.compile(r"^\s*\d+/\d+\s+Test\s+#\d+:\s+(?P<target>[^\s]+)\s+\.+", re.IGNORECASE),
    ),
]

# Combined list of all patterns in evaluation order
# Order matters – tuned for useful diagnostics first.
ERROR_SPECS: List[PatternSpec] = CONFIGURE_SPECS + LIT_SPECS + TEST_SPECS + BUILD_SPECS

# Queryable registries
SPECS_BY_NAME: Dict[str, PatternSpec] = {}
SPECS_BY_TYPE: Dict[str, List[PatternSpec]] = {"config": [], "build": [], "test": [], "lit": []}
for spec in (CONFIGURE_SPECS + BUILD_SPECS + LIT_SPECS + TEST_SPECS):
    SPECS_BY_NAME[spec.name] = spec
    SPECS_BY_TYPE.setdefault(spec.type, []).append(spec)


def get_specs(query: Optional[Iterable[str] | str] = None, *, type: Optional[str] = None) -> List[PatternSpec]:
    """Return pattern specs by name(s) or by type.

    If ``query`` is None, returns all specs (unordered). If a string, returns
    the matching spec (if present). If an iterable of strings, returns the
    matching specs in the given order. If ``ptype`` is provided, returns all
    specs of that category.
    """
    if type is not None:
        return list(SPECS_BY_TYPE.get(type, []))
    if query is None:
        return list(SPECS_BY_NAME.values())
    if isinstance(query, str):
        return [SPECS_BY_NAME[query]] if query in SPECS_BY_NAME else []
    out: List[PatternSpec] = []
    for name in query:
        if name in SPECS_BY_NAME:
            out.append(SPECS_BY_NAME[name])
    return out


# -----------------------------------------------------------------------------
# Matching utilities
# -----------------------------------------------------------------------------

_BUILD_PREFIX_RE = re.compile(r"^.*?/cccl/build/[^/]+/[^/]+/")
_SRC_PREFIX_RE = re.compile(r"^.*?/cccl/")


def _normalize_file(path: str) -> str:
    """Return a canonical, repo-relative POSIX path when possible.

    - Strips build and repo absolute prefixes.
    - Normalizes and collapses any "../" segments (e.g., lib/.../../../../foo -> foo).
    - Uses POSIX separators regardless of host OS to keep outputs stable.
    """
    if not path:
        return ""
    # Work with POSIX-style separators
    p = path.replace("\\", "/")
    # Drop leading build and repo prefixes
    p = _BUILD_PREFIX_RE.sub("", p)
    p = _SRC_PREFIX_RE.sub("", p)
    # Collapse .. and . segments
    p = posixpath.normpath(p)
    # Remove any accidental leading ./
    if p.startswith("./"):
        p = p[2:]
    return p


def find_spec_and_match(line: str) -> Optional[tuple[PatternSpec, re.Match[str]]]:
    """Return the first (spec, match) for the line, or None."""

    for spec in ERROR_SPECS:
        match = spec.error.match(line)
        if match:
            return spec, match
    return None


def iter_matches(
    lines: Iterable[str], limit: Optional[int] = 1
) -> Iterator[Dict[str, str]]:
    """Yield dicts of capture groups for each matching line.

    Parameters
    ----------
    lines:
        Iterable of log lines.
    limit:
        Maximum number of matches to yield. ``None`` yields all.
    """

    count = 0
    raw_lines = list(lines)
    for i, raw in enumerate(raw_lines):
        line = raw.rstrip("\n")
        found = find_spec_and_match(line)
        if found:
            spec, match = found
            result = match.groupdict()
            # Attach origin info for tracing
            result["pattern_type"] = spec.type
            result["pattern_name"] = spec.name
            # Preserve absolute path as captured, and provide normalized + basename variants.
            abs_filepath = (result.get("file", "") or "").strip()
            if abs_filepath:
                rel_filepath = _normalize_file(abs_filepath).strip()
                filename = os.path.basename(rel_filepath or abs_filepath)
                # For legacy consumers, keep 'file' as the normalized relative path
                result["file"] = (rel_filepath or abs_filepath).strip()
                result["abs_filepath"] = abs_filepath
                result["rel_filepath"] = rel_filepath
                result["filename"] = filename

            result["full"] = line

            # Build contextual body: preamble (inclusive) .. error line .. epilogue (inclusive)
            preamble_lines: List[str] = []
            start_idx: Optional[int] = None
            forward_begin_idx: Optional[int] = None
            if spec.context_begin is not None:
                for j in range(i - 1, -1, -1):
                    if spec.context_begin.search(raw_lines[j].rstrip("\n")):
                        start_idx = j
                        break
                if start_idx is None:
                    for j in range(i + 1, len(raw_lines)):
                        if spec.context_begin.search(raw_lines[j].rstrip("\n")):
                            forward_begin_idx = j
                            break
                if start_idx is not None:
                    preamble_lines = [s.rstrip("\n") for s in raw_lines[start_idx:i]]

            end_idx: Optional[int] = None
            epilogue_lines: List[str] = []
            if spec.context_end is not None:
                for k in range(i + 1, len(raw_lines)):
                    if spec.context_end.search(raw_lines[k].rstrip("\n")):
                        end_idx = k
                        break
                if end_idx is not None and (start_idx is not None or forward_begin_idx is None):
                    # Include or exclude the matched end line per spec
                    stop = end_idx + (1 if spec.context_end_inclusive else 0)
                    epilogue_lines = [s.rstrip("\n") for s in raw_lines[i + 1: stop]]

            # Build context: if a forward-begin block exists (e.g., LIT FAILED section),
            # capture exactly from that banner through the closing marker.
            if forward_begin_idx is not None and end_idx is not None:
                stop = end_idx + (1 if spec.context_end_inclusive else 0)
                context_parts = [s.rstrip("\n") for s in raw_lines[forward_begin_idx: stop]]
            else:
                # Insert an alert emoji line just before the error line in context.
                # Indent the emoji to match the error line's leading whitespace
                leading_ws = re.match(r"^(\s*)", line).group(1) if line else ""
                alert_line = f"{leading_ws}⚠️"
                # If context_begin was found, include it per spec; otherwise use preamble as-is
                if start_idx is not None and spec.context_begin is not None and not spec.context_begin_inclusive:
                    # Drop the first preamble line (the begin match)
                    preamble_lines = preamble_lines[1:] if preamble_lines else []
                context_parts = preamble_lines + [alert_line] + [line] + epilogue_lines
            context_str = "\n".join(context_parts)
            result["context"] = context_str

            # Derive target name using spec-provided regex; search context first, then full line.
            target_name: Optional[str] = None
            tn_re = getattr(spec, "target_name", None)
            if tn_re is not None:
                mtn = None
                for s in (context_str, line):
                    if not s:
                        continue
                    mtn = tn_re.search(s)
                    if mtn:
                        break
                if mtn:
                    target_name = mtn.groupdict().get("target") or mtn.group(1)
                    if target_name:
                        target_name = target_name.strip()
            if target_name:
                result["target_name"] = target_name

            # Optional refinement: run additional patterns against the captured
            # context and, on a match, replace all fields except 'context'.
            if spec.refine_error_by_type:
                # Evaluate in registry order to keep behavior deterministic.
                refined = None
                for rtype in spec.refine_error_by_type:
                    for rspec in SPECS_BY_TYPE.get(rtype, []):
                        for cline in context_str.splitlines():
                            rmatch = rspec.error.match(cline.strip("\n"))
                            if rmatch:
                                refined = (rspec, rmatch)
                                break
                        if refined:
                            break
                    if refined:
                        break
                if refined:
                    _, rmatch = refined
                    r = rmatch.groupdict()
                    # Normalize through the same path handling
                    abs_fp = (r.get("file", "") or "").strip()
                    rel_fp = _normalize_file(abs_fp).strip() if abs_fp else ""
                    fname = (
                        os.path.basename(rel_fp or abs_fp)
                        if (rel_fp or abs_fp)
                        else ""
                    )
                    # Replace parent fields (except context)
                    for k in ("file", "line", "msg"):
                        if k in r:
                            result[k] = r[k]
                    # Normalize file-related fields if present
                    if abs_fp:
                        result["file"] = rel_fp or abs_fp
                        result["abs_filepath"] = abs_fp
                        result["rel_filepath"] = rel_fp
                        result["filename"] = fname
                    result["full"] = cline if 'cline' in locals() else result["full"]
            yield result
            count += 1
            if limit is not None and count >= limit:
                break


# -----------------------------------------------------------------------------
# Output formatting
# -----------------------------------------------------------------------------


SUMMARY_LIMIT = 80


def build_summary(m: Dict[str, str], limit: int = SUMMARY_LIMIT) -> str:
    """Build a compact, consistent summary used across all formats.

    Uses the basename filename (when available) for location to keep summaries
    short and stable. The summary is formatted as:

        `file:line`: `truncated message`

    The total summary length is capped by ``limit``.
    """

    loc_file = m.get("filename", "") or m.get("file", "") or m.get("target_name", "")
    loc = f"{loc_file}:{m.get('line', '')}".strip(":")
    msg = m.get("msg", "").strip().replace("`", "'")
    prefix = f"`{loc}`: `" if loc else "`"
    max_msg = max(8, limit - len(prefix) - 1)
    msg_display = msg
    if len(msg_display) > max_msg:
        msg_display = msg_display[: max_msg - 3] + "..."
    return f"{prefix}{msg_display}`"


def format_json(matches: List[Dict[str, str]]) -> str:
    """Return matches encoded as JSON, with extras for consumers.

    Adds:
      - "location": "file:line" convenience field (normalized "file")
      - "summary":  shared truncated summary identical to MD/GH
    """

    out: List[Dict[str, str]] = []
    for m in matches:
        # Use normalized file for location; fall back to target_name if file is missing.
        loc_base = m.get('file', '') or m.get('target_name', '')
        loc = f"{loc_base}:{m.get('line', '')}".strip(":")
        summary = build_summary(m)
        enriched = dict(m)
        # Rename convenience field to 'location' (was 'loc').
        enriched["location"] = loc
        enriched["summary"] = summary
        out.append(enriched)
    return json.dumps(out, indent=2) + "\n"


def format_md(matches: List[Dict[str, str]]) -> str:
    """Return matches in compact Markdown form.

    Sections per match:
      - Shared summary (same as JSON "summary")
      - Location line using normalized "rel_filepath:line"
      - Full Error pre block with full context
    """

    # Build from JSON to ensure a single source of truth
    records: List[Dict[str, str]] = json.loads(format_json(matches))
    sections: List[str] = []
    for m in records:
        summary = m.get("summary", build_summary(m))
        body = (m.get("context") or m.get("full", "") or "").replace("`", "'")
        rel = m.get("rel_filepath", "") or m.get("file", "") or m.get("target_name", "")
        line = m.get("line", "")
        loc = f"{rel}:{line}".strip(":")
        tgt = m.get("target_name")
        target_line = f"🎯 Target Name: {tgt}\n\n" if tgt else ""
        sections.append(
            f"📝 {summary}\n\n"
            f"📍 Location: `{loc}`\n\n"
            f"{target_line}"
            f"🔍 Full Error:\n\n<pre>\n{body}\n</pre>"
        )
    return "\n\n".join(sections) + ("\n" if sections else "")


FORMATTERS = {
    "json": format_json,
    "md": format_md,
}


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract error lines from logs",
        epilog="Reads from <filenames> or stdin and prints the first N matches",
    )
    parser.add_argument(
        "filenames",
        nargs="*",
        help="Log file(s) to parse; reads stdin if omitted",
    )
    parser.add_argument(
        "-n", type=int, default=1, help="Number of matches to output (0 for all)"
    )
    parser.add_argument(
        "-o", "--output", help="Write results to FILE instead of stdout"
    )
    parser.add_argument(
        "--format", choices=FORMATTERS.keys(), default="json", help="Output format"
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    formatter = FORMATTERS[args.format]
    limit = None if args.n == 0 else args.n

    def _dedupe(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
        order: List[tuple[str, str]] = []  # (summary, tkey)
        buckets: Dict[str, Dict[str, Dict[str, str]]] = {}
        for m in records:
            # Build a stable summary key identical to JSON summary
            s = build_summary(m)
            t = (m.get("target_name") or "").strip()
            if s not in buckets:
                buckets[s] = {}
            # If a targetless entry arrives but a targeted one already exists, skip it
            if not t and any(k for k in buckets[s].keys() if k):
                continue
            if t in buckets[s]:
                # already have an entry for (summary, target); keep first
                continue
            if t and "" in buckets[s]:
                # Prefer the one with target over the prior targetless one
                buckets[s][t] = m
                # Replace in order list at the position of the targetless one
                for idx, (os, ot) in enumerate(order):
                    if os == s and ot == "":
                        order[idx] = (s, t)
                        break
                del buckets[s][""]
            else:
                buckets[s][t] = m
                order.append((s, t))
        # Flatten preserving order
        result: List[Dict[str, str]] = []
        for s, t in order:
            result.append(buckets[s][t])
        return result

    if args.filenames:
        # Preserve per-file boundaries: format each file independently and
        # concatenate, matching the behavior used by reference outputs.
        chunks: List[str] = []
        for name in args.filenames:
            with open(name, "r", errors="ignore") as f:
                file_lines = list(f)
            file_matches = list(iter_matches(file_lines, limit=None))
            file_matches = _dedupe(file_matches)
            if limit is not None:
                file_matches = file_matches[:limit]
            chunks.append(formatter(file_matches))
        output = "".join(chunks)
    else:
        lines = list(sys.stdin)
        matches = list(iter_matches(lines, limit=None))
        matches = _dedupe(matches)
        if limit is not None:
            matches = matches[:limit]
        output = formatter(matches)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
    else:
        sys.stdout.write(output)


if __name__ == "__main__":
    main()
