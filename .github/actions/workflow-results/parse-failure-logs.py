#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import textwrap
from collections import defaultdict
from pathlib import Path


def extract_jobs(workflow):
    jobs = []
    for group in workflow.values():
        if "standalone" in group:
            jobs += group["standalone"]
        if "two_stage" in group:
            for two_stage in group["two_stage"]:
                jobs += two_stage["producers"]
                jobs += two_stage["consumers"]
    return jobs


def _resolve_job_name_and_url(name: str, job_urls: dict[str, str]) -> tuple[str, str]:
    """Return a display name and URL for a job.

    Attempts exact match by name; otherwise, chooses the longest GitHub job
    name that contains the provided name (to include matrix details).
    """

    if name in job_urls:
        return name, job_urls[name]
    candidates = [n for n in job_urls.keys() if n.startswith(name) or name in n]
    if candidates:
        gh_name = max(candidates, key=len)
        return gh_name, job_urls.get(gh_name, "")
    return name, ""


def _first_match_via_parser(parser: Path, log_path: Path) -> dict | None:
    """Run parse_error.py on a log and return the first JSON match, if any."""

    if not log_path.exists():
        return None
    cmd = [
        sys.executable,
        str(parser),
        "-n",
        "1",
        "--format",
        "json",
        str(log_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if isinstance(data, list) and data:
        return data[0]
    return None


def _generate_job_id_map(workflow: dict) -> dict[str, str]:
    """Map full GitHub job name to custom job id from workflow.json.

    Mirrors logic in parse-job-times to keep name construction consistent.
    """
    job_id_map: dict[str, str] = {}
    for group_name, group_json in workflow.items():
        standalone = group_json.get("standalone", [])
        for job in standalone:
            name = f"{group_name} / {job['name']}"
            job_id_map[name] = job["id"]
        for pc in group_json.get("two_stage", []):
            for job in pc.get("producers", []) + pc.get("consumers", []):
                name = f"{group_name} / {pc['id']} / {job['name']}"
                job_id_map[name] = job["id"]
    return job_id_map


def _deep_link_for_location(repo: str, sha: str, workspace: str, location_disp: str, file_path: str) -> str:
    """Return a GitHub blob URL for a location if it exists in the repo checkout.

    - Ensures the path resolves within `workspace` and is a file.
    - Extracts the trailing line number from `location_disp` of the form
      "path:line". If no line can be parsed, returns empty string.
    """
    if not (repo and sha and file_path and location_disp):
        return ""
    # Extract line number from the display string (use right-most colon)
    line = ""
    if ":" in location_disp:
        try:
            candidate_line = location_disp.rsplit(":", 1)[1].strip()
            # Validate numeric line
            int(candidate_line)
            line = candidate_line
        except Exception:
            line = ""
    if not line:
        return ""
    try:
        ws_path = Path(workspace).resolve()
        candidate_path = (ws_path / file_path.lstrip("/")).resolve()
        # Ensure candidate is within workspace
        candidate_path.relative_to(ws_path)
        if not candidate_path.is_file():
            return ""
        repo_rel = candidate_path.relative_to(ws_path).as_posix()
        return f"https://github.com/{repo}/blob/{sha}/{repo_rel}#L{line}"
    except Exception:
        return ""


def _make_link(text, url):
    # Need to make html link in list item to avoid markdown parsing issues
    return f"<a href=\"{url}\">{text}</a>" if url else text


def _make_heading(level, text, anchor=None):
    # Same for heading markup
    if anchor:
        return f"<h{level} id=\"{anchor}\">{text}</h{level}>"
    else:
        return f"<h{level}>{text}</h{level}>"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow_json")
    parser.add_argument("jobs_json")
    parser.add_argument("jobs_dir")
    args = parser.parse_args()

    with open(args.workflow_json) as f:
        workflow = json.load(f)
    with open(args.jobs_json) as f:
        gha_jobs = json.load(f)

    # Map GH job name (includes matrix) to job info and URL
    job_by_name = {job["name"]: job for job in gha_jobs}

    # Build mapping from job id -> info
    jobs_info = {}
    id_to_display: dict[str, str] = {}
    id_to_url: dict[str, str] = {}
    name_to_id = _generate_job_id_map(workflow)

    # Build reverse maps using GH names → ids
    for gh_name, gh_job in job_by_name.items():
        if gh_name in name_to_id:
            cid = name_to_id[gh_name]
            id_to_display[cid] = gh_name
            id_to_url[cid] = gh_job.get("html_url", "")

    for job in extract_jobs(workflow):
        matrix = job["origin"]["matrix_job"]
        name = matrix.get("job_name", job["id"])
        project = matrix.get("project", "unknown")
        # Prefer mapping via custom id; fallback to fuzzy name match.
        disp_name = id_to_display.get(job["id"]) or name
        url = (
            id_to_url.get(job["id"])
            or _resolve_job_name_and_url(
                name, {k: v.get("html_url", "") for k, v in job_by_name.items()}
            )[1]
        )
        jobs_info[job["id"]] = {
            "name": name,  # matrix job name from workflow
            "display": disp_name,  # GH job name with matrix
            "project": project,
            "url": url,
        }

    # errors[project][summary] -> {
    #   full, context,
    #   jobs:set[(display_name, url)],
    #   location, file, abs,
    #   targets_all:set[str],
    #   targets_by_job: dict[display_name, set[str]]
    # }
    errors = defaultdict(
        lambda: defaultdict(
            lambda: {
                "full": "",
                "context": "",
                "jobs": set(),
                "location": "",
                "file": "",
                "abs": "",
                "filename": "",
                "line": "",
                "msg": "",
                "targets_all": set(),
                "targets_by_job": defaultdict(set),
            }
        )
    )
    unmatched = defaultdict(set)

    for job_id, info in jobs_info.items():
        job_dir = os.path.join(args.jobs_dir, job_id)
        if not os.path.isdir(job_dir):
            continue
        if os.path.exists(os.path.join(job_dir, "success")):
            continue

        found = False
        parser = Path(__file__).resolve().parents[3] / "ci" / "util" / "parse_error.py"
        for log_name in ["configure.log", "build.log", "test.log"]:
            log_path = os.path.join(job_dir, log_name)
            match = _first_match_via_parser(parser, Path(log_path))
            if match:
                found = True
                filepath = (match.get("rel_filepath") or match.get("file", "") or "").strip()
                abs_path = (match.get("abs_filepath") or "").strip()
                line_no = (match.get("line", "") or "").strip()
                summary = (match.get("summary") or "").strip()
                target = (match.get("target_name") or "").strip()
                location = f"{filepath}:{line_no}".strip(":")
                entry = errors[info["project"]][summary]
                if not entry["full"]:
                    entry["full"] = match.get("full", "")
                if not entry["context"]:
                    entry["context"] = match.get("context", "") or match.get("full", "")
                if not entry["msg"]:
                    entry["msg"] = (match.get("msg") or "").strip()
                if not entry["location"]:
                    entry["location"] = location
                if not entry["file"]:
                    entry["file"] = filepath
                if not entry["abs"]:
                    entry["abs"] = abs_path
                if not entry["filename"]:
                    entry["filename"] = (match.get("filename") or "").strip()
                if not entry["line"]:
                    entry["line"] = line_no
                disp = info.get("display", info["name"])
                entry["jobs"].add((disp, info["url"]))
                if target:
                    entry["targets_all"].add(target)
                    entry["targets_by_job"][disp].add(target)
                break
            if found:
                break
        if not found:
            unmatched[info["project"]].add(
                (info.get("display", info["name"]), info["url"])
            )

    if not errors and not unmatched:
        return

    print(f"<details><summary>{_make_heading(2, '🚨 Failure Log', 'failure-log')}</summary>\n")
    # Build repo/SHA context for deep links to file locations in GitHub UI.
    repo = (os.environ.get("GITHUB_REPOSITORY") or "NVIDIA/cccl").strip()
    sha = (os.environ.get("GITHUB_SHA") or "").strip()
    workspace = os.environ.get("GITHUB_WORKSPACE") or os.getcwd()
    # Summary page URL (if available) for linking PR comment elements
    summary_url = ""
    try:
        summary_url_path = Path("workflow/summary_url.txt")
        if summary_url_path.exists():
            summary_url = summary_url_path.read_text(encoding="utf-8").strip()
    except Exception:
        summary_url = ""
    # Collect compact error rows for PR comment
    compact_rows: list[str] = []
    error_counter = 0
    for project in sorted(errors):
        for summary in sorted(errors[project]):
            data = errors[project][summary]
            short_summary = textwrap.shorten(summary, width=120, placeholder="...")
            heading = f"{project}: {short_summary}"
            error_counter += 1
            # Anchor to the -jobs section -- it's towards the end of the error and makes sure the
            # browser scrolls down far enough. Autoscrolling the summary page behaves oddly.
            # It also expands the job section to make the failing configs visible, so that's nice, too.
            anchor_url = f"{summary_url}#user-content-error-{error_counter}-jobs" if summary_url else ""
            msg = (data.get("msg") or "").strip()

            print(f"<details><summary>{_make_heading(3, f'⚠️ {heading}', f'error-{error_counter}')}</summary>\n")

            if anchor_url:
                print(f"- 🔗 **Shareable Link**: {_make_link('here', anchor_url)}")

            # Error message
            msg = (data.get("msg") or "").strip()
            if msg:
                safe_msg = msg.replace("`", "\\`")
                print(f"- 💬 **Error Message**: `{safe_msg}`")

            if data.get("location"):
                loc_disp = data["location"].strip()
                loc_file = data.get("file", "").strip()
                url = _deep_link_for_location(repo, sha, workspace, loc_disp, loc_file)
                print(f"- 📍 **Location**: {_make_link(f"`{loc_disp}`", url)}")

            # Summarize targets across all jobs for this error
            targets = sorted(t for t in data.get("targets_all", set()) if t)
            if targets:
                def _md_code(s: str) -> str:
                    return f"`{(s or '').replace('`', '\\`')}`"
                targets_md = ", ".join(_md_code(t) for t in targets)
                print(f"- 🎯 **Target(s)**: {targets_md}")

            print("<ul>")

            # Nested details for Full Error context
            print("<li><details><summary>🔍 <b>Full Error With Context</b></summary>\n")
            print("<pre>")
            print(data.get("context") or data.get("full") or "")
            print("</pre>\n</details>\n</li>\n")

            # Collapsible Jobs section (placed after Full Error)
            job_count = len(data.get("jobs", []))
            print(
                f"<li><details><summary><span id=\"error-{error_counter}-jobs\">🧰 <b>{job_count} Job(s)</b></span></summary>\n")
            for job_name, url in sorted(data["jobs"], key=lambda x: x[0]):
                print(f"- {_make_link(job_name, url)}")
            print("</details></li>\n")
            print("</ul>")
            print("</details>")

            # Build compact PR comment row in the format:
            # - **project** (N job(s)): <filename:line (linked if valid)>: `<full untruncated message>` [full log]
            # Choose last job alphabetically by display name for the log link
            job_url = ""
            if data["jobs"]:
                last_job = sorted(data["jobs"], key=lambda x: x[0])[-1]
                _, job_url = last_job
            # Display only filename:line (no path) using values from JSON
            filename = (data.get("filename") or "").strip()
            line_no = (data.get("line") or "").strip()
            loc_file = (data.get("file") or "").strip()  # repo-relative path for linking
            loc_disp_short = f"{filename}:{line_no}".strip(":") if filename or line_no else ""
            loc_url = _deep_link_for_location(repo, sha, workspace, loc_disp_short, loc_file)
            loc_md = (
                f"[`{loc_disp_short}`]({loc_url})" if (loc_disp_short and loc_url)
                else f"`{loc_disp_short}`" if loc_disp_short else ""
            )
            # Use truncated message for compact PR comment; collapse whitespace
            msg_raw = (data.get("msg") or summary or "").strip()
            msg_trunc = textwrap.shorten(msg_raw, width=60, placeholder="...")
            msg_one_line = " ".join(msg_trunc.split())
            msg_one_line = msg_one_line.replace("`", "\\`")
            jobs_text = f"{len(data.get('jobs', []))} job{'s' if len(data.get('jobs', [])) != 1 else ''}"
            jobs_md = f"[{jobs_text}]({anchor_url})" if anchor_url else jobs_text
            row = f"**{project}**: ({jobs_md}): {loc_md} `{msg_one_line}`"
            if job_url:
                row += f" [full log]({job_url})"
            compact_rows.append(row)
    # end for project/errors
    if unmatched:
        for project in sorted(unmatched):
            print(f"<details><summary>{_make_heading(3, project)}</summary>")
            for job_name, url in sorted(unmatched[project], key=lambda x: x[0]):
                print(f"- {_make_link(job_name, url)}")
            print("\n</details>\n")
    print("</details>")

    # Write compact rows for PR comment consumption
    try:
        os.makedirs("workflow", exist_ok=True)
        with open("workflow/errors_list.md", "w") as f:
            for row in compact_rows:
                f.write(row + "\n")
    except Exception:
        pass


if __name__ == "__main__":
    main()
