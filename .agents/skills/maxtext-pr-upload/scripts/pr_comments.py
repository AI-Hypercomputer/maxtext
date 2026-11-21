# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fetches review states and inline comment threads for MaxText PRs, flagging unanswered comments."""

import argparse
import json
import re
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request

DEFAULT_REPO = "AI-Hypercomputer/maxtext"
API_BASE = f"https://api.github.com/repos/{DEFAULT_REPO}"
TRANSIENT_HTTP_CODES = {500, 502, 503, 504}


def _default_github_repo() -> str:
  """Infers owner/repo from git remote origin, defaulting to AI-Hypercomputer/maxtext."""
  try:
    proc = subprocess.run(
        ["git", "-c", "safe.directory=*", "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if proc.returncode == 0:
      m = re.search(
          r"github\.com[:/]([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?\s*$",
          proc.stdout,
      )
      if m:
        return m.group(1)
  except OSError:
    pass
  return DEFAULT_REPO


def api_get(
    path: str,
    throttle_s: float = 1.5,
    max_retries: int = 3,
    repo: str = DEFAULT_REPO,
) -> Any:
  """Performs a throttled unauthenticated GitHub REST GET with rate-limit and 5xx retry handling."""
  base_url = f"https://api.github.com/repos/{repo}" if repo else API_BASE
  url = path if path.startswith("http") else f"{base_url}{path}"
  req = urllib.request.Request(
      url,
      headers={
          "Accept": "application/vnd.github+json",
          "User-Agent": "maxtext-pr-comments",
      },
  )
  for attempt in range(max_retries):
    time.sleep(throttle_s)
    try:
      with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
      if e.code == 403 and e.headers.get("x-ratelimit-remaining") == "0":
        reset = int(e.headers.get("x-ratelimit-reset", "0"))
        wait_s = max(1, reset - int(time.time()))
        if attempt + 1 < max_retries and wait_s <= 120:
          time.sleep(wait_s)
          continue
        raise RuntimeError(f"GitHub API rate limit exhausted; resets in {wait_s}s.") from e
      if e.code in TRANSIENT_HTTP_CODES and attempt + 1 < max_retries:
        time.sleep(2.0 ** (attempt + 1))
        continue
      raise
    except urllib.error.URLError:
      if attempt + 1 < max_retries:
        time.sleep(2.0 ** (attempt + 1))
        continue
      raise
  raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts.")


_BOT_BOILERPLATE = (
    "running an automatic MaxText review",
    "I've received your request, and I'm working on it now",
    "unable to process your request",
    "encountered an error while processing your request",
    "## Summary of Changes",
    "reached your daily quota",
    "Codecov",
)


def api_get_paginated(
    path_no_query: str,
    throttle_s: float = 1.5,
    max_pages: int = 10,
    repo: str = DEFAULT_REPO,
) -> list[dict[str, Any]]:
  """Fetches up to max_pages of per_page=100 GitHub REST list items."""
  out: list[dict[str, Any]] = []
  for page in range(1, max_pages + 1):
    items = api_get(
        f"{path_no_query}?per_page=100&page={page}",
        throttle_s=throttle_s,
        repo=repo,
    )
    if not isinstance(items, list) or not items:
      break
    out.extend(items)
    if len(items) < 100:
      break
  return out


def find_top_level_comments(
    issue_comments: list[dict[str, Any]],
    author_login: str,
) -> list[dict[str, Any]]:
  """Extracts substantive top-level issue comments and flags unanswered ones."""
  ordered = sorted(
      issue_comments,
      key=lambda c: (c.get("created_at") or "", c.get("id") or 0),
  )
  author_posts = [
      (c.get("created_at") or "", c.get("id") or 0)
      for c in ordered
      if (c.get("user") or {}).get("login", "") == author_login
  ]
  summaries: list[dict[str, Any]] = []
  for c in ordered:
    user = c.get("user") or {}
    login = user.get("login", "")
    body = c.get("body") or ""
    is_bot = user.get("type") == "Bot" or login.endswith("[bot]")
    if is_bot and any(tok in body for tok in _BOT_BOILERPLATE):
      continue
    if login == author_login:
      continue
    c_time = c.get("created_at") or ""
    c_id = c.get("id") or 0
    answered = any(
        (at > c_time if (at and c_time) else aid > c_id) or (at == c_time and aid > c_id) for at, aid in author_posts
    )
    summaries.append(
        {
            "id": c_id,
            "reviewer": login,
            "created_at": c_time,
            "body": body,
            "needs_reply": not answered,
        }
    )
  return summaries


def group_inline_threads(comments: list[dict[str, Any]], author_login: str) -> list[dict[str, Any]]:
  """Groups inline review comments into threads and flags threads where the last reply is not by author_login."""
  threads: dict[int, list[dict[str, Any]]] = {}
  for c in sorted(comments, key=lambda item: item.get("id", 0)):
    root_id = c.get("in_reply_to_id") or c["id"]
    threads.setdefault(root_id, []).append(c)

  summaries: list[dict[str, Any]] = []
  for root_id, items in threads.items():
    root = items[0]
    last = items[-1]
    last_user = (last.get("user") or {}).get("login", "")
    needs_reply = last_user != author_login
    summaries.append(
        {
            "root_id": root_id,
            "path": root.get("path", ""),
            "line": root.get("line") or root.get("original_line"),
            "reviewer": (root.get("user") or {}).get("login", ""),
            "last_user": last_user,
            "needs_reply": needs_reply,
            "created_at": root.get("created_at", ""),
            "body": root.get("body", ""),
            "replies": len(items) - 1,
        }
    )
  return summaries


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("prs", nargs="+", type=int, help="PR numbers to inspect")
  parser.add_argument(
      "--since",
      default="",
      help="Only show comments created at or after ISO timestamp",
  )
  parser.add_argument(
      "--full",
      action="store_true",
      help="Print full comment bodies instead of 120-char previews",
  )
  parser.add_argument(
      "--repo",
      default=_default_github_repo(),
      help="GitHub owner/repo (default: origin remote or AI-Hypercomputer/maxtext)",
  )
  parser.add_argument("--throttle", type=float, default=1.5, help="Seconds between API requests")
  args = parser.parse_args()

  for pr_num in args.prs:
    pr = api_get(f"/pulls/{pr_num}", throttle_s=args.throttle, repo=args.repo)
    author = (pr.get("user") or {}).get("login", "")
    reviews = api_get_paginated(f"/pulls/{pr_num}/reviews", throttle_s=args.throttle, repo=args.repo)
    inline = api_get_paginated(f"/pulls/{pr_num}/comments", throttle_s=args.throttle, repo=args.repo)
    issue_comments = api_get_paginated(f"/issues/{pr_num}/comments", throttle_s=args.throttle, repo=args.repo)

    review_states = [
        f"{(r.get('user') or {}).get('login')}:{r.get('state')}"
        for r in reviews
        if r.get("state") and r.get("state") != "COMMENTED"
    ]
    threads = group_inline_threads(inline, author)
    top_comments = find_top_level_comments(issue_comments, author)
    if args.since:
      threads = [t for t in threads if t["created_at"] >= args.since]
      top_comments = [tc for tc in top_comments if tc["created_at"] >= args.since]

    inline_unanswered = sum(1 for t in threads if t["needs_reply"])
    top_unanswered = sum(1 for tc in top_comments if tc["needs_reply"])
    rev_str = ", ".join(review_states) or "none"
    print(
        f"PR #{pr_num} (author={author}) | Reviews: {rev_str} | Inline"
        f" unanswered: {inline_unanswered}/{len(threads)} | Top-level"
        f" unanswered: {top_unanswered}/{len(top_comments)}"
    )
    for t in threads:
      flag = "[NEEDS_REPLY]" if t["needs_reply"] else "[ANSWERED]"
      body = t["body"] if args.full else t["body"].replace("\n", " ")[:120]
      print(f"  {flag} id={t['root_id']} ({t['reviewer']})" f" {t['path']}:{t['line']} -> {body}")
    for tc in top_comments:
      flag = "[NEEDS_REPLY]" if tc["needs_reply"] else "[ANSWERED]"
      body = tc["body"] if args.full else tc["body"].replace("\n", " ")[:120]
      print(f"  {flag} [TOP-LEVEL] id={tc['id']} ({tc['reviewer']})" f" issue_comment -> {body}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
