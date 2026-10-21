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
import sys
import time
from typing import Any
import urllib.error
import urllib.request

API_BASE = "https://api.github.com/repos/AI-Hypercomputer/maxtext"
TRANSIENT_HTTP_CODES = {500, 502, 503, 504}


def api_get(path: str, throttle_s: float = 1.5, max_retries: int = 3) -> Any:
  """Performs a throttled unauthenticated GitHub REST GET with rate-limit and 5xx retry handling."""
  url = path if path.startswith("http") else f"{API_BASE}{path}"
  req = urllib.request.Request(
      url,
      headers={"Accept": "application/vnd.github+json", "User-Agent": "maxtext-pr-comments"},
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
  parser.add_argument("--since", default="", help="Only show comments created at or after ISO timestamp")
  parser.add_argument("--full", action="store_true", help="Print full comment bodies instead of 120-char previews")
  parser.add_argument("--throttle", type=float, default=1.5, help="Seconds between API requests")
  args = parser.parse_args()

  for pr_num in args.prs:
    pr = api_get(f"/pulls/{pr_num}", throttle_s=args.throttle)
    author = (pr.get("user") or {}).get("login", "")
    reviews = api_get(f"/pulls/{pr_num}/reviews?per_page=100", throttle_s=args.throttle)
    inline = api_get(f"/pulls/{pr_num}/comments?per_page=100", throttle_s=args.throttle)

    review_states = [
        f"{(r.get('user') or {}).get('login')}:{r.get('state')}"
        for r in reviews
        if r.get("state") and r.get("state") != "COMMENTED"
    ]
    threads = group_inline_threads(inline, author)
    if args.since:
      threads = [t for t in threads if t["created_at"] >= args.since]

    unanswered = sum(1 for t in threads if t["needs_reply"])
    rev_str = ", ".join(review_states) or "none"
    print(f"PR #{pr_num} (author={author}) | Reviews: {rev_str} | Unanswered: {unanswered}/{len(threads)}")
    for t in threads:
      flag = "[NEEDS_REPLY]" if t["needs_reply"] else "[ANSWERED]"
      body = t["body"] if args.full else t["body"].replace("\n", " ")[:120]
      print(f"  {flag} id={t['root_id']} ({t['reviewer']}) {t['path']}:{t['line']} -> {body}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
