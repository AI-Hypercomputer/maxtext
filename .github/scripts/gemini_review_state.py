# Copyright 2026 Google LLC
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

"""Deterministic state handling for the MaxText Gemini pull request review.

Used by `.github/workflows/gemini_review.yml`, with two subcommands:

* `prepare` runs before the model. It decides the review mode (`skip`, `full`, or `incremental`), deletes pending
  reviews left behind by cancelled runs, and writes the context the prompt reads: `review_mode.json`,
  `review_delta.md` (what changed since the bot's last review), `previous_threads.json` (the bot's own unresolved
  review threads), and `other_findings.json` (the bot's other findings, which must not be raised again). If the GitHub
  API fails, explicit requests fall back to a full review without earlier state and automatic reviews are skipped.
* `apply` runs after the model. It reads the per-thread verdicts from the model's final response and replies to, or
  resolves, the bot's own threads. The model itself never gets reply or resolve tools.

SECURITY: this script never checks out, builds, or executes pull request code. It reads the pull request only through
the GitHub API (via the `gh` CLI), and treats all pull request text as untrusted data.

Only the Python standard library and the `gh` CLI are used, so that the script runs on a stock GitHub runner.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

MODE_SKIP = "skip"
MODE_FULL = "full"
MODE_INCREMENTAL = "incremental"

VERDICT_FIXED = "fixed"
VERDICT_STILL_OPEN_UNCHANGED = "still_open_unchanged"
VERDICT_STILL_OPEN_CHANGED = "still_open_changed"
VERDICT_DISPUTED_ACCEPT = "disputed_accept"
VERDICT_DISPUTED_REJECT = "disputed_reject"
VERDICTS = frozenset(
    {
        VERDICT_FIXED,
        VERDICT_STILL_OPEN_UNCHANGED,
        VERDICT_STILL_OPEN_CHANGED,
        VERDICT_DISPUTED_ACCEPT,
        VERDICT_DISPUTED_REJECT,
    }
)

# Hidden markers in the bot's own comments.
CAP_MARKER = "<!-- maxtext-review:auto-review-limit -->"
FOLLOWUP_MARKER = "<!-- maxtext-review:followup:{verdict} -->"
VERDICTS_TAG = "maxtext-review-verdicts"

# Only replies from these participants (or the pull request author) count as a dispute of a finding.
TRUSTED_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})

DEFAULT_MAX_AUTO_REVIEWS = 20
# The compare API returns at most 300 files. A diff with exactly 300 files is indistinguishable from a truncated one,
# so it is also treated as incomplete (and falls back to a full review).
COMPARE_MAX_FILES = 300
# Above this size, the delta is too large to be useful in incremental mode, so a full review runs instead.
MAX_DELTA_CHARS = 200_000
MAX_PREVIOUS_THREADS = 30
# Each thread in the prompt keeps its first comment (the finding) and its most recent replies, up to this many in total.
MAX_PROMPT_THREAD_COMMENTS = 6
MAX_RESOLVED_FINDINGS = 30
# Unresolved bot threads beyond MAX_PREVIOUS_THREADS are listed only briefly, up to this many.
MAX_OVERFLOW_FINDINGS = 30
MAX_COMMENT_CHARS = 2_000
MAX_RESOLVED_FINDING_CHARS = 300
MAX_REASON_CHARS = 500
NO_DELTA_TEXT = "No incremental delta: this is a full review."

THREADS_QUERY = """
query($owner: String!, $repo: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id
          isResolved
          isOutdated
          path
          line
          originalLine
          root: comments(first: 1) {
            nodes { databaseId author { login } authorAssociation body createdAt }
          }
          tail: comments(last: 50) {
            totalCount
            nodes { databaseId author { login } authorAssociation body createdAt }
          }
        }
      }
    }
  }
}
"""

RESOLVE_MUTATION = """
mutation($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) { thread { id isResolved } }
}
"""


class GitHubError(Exception):
  """Raised when a GitHub API call fails."""

  def __init__(self, message, not_found=False):
    super().__init__(message)
    self.not_found = not_found


# ---------------------------------------------------------------------------------------------------------------------
# GitHub API access (through the `gh` CLI).
# ---------------------------------------------------------------------------------------------------------------------


def gh_api(path, method="GET", fields=None, paginate=False):
  """Calls the GitHub REST or GraphQL API with `gh api` and returns the decoded JSON response.

  Args:
    path: API path, e.g. `repos/o/r/pulls/1`, or `graphql`.
    method: HTTP method.
    fields: Mapping of name to value. Strings are sent with `-f` (raw), everything else with `-F` (typed).
    paginate: Whether to follow pagination; the pages of a list response are concatenated.

  Returns:
    The decoded JSON response.

  Raises:
    GitHubError: If the call fails.
  """
  cmd = ["gh", "api", "--method", method, path]
  if paginate:
    cmd += ["--paginate", "--slurp"]
  for name, value in (fields or {}).items():
    if value is None:
      continue
    if isinstance(value, str):
      cmd += ["-f", f"{name}={value}"]
    else:
      cmd += ["-F", f"{name}={json.dumps(value)}"]
  result = subprocess.run(cmd, capture_output=True, text=True, check=False)
  if result.returncode != 0:
    stderr = result.stderr.strip()
    raise GitHubError(f"gh api {method} {path} failed: {stderr}", not_found="HTTP 404" in stderr)
  data = json.loads(result.stdout) if result.stdout.strip() else None
  if paginate and isinstance(data, list) and all(isinstance(page, list) for page in data):
    data = [item for page in data for item in page]
  if isinstance(data, dict) and data.get("errors"):
    raise GitHubError(f"gh api {path} returned errors: {data['errors']}")
  return data


def fetch_review_threads(owner, repo, number):
  """Returns all review threads of the pull request, as GraphQL nodes normalized by `normalize_thread`."""
  threads = []
  cursor = None
  while True:
    data = gh_api(
        "graphql",
        method="POST",
        fields={"query": THREADS_QUERY, "owner": owner, "repo": repo, "number": number, "cursor": cursor},
    )
    connection = data["data"]["repository"]["pullRequest"]["reviewThreads"]
    threads.extend(normalize_thread(t) for t in connection["nodes"])
    if not connection["pageInfo"]["hasNextPage"]:
      return threads
    cursor = connection["pageInfo"]["endCursor"]


def fetch_compare_files(repository, base_ref, sha):
  """Returns the files of the merge-base diff `base_ref...sha`, or None if unavailable or incomplete."""
  try:
    # `per_page=1` limits the commit list, which is not used; the files are always returned on the first page.
    data = gh_api(f"repos/{repository}/compare/{base_ref}...{sha}?per_page=1")
  except GitHubError as error:
    print(f"::warning::Could not compare {base_ref}...{sha}: {error}")
    return None
  files = data.get("files") or []
  if len(files) >= COMPARE_MAX_FILES:
    print(f"::notice::{base_ref}...{sha} changes {len(files)} files or more; the per-file diff may be incomplete.")
    return None
  return files


# ---------------------------------------------------------------------------------------------------------------------
# Pure helpers (unit tested in gemini_review_state_test.py).
# ---------------------------------------------------------------------------------------------------------------------


def normalize_thread(node):
  """Merges the `root` (first comment) and `tail` (last comments) connections of a thread into `comments`.

  Long threads are fetched as their first comment plus their last comments, so that both the finding and the latest
  replies are always present. Comments in between may be missing.

  Args:
    node: A review thread node from `THREADS_QUERY`.

  Returns:
    The node with `comments: {"nodes": [...]}`, oldest first and without duplicates.
  """
  if "comments" in node:
    return node
  comments = []
  seen = set()
  for comment in (node.get("root") or {}).get("nodes", []) + (node.get("tail") or {}).get("nodes", []):
    if comment.get("databaseId") in seen:
      continue
    seen.add(comment.get("databaseId"))
    comments.append(comment)
  normalized = {k: v for k, v in node.items() if k not in ("root", "tail")}
  normalized["comments"] = {"nodes": comments}
  return normalized


def is_same_login(login, bot_login):
  """Compares logins, ignoring the `[bot]` suffix that REST includes and GraphQL omits for GitHub Apps."""
  if not login or not bot_login:
    return False
  return login.removesuffix("[bot]").lower() == bot_login.removesuffix("[bot]").lower()


def file_fingerprint(file):
  """Returns a fingerprint of a file's change in a pull request that is stable across rebases.

  Only the added and removed lines are hashed: hunk headers (line numbers) and context lines can change when the pull
  request is rebased onto a newer base, even though the change itself did not.

  Args:
    file: A file entry from the GitHub compare API.

  Returns:
    A hex digest.
  """
  digest = hashlib.sha256()
  digest.update(f"{file.get('status')}\0{file.get('previous_filename') or ''}\0".encode())
  patch = file.get("patch")
  if patch is None:
    # Binary or very large file: GitHub omits the patch, so fall back to the content of the file after the change.
    digest.update(f"blob:{file.get('sha')}".encode())
  else:
    for line in patch.splitlines():
      if line.startswith(("+", "-")):
        digest.update(line.encode())
        digest.update(b"\n")
  return digest.hexdigest()


def compute_delta(old_files, new_files):
  """Returns the files whose change differs between two versions of a pull request.

  Args:
    old_files: Compare API file entries at the last reviewed commit.
    new_files: Compare API file entries at the current head.

  Returns:
    A list of dicts with `filename`, `kind` (`added`, `changed`, or `removed` relative to the last review),
    `old_patch`, and `new_patch`, sorted by filename.
  """
  old_by_name = {f["filename"]: f for f in old_files}
  new_by_name = {f["filename"]: f for f in new_files}
  delta = []
  for filename in sorted(old_by_name.keys() | new_by_name.keys()):
    old = old_by_name.get(filename)
    new = new_by_name.get(filename)
    if old is None:
      kind = "added"
    elif new is None:
      kind = "removed"
    elif file_fingerprint(old) != file_fingerprint(new):
      kind = "changed"
    else:
      continue
    delta.append(
        {
            "filename": filename,
            "kind": kind,
            "old_patch": (old or {}).get("patch"),
            "new_patch": (new or {}).get("patch"),
        }
    )
  return delta


def render_delta(delta, old_sha, new_sha):
  """Renders the delta as Markdown for the prompt."""
  kind_text = {
      "added": "newly changed by the pull request since the last review",
      "changed": "change modified since the last review",
      "removed": "no longer changed by the pull request (the earlier change was reverted)",
  }
  lines = [
      f"Changes since the last review (reviewed commit `{old_sha}`, current head `{new_sha}`).",
      "Each section shows the pull request's diff for the file at the last review and now.",
      "",
  ]
  for entry in delta:
    lines += [f"### `{entry['filename']}`: {kind_text[entry['kind']]}", ""]
    for label, patch in (("At the last review", entry["old_patch"]), ("Now", entry["new_patch"])):
      if entry["kind"] == "added" and label == "At the last review":
        continue
      if entry["kind"] == "removed" and label == "Now":
        continue
      text = patch if patch is not None else "(diff not available)"
      fence = code_fence(text)
      lines += [f"{label}:", "", f"{fence}diff", text, fence, ""]
  return "\n".join(lines)


def code_fence(text):
  """Returns a backtick fence longer than any backtick run in `text`, so the text cannot close the block."""
  longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
  return "`" * max(3, longest + 1)


def is_trusted_reply(comment, bot_login, pr_author):
  """Returns whether a thread reply may dispute a finding: by the pull request author or a repository maintainer."""
  login = (comment.get("author") or {}).get("login")
  if is_same_login(login, bot_login):
    return False
  return is_same_login(login, pr_author) or comment.get("authorAssociation") in TRUSTED_ASSOCIATIONS


def summarize_thread(thread, bot_login, pr_author):
  """Returns the prompt view of a review thread.

  Only the first comment (the finding) and the most recent replies are kept, so that one long discussion cannot crowd
  out the review. `plan_thread_actions` still checks the full live thread.

  Args:
    thread: A normalized review thread node.
    bot_login: The bot's login.
    pr_author: Login of the pull request author.

  Returns:
    A dict with `thread_id`, `path`, `line`, `is_outdated`, `omitted_comments`, and `comments`.
  """
  comments = thread["comments"]["nodes"]
  kept = list(enumerate(comments))
  if len(kept) > MAX_PROMPT_THREAD_COMMENTS:
    recent = MAX_PROMPT_THREAD_COMMENTS - 1
    kept = kept[:1] + kept[-recent:]
  return {
      "thread_id": thread["id"],
      "path": thread.get("path"),
      "line": thread.get("line") or thread.get("originalLine"),
      "is_outdated": bool(thread.get("isOutdated")),
      "omitted_comments": len(comments) - len(kept),
      "comments": [
          {
              "author": (c.get("author") or {}).get("login"),
              "is_bot": is_same_login((c.get("author") or {}).get("login"), bot_login),
              "trusted": i > 0 and is_trusted_reply(c, bot_login, pr_author),
              "created_at": c.get("createdAt"),
              "body": (c.get("body") or "")[:MAX_COMMENT_CHARS],
          }
          for i, c in kept
      ],
  }


def _started_by_bot(thread, bot_login):
  """Returns whether the bot wrote the first comment of the thread."""
  comments = thread["comments"]["nodes"]
  return bool(comments) and is_same_login((comments[0].get("author") or {}).get("login"), bot_login)


def select_bot_threads(threads, bot_login):
  """Returns the unresolved review threads started by the bot, oldest first."""
  return [t for t in threads if not t.get("isResolved") and _started_by_bot(t, bot_login)]


def compact_finding(thread):
  """Returns the short prompt view of a bot finding that needs no verdict."""
  return {
      "path": thread.get("path"),
      "line": thread.get("line") or thread.get("originalLine"),
      "resolved": bool(thread.get("isResolved")),
      "finding": (thread["comments"]["nodes"][0].get("body") or "")[:MAX_RESOLVED_FINDING_CHARS],
  }


def other_bot_findings(threads, bot_login):
  """Returns the bot's findings that are not in `previous_threads.json`, so that the model does not raise them again.

  Args:
    threads: All review threads (normalized GraphQL nodes), oldest first.
    bot_login: The bot's login.

  Returns:
    Compact views (see `compact_finding`) of the newest resolved bot findings, followed by the unresolved bot findings
    beyond the first `MAX_PREVIOUS_THREADS`.
  """
  bot_threads = [t for t in threads if _started_by_bot(t, bot_login)]
  resolved = [t for t in bot_threads if t.get("isResolved")][-MAX_RESOLVED_FINDINGS:]
  overflow = select_bot_threads(bot_threads, bot_login)[MAX_PREVIOUS_THREADS:][:MAX_OVERFLOW_FINDINGS]
  return [compact_finding(t) for t in resolved + overflow]


def bot_reviews(reviews, bot_login):
  """Returns the bot's submitted summary reviews, oldest first.

  Replying to a review comment through the REST API creates a separate review with an empty body, so reviews with an
  empty body (the bot's own thread follow-ups) are not summary reviews and are ignored. A summary review's `commit_id`
  is the commit it reviewed: the prompt pins it to `head_sha`, and stale pending reviews are deleted before each run.

  Args:
    reviews: Review entries from the GitHub REST API.
    bot_login: The bot's login.

  Returns:
    The bot's submitted reviews with a non-empty body, sorted by submission time.
  """
  submitted = [
      r
      for r in reviews
      if is_same_login((r.get("user") or {}).get("login"), bot_login)
      and r.get("state") != "PENDING"
      and (r.get("body") or "").strip()
  ]
  return sorted(submitted, key=lambda r: r.get("submitted_at") or "")


def decide_mode(requested_mode, reviews, head_sha, max_auto_reviews):
  """Decides the review mode before looking at diffs.

  Args:
    requested_mode: `full` for explicit requests, `auto` for automatic triggers.
    reviews: The bot's submitted summary reviews, oldest first.
    head_sha: Current head commit of the pull request.
    max_auto_reviews: Maximum number of bot summary reviews on a pull request before automatic reviews stop.

  Returns:
    A tuple `(mode, reason)`. `mode` is `skip`, `full`, or `compare` (compare the diffs to choose between `skip`,
    `incremental`, and `full`).
  """
  if requested_mode == MODE_FULL:
    return MODE_FULL, "review explicitly requested"
  if len(reviews) >= max_auto_reviews:
    return MODE_SKIP, f"automatic review limit reached ({len(reviews)} >= {max_auto_reviews} reviews)"
  if not reviews:
    return MODE_FULL, "no earlier review"
  previous_sha = reviews[-1].get("commit_id")
  if previous_sha == head_sha:
    return MODE_SKIP, f"head {head_sha} was already reviewed"
  return "compare", f"last review was on {previous_sha}"


def extract_verdicts(text):
  """Extracts the per-thread verdicts from the model's final response.

  The prompt asks the model to end its response with a fenced block:

      ```json maxtext-review-verdicts
      {"verdicts": [{"thread_id": "...", "verdict": "fixed", "reason": "..."}]}
      ```

  Args:
    text: The model's final response.

  Returns:
    The list of verdict dicts, or None if no valid block was found.
  """
  if not text:
    return None
  blocks = re.findall(r"```json[ \t]+" + VERDICTS_TAG + r"[ \t]*\n(.*?)\n```", text, flags=re.DOTALL)
  if not blocks:
    return None
  try:
    data = json.loads(blocks[-1])
  except json.JSONDecodeError:
    return None
  verdicts = data.get("verdicts") if isinstance(data, dict) else None
  if not isinstance(verdicts, list):
    return None
  return [v for v in verdicts if isinstance(v, dict)]


def sanitize_reason(reason):
  """Makes model-written text safe to post under the bot identity.

  Keeps only the first line, and removes or neutralizes HTML, mentions, links, images, and code fences, so that the
  reply is a single short sentence of plain text (inline code spans are kept).

  Args:
    reason: The model-written reason.

  Returns:
    The sanitized text.
  """
  text = re.sub(r"<!--.*?-->", "", str(reason or ""), flags=re.DOTALL).strip()
  text = text.splitlines()[0] if text else ""
  # No HTML tags, and no Markdown links or images: `[`, `]`, `<`, and `>` are escaped.
  text = re.sub(r"<[^>]*>", "", text)
  text = re.sub(r"([\[\]<>])", r"\\\1", text)
  # No code fences (e.g. `suggestion` blocks): a run of backticks becomes a single one.
  text = re.sub(r"`{2,}", "`", text)
  text = " ".join(text.split())
  # A zero-width space after `@` prevents mentions; GitHub renders it invisibly.
  text = text.replace("@", "@\u200b")
  if len(text) > MAX_REASON_CHARS:
    text = text[: MAX_REASON_CHARS - 1] + "…"
  return text


def plan_thread_actions(verdicts, live_threads, allowed_thread_ids, bot_login, head_sha, *, pr_author):
  """Turns the model's verdicts into replies and resolutions, enforcing the guardrails.

  Guardrails:
  * Only threads from `allowed_thread_ids` (the bot's own threads, unresolved when the review started) that are still
    unresolved are touched, so human threads and threads resolved by a person are never changed.
  * At most one reply per thread per run (the first verdict for a thread wins), and threads are never reopened.
  * No "still applies" reply while the bot already has the last word in the thread.
  * Dispute verdicts apply only after a reply from the pull request author or a repository maintainer (see
    `is_trusted_reply`), and a `disputed_reject` reply is posted at most once per thread.

  Args:
    verdicts: Verdict dicts from `extract_verdicts`.
    live_threads: Current review threads (normalized GraphQL nodes).
    allowed_thread_ids: Thread IDs listed in `previous_threads.json`.
    bot_login: The bot's login.
    head_sha: The commit the verdicts refer to.
    pr_author: Login of the pull request author.

  Returns:
    A list of action dicts with `thread_id`, `verdict`, `reply_to` (comment database ID), `body`, and `resolve`
    (bool).
  """
  live_by_id = {t["id"]: t for t in live_threads}
  short_sha = head_sha[:7]
  seen = set()
  actions = []
  for verdict in verdicts:
    thread_id = verdict.get("thread_id")
    kind = verdict.get("verdict")
    if thread_id in seen or thread_id not in allowed_thread_ids or kind not in VERDICTS:
      continue
    seen.add(thread_id)
    thread = live_by_id.get(thread_id)
    if thread is None or thread.get("isResolved"):
      continue
    comments = thread["comments"]["nodes"]
    if not comments:
      continue
    authors = [(c.get("author") or {}).get("login") for c in comments]
    bot_has_last_word = is_same_login(authors[-1], bot_login)
    trusted_reply = any(is_trusted_reply(c, bot_login, pr_author) for c in comments[1:])
    already_rejected = any(
        FOLLOWUP_MARKER.format(verdict=VERDICT_DISPUTED_REJECT) in (c.get("body") or "") for c in comments
    )
    reason = sanitize_reason(verdict.get("reason"))
    marker = FOLLOWUP_MARKER.format(verdict=kind)
    body = None
    resolve = False
    if kind == VERDICT_STILL_OPEN_UNCHANGED:
      continue
    elif kind == VERDICT_STILL_OPEN_CHANGED:
      if bot_has_last_word:
        continue
      body = f"Still applies after `{short_sha}`: {reason}"
    elif kind == VERDICT_FIXED:
      body = f"Addressed in `{short_sha}`: {reason}"
      resolve = True
    elif kind == VERDICT_DISPUTED_ACCEPT:
      if not trusted_reply:
        continue
      body = f"Thanks, that makes sense: {reason}"
      resolve = True
    elif kind == VERDICT_DISPUTED_REJECT:
      if bot_has_last_word or not trusted_reply or already_rejected:
        continue
      body = f"I still think this applies: {reason} Leaving this open for a human reviewer to decide."
    actions.append(
        {
            "thread_id": thread_id,
            "verdict": kind,
            "reply_to": comments[0]["databaseId"],
            "body": f"{body}\n\n{marker}",
            "resolve": resolve,
        }
    )
  return actions


# ---------------------------------------------------------------------------------------------------------------------
# Subcommands.
# ---------------------------------------------------------------------------------------------------------------------


def write_output(name, value):
  """Writes a GitHub Actions step output."""
  output_path = os.environ.get("GITHUB_OUTPUT")
  if output_path:
    with open(output_path, "a", encoding="utf-8") as output:
      output.write(f"{name}={value}\n")
  print(f"{name}={value}")


def write_json(path, data):
  """Writes `data` as indented JSON that is safe to embed in a Markdown code fence.

  The prompt embeds these files in fixed ```json fences, and they contain untrusted text (e.g., PR comment bodies and
  file names). Backticks are escaped as `\\u0060`, which is equivalent JSON, so that no string can close the fence.

  Args:
    path: Output file path.
    data: JSON-serializable data.
  """
  with open(path, "w", encoding="utf-8") as f:
    f.write(json.dumps(data, indent=2).replace("`", "\\u0060"))
    f.write("\n")


def post_cap_comment_once(repository, number, max_auto_reviews):
  """Posts the automatic review limit comment, unless the pull request already has one."""
  comments = gh_api(f"repos/{repository}/issues/{number}/comments?per_page=100", paginate=True) or []
  if any(CAP_MARKER in (c.get("body") or "") for c in comments):
    return
  body = (
      f"🤖 This pull request reached the limit of {max_auto_reviews} automatic MaxText reviews, so new pushes will "
      "not be reviewed automatically. Comment `@gemini-cli /review` or add the `gemini-review` label to request "
      f"another review.\n\n{CAP_MARKER}"
  )
  gh_api(f"repos/{repository}/issues/{number}/comments", method="POST", fields={"body": body})


def delete_pending_reviews(repository, number, reviews, bot_login):
  """Deletes the bot's pending reviews on the pull request, best effort.

  A run that is cancelled after creating its pending review and before submitting it leaves the pending review behind.
  The next run could then not create its own, and would add to and publish the stale one, so it is deleted first.
  Runs for one pull request never overlap, so this never deletes the pending review of a running review.

  Args:
    repository: `owner/repo`.
    number: Pull request number.
    reviews: Review entries from the GitHub REST API, which include only the authenticated user's pending reviews.
    bot_login: The bot's login.
  """
  for review in reviews:
    if review.get("state") != "PENDING" or not is_same_login((review.get("user") or {}).get("login"), bot_login):
      continue
    review_id = review.get("id")
    try:
      gh_api(f"repos/{repository}/pulls/{number}/reviews/{review_id}", method="DELETE")
      print(f"Deleted stale pending review {review_id}.")
    except GitHubError as error:
      print(f"::warning::Could not delete stale pending review {review_id}: {error}")


def gather_review_state(args):
  """Reads the pull request through the GitHub API and decides the review mode.

  Args:
    args: Parsed `prepare` arguments.

  Returns:
    A dict with `mode`, `reason`, `head_sha`, `previous_review_sha`, `changed_files`, `delta_text`, `threads` (prompt
    view of the bot's unresolved threads), and `other_findings`.

  Raises:
    GitHubError: If a GitHub API call fails.
  """
  repository = args.repository
  owner, repo = repository.split("/", 1)
  number = args.pull_number

  pull = gh_api(f"repos/{repository}/pulls/{number}")
  head_sha = pull["head"]["sha"]
  base_ref = pull["base"]["ref"]
  pr_author = (pull.get("user") or {}).get("login")
  all_reviews = gh_api(f"repos/{repository}/pulls/{number}/reviews?per_page=100", paginate=True) or []
  reviews = bot_reviews(all_reviews, args.bot_login)
  all_threads = fetch_review_threads(owner, repo, number)
  open_threads = select_bot_threads(all_threads, args.bot_login)
  threads = open_threads[:MAX_PREVIOUS_THREADS]
  if len(open_threads) > MAX_PREVIOUS_THREADS:
    print(
        f"::notice::{len(open_threads) - MAX_PREVIOUS_THREADS} unresolved bot thread(s) beyond the first "
        f"{MAX_PREVIOUS_THREADS} get no follow-up in this run."
    )

  mode, reason = decide_mode(args.mode, reviews, head_sha, args.max_auto_reviews)
  delta_text = NO_DELTA_TEXT
  previous_sha = reviews[-1].get("commit_id") if reviews else None
  changed_files = []

  if mode == "compare":
    old_files = fetch_compare_files(repository, base_ref, previous_sha)
    new_files = fetch_compare_files(repository, base_ref, head_sha)
    if old_files is None or new_files is None:
      mode, reason = MODE_FULL, "could not compare with the last reviewed commit"
    else:
      delta = compute_delta(old_files, new_files)
      if not delta:
        mode, reason = MODE_SKIP, f"the pull request diff is unchanged since {previous_sha}"
      else:
        delta_text = render_delta(delta, previous_sha, head_sha)
        changed_files = [{"filename": d["filename"], "kind": d["kind"]} for d in delta]
        if len(delta_text) > MAX_DELTA_CHARS:
          mode, reason = MODE_FULL, f"the delta since {previous_sha} is too large for an incremental review"
          delta_text = NO_DELTA_TEXT
        else:
          mode, reason = MODE_INCREMENTAL, f"{len(delta)} file(s) changed since {previous_sha}"

  if mode != MODE_SKIP:
    delete_pending_reviews(repository, number, all_reviews, args.bot_login)
  elif args.mode != MODE_FULL and len(reviews) >= args.max_auto_reviews:
    try:
      post_cap_comment_once(repository, number, args.max_auto_reviews)
    except GitHubError as error:
      print(f"::warning::Could not post the automatic review limit comment: {error}")

  return {
      "mode": mode,
      "reason": reason,
      "head_sha": head_sha,
      "previous_review_sha": previous_sha,
      "changed_files": changed_files,
      "delta_text": delta_text,
      "threads": [summarize_thread(t, args.bot_login, pr_author) for t in threads],
      "other_findings": other_bot_findings(all_threads, args.bot_login),
  }


def fallback_review_state(requested_mode, error):
  """Returns the review state to use when `gather_review_state` fails.

  Explicit requests degrade to a full review without earlier review state, so they never depend on this script.
  Automatic reviews are skipped, so that a GitHub API outage neither fails the run nor bypasses the limit.

  Args:
    requested_mode: `full` or `auto`.
    error: The exception raised by `gather_review_state`.

  Returns:
    A dict like the one returned by `gather_review_state`.
  """
  mode = MODE_FULL if requested_mode == MODE_FULL else MODE_SKIP
  return {
      "mode": mode,
      "reason": f"could not read the earlier review state ({type(error).__name__}: {error})",
      "head_sha": "",
      "previous_review_sha": None,
      "changed_files": [],
      "delta_text": NO_DELTA_TEXT,
      "threads": [],
      "other_findings": [],
  }


def prepare(args):
  """Decides the review mode and writes the prompt context files."""
  os.makedirs(args.output_dir, exist_ok=True)
  try:
    result = gather_review_state(args)
  except (GitHubError, OSError, KeyError, IndexError, TypeError, AttributeError, ValueError) as error:
    # ValueError includes json.JSONDecodeError; TypeError and AttributeError cover unexpected `null` API responses.
    print(f"::warning::Could not read the earlier review state, so it is ignored: {error}")
    result = fallback_review_state(args.mode, error)

  print(f"Review mode: {result['mode']} ({result['reason']})")
  write_json(
      os.path.join(args.output_dir, "review_mode.json"),
      {
          "mode": result["mode"],
          "reason": result["reason"],
          "head_sha": result["head_sha"],
          "previous_review_sha": result["previous_review_sha"],
          "changed_files": result["changed_files"],
      },
  )
  with open(os.path.join(args.output_dir, "review_delta.md"), "w", encoding="utf-8") as f:
    f.write(result["delta_text"] + "\n")
  write_json(os.path.join(args.output_dir, "previous_threads.json"), result["threads"])
  write_json(os.path.join(args.output_dir, "other_findings.json"), result["other_findings"])
  write_output("mode", result["mode"])
  write_output("head_sha", result["head_sha"])
  write_output("previous_thread_count", len(result["threads"]))


def apply(args):
  """Applies the model's per-thread verdicts to the bot's own threads."""
  repository = args.repository
  owner, repo = repository.split("/", 1)
  number = args.pull_number
  with open(os.path.join(args.output_dir, "previous_threads.json"), encoding="utf-8") as f:
    allowed_thread_ids = {t["thread_id"] for t in json.load(f)}
  if not allowed_thread_ids:
    print("No earlier bot threads; nothing to apply.")
    return

  verdicts = extract_verdicts(os.environ.get("GEMINI_RESPONSE", ""))
  if verdicts is None:
    print("::warning::No valid verdicts block in the model response; leaving earlier review threads unchanged.")
    return

  try:
    pull = gh_api(f"repos/{repository}/pulls/{number}")
    pr_author = (pull.get("user") or {}).get("login")
    live_threads = fetch_review_threads(owner, repo, number)
  except GitHubError as error:
    print(f"::warning::Could not read the review threads; leaving earlier review threads unchanged: {error}")
    return
  actions = plan_thread_actions(
      verdicts, live_threads, allowed_thread_ids, args.bot_login, args.head_sha, pr_author=pr_author
  )
  failures = 0
  for action in actions:
    try:
      gh_api(
          f"repos/{repository}/pulls/{number}/comments/{action['reply_to']}/replies",
          method="POST",
          fields={"body": action["body"]},
      )
      if action["resolve"]:
        gh_api("graphql", method="POST", fields={"query": RESOLVE_MUTATION, "threadId": action["thread_id"]})
      print(f"Thread {action['thread_id']}: {action['verdict']} (resolved={action['resolve']})")
    except GitHubError as error:
      failures += 1
      print(f"::warning::Could not update thread {action['thread_id']}: {error}")
  print(f"Applied {len(actions) - failures} of {len(actions)} thread action(s) from {len(verdicts)} verdict(s).")


def parse_max_auto_reviews(value):
  """Parses the per-pull request limit of automatic reviews, falling back to the default if it is invalid."""
  if value is None or not str(value).strip():
    return DEFAULT_MAX_AUTO_REVIEWS
  try:
    parsed = int(str(value).strip())
  except ValueError:
    parsed = -1
  if parsed < 0:
    print(f"::warning::Invalid GEMINI_MAX_AUTO_REVIEWS {value!r}; using {DEFAULT_MAX_AUTO_REVIEWS}.")
    return DEFAULT_MAX_AUTO_REVIEWS
  return parsed


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  subparsers = parser.add_subparsers(dest="command", required=True)
  for name in ("prepare", "apply"):
    sub = subparsers.add_parser(name)
    sub.add_argument("--repository", default=os.environ.get("REPOSITORY"))
    sub.add_argument("--pull-number", type=int, default=os.environ.get("PULL_REQUEST_NUMBER"))
    sub.add_argument("--bot-login", default=os.environ.get("BOT_LOGIN") or "github-actions[bot]")
    sub.add_argument("--output-dir", default=".gemini")
  subparsers.choices["prepare"].add_argument("--mode", choices=["auto", MODE_FULL], default=MODE_FULL)
  subparsers.choices["prepare"].add_argument(
      "--max-auto-reviews", type=parse_max_auto_reviews, default=os.environ.get("MAX_AUTO_REVIEWS") or ""
  )
  subparsers.choices["apply"].add_argument("--head-sha", required=True)
  args = parser.parse_args(argv)

  if not args.repository or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository):
    parser.error(f"invalid --repository: {args.repository!r}")
  if not args.pull_number or args.pull_number <= 0:
    parser.error(f"invalid --pull-number: {args.pull_number!r}")

  if args.command == "prepare":
    prepare(args)
  else:
    apply(args)
  return 0


if __name__ == "__main__":
  sys.exit(main())
