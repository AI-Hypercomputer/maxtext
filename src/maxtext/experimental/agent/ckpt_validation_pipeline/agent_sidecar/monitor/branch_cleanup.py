# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prunes abandoned Overwatch agent branches older than a specified number of days."""

import argparse
import datetime
import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BRANCH_PREFIX = "fix-validation-pipeline-"


def prune_abandoned_branches(days: int, dry_run: bool = False):
  """Finds and prunes remote git branches matching fix-validation-pipeline-* older than `days`."""
  logger.info("Scanning for remote branches matching prefix '%s' older than %s days...", BRANCH_PREFIX, days)

  try:
    # List remote branches matching prefix
    cmd = ["git", "branch", "-r", "--list", f"origin/{BRANCH_PREFIX}*"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    branches = [b.strip() for b in result.stdout.splitlines() if b.strip()]

    if not branches:
      logger.info("No remote branches matching '%s*' found.", BRANCH_PREFIX)
      return

    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
    pruned_count = 0

    for branch_ref in branches:
      # Extract raw branch name without origin/
      branch_name = branch_ref.replace("origin/", "", 1)
      if not branch_name.startswith(BRANCH_PREFIX):
        continue  # Strict safety check so we NEVER delete manual branches

      try:
        # Get commit timestamp of head commit
        ts_cmd = ["git", "log", "-1", "--format=%cI", branch_ref]
        ts_res = subprocess.run(ts_cmd, capture_output=True, text=True, check=True)
        commit_iso = ts_res.stdout.strip()
        commit_dt = datetime.datetime.fromisoformat(commit_iso)

        if commit_dt < cutoff:
          logger.info("Branch %s is abandoned (last commit %s < cutoff %s).", branch_name, commit_iso, cutoff.isoformat())
          if dry_run:
            logger.info("[DRY RUN] Would delete remote branch: origin/%s", branch_name)
          else:
            del_cmd = ["git", "push", "origin", "--delete", branch_name]
            subprocess.run(del_cmd, capture_output=True, text=True, check=True)
            logger.info("Successfully pruned abandoned branch: origin/%s", branch_name)
            pruned_count += 1
        else:
          logger.info("Keeping branch %s (active within %s days).", branch_name, days)
      except Exception as e:
        logger.error("Failed to prune branch %s: %s", branch_ref, e)

    logger.info("Branch cleanup completed. Total pruned: %s", pruned_count)

  except Exception as e:
    logger.error("Error during branch cleanup: %s", e)
    sys.exit(1)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Prune abandoned agent branches.")
  parser.add_argument("--days", type=int, default=14, help="Inactivity threshold in days (default: 14).")
  parser.add_argument("--dry-run", action="store_true", help="Print branches to delete without executing.")
  args = parser.parse_args()

  prune_abandoned_branches(args.days, args.dry_run)
