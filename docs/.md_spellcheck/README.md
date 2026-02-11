A git pre-commit hook to spell-check markdown files using GNU Aspell, http://aspell.net/

This is a minimally modified from https://github.com/mprpic/git-spell-check to be used
with the `pre-commit` framework, https://pre-commit.com/

To test, install `aspell`, and run

```
$ pre-commit run --all-files
MD checker via GNU aspell................................................Spell check failed on the following words:
-------------------------------------------------
anotherword
typooo
unstaged
File: test-docs/sample.md	on line: 15	Typo: anotherword
-------------------
File: test-docs/sample.md	on line: 7	Typo: typooo
-------------------
File: test-docs/clean.md	on line: 6	Typo: unstaged
File: test-docs/clean.md	on line: 7	Typo: unstaged
-------------------

Add any of the misspelled words into your project dictionary (/home/br/repos/test-aspell-hook/.aspell-project-dict.txt)?
  * a[ll]     (add all words into dict, continue with commit)
  * s[ome]    (add some words into dict, fix others, no commit)
  * i[gnore]  (add some words into dict, ignore rest, continue with commit)
  * n[one]    (no commit)

n
No words were added to your project dictionary.
Please fix remaining typos, use "git add" to add fixed files, and commit.
Failed
- hook id: aspell-md
- exit code: 1
```

To use on other projects, add an entry to the project's `.pre-commit-hooks-config.yaml`
