---
name: documentation-style
description: >
  Guide for writing narrative documentation and Python docstrings in this
  repository. Use this when writing, editing, or reviewing any documentation,
  README content, or docstrings.
---

# Documentation for MaxText

## Narrative documentation (.md files under the docs/ folder)

Follow the [Google Developer Documentation Style Guide](https://developers.google.com/style). Key rules are summarized below.

### Voice and tone

- Write in a conversational, friendly, and respectful tone: not formal, not frivolous.
- Sound like a knowledgeable friend: helpful and direct, not pedantic or pushy.
- Avoid buzzwords, jargon, exclamation marks, pop-culture references, and filler phrases like *please note* or *at this time*.
- Do not use *simply*, *easy*, *quickly*, *just* or similar words that minimize difficulty.
- Write for a global audience. Avoid culturally specific idioms or references.

### Grammar and language

- Use **second person** ("you"), not "we".
- Use **active voice**. Make clear who or what performs the action.
  - Recommended: *Send a query to the service.*
  - Not recommended: *The service is queried.*
- Use **present tense**.
- Use **standard American English** spelling and punctuation.
- Put **conditions and context before instructions**, not after.
  - Recommended: *To delete the document, click **Delete**.*
  - Not recommended: *Click **Delete** if you want to delete the document.*

### Formatting

- Use **numbered lists** for sequential steps; use **bulleted lists** for non-sequential items.
- Use **serial (Oxford) commas**.
- Put all code, commands, and code-related terms in **code font** (backticks) or in a code-block (if multiple lines). This includes file names, directory names, environment variable names, and UI elements that are not a single word.
- Put UI elements in **bold**.
- Use [descriptive link text](https://developers.google.com/style/cross-references#descriptive-link-text) — never "click here" or "this link".
- Avoid collapsed sections.
- Avoid complicated tables.

### Syntax

The MaxText documentation uses MyST Markdown, which supports additional syntax beyond standard Markdown. Use MyST syntax when it improves readability or functionality, including for admonitions, cross-references, and math. See the [MyST documentation](https://myst-parser.readthedocs.io/en/latest/) for details.

### Page title and headings

- Use **sentence case** for all headings and document titles.
- Use imperative verb forms vs gerunds. For example: Use "Set up the environment" vs. "Setting up the environment".
- Be descriptive, active, and unique (and keep it short).
- Include the most important keywords that are likely to be known to users into the title (and don’t overstuff the title with keywords; it should be human readable).
- Include secondary keywords in the introductory sentences.
- Don’t put links in titles; put them in intro sentences.

### Word list and product names

- Do not use Jax. Use JAX
- Do not use Maxtext, maxtext, or MAXTEXT. Use MaxText
- Do not use GCP or Google Cloud Platform. Use Google Cloud.
- Do not use GCE. Use Compute Engine.
- Do not use GCS. Use Cloud Storage.
- GKE is ok to use; prefer to spell out on first mention in a doc as Google Kubernetes Engine (GKE).
- Trillium is ok to use on first mention, in parentheses, after the version number. Thereafter, use the version number if needed.
- Hugging Face has a space and is capitalized.

### Structure and organization

- Put the most important information first.
- Use headings to help users quickly find the section they need. Most people scan when reading instead of reading every word.
- Avoid including information that is available in the docs. Link to it instead.
- Do not pre-announce upcoming content (e.g. avoid "In the next section, we will...").

### Introduction

After the title, provide 1-3 sentences describing why the reader should care about the topic. This is required for several reasons:
- Orients the reader to what the topic does and who it’s for
- Communicates what the user will accomplish with this topic
- Is optimized for SEO, GEO, and accessibility (no empty headings)

### Code blocks

Code blocks should be optimized for copy/paste. They should be atomic units. They should have brief introductory sentences describing what they do.

Optional: It is a good practice after a code block to have a sentence explaining what the code did, what success looks like, or what failure looks like if a command was unsuccessful. This can include what the expected output looks like, or can include an error message if the command was unsuccessful. This is an appropriate place to link to troubleshooting guidance for errors that may have occurred (be sparing).

---

## Python Docstrings

Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

### Format

```python
def example(param: int, other: str = "default") -> str:
    """One-line summary in imperative mood, no trailing period.

    Optional longer description. Explain non-obvious behavior, important
    caveats, or context that helps the reader understand the function's
    purpose — not just what it does mechanically.

    Args:
        param: Description of this parameter.
        other: Description of this parameter. Defaults to "default".

    Returns:
        Description of the return value.

    Raises:
        ValueError: When and why this error is raised.
    """
```

### Rules

- **First line**: imperative mood (e.g. *Compute*, *Return*, *Load*), no period, ≤ 79 characters.
- Separate the summary line from the body with a blank line.
- Include `Args`, `Returns`, and `Raises` sections whenever applicable.
- Use **type hints in the function signature**, not in the docstring body.
- Do not restate parameter names or types verbatim — describe their meaning and purpose.
- For `Args` entries that need more than one line, indent continuation lines by 4 spaces.
- Omit sections that don't apply (e.g. omit `Returns` for functions that return `None`).
- Class docstrings go on the class, not on `__init__`. Document `__init__` args in the class docstring `Args` section.
