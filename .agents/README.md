# MaxText agent skills

This directory contains reusable workflows for AI coding assistants working with
MaxText. Skills use the [Agent Skills format](https://agentskills.io/specification)
and are maintained in `.agents/skills/`.

## Layout

```text
.agents/
├── README.md
└── skills/
    └── <skill-name>/
        └── SKILL.md
```

Each skill has a `SKILL.md` with YAML `name` and `description` fields, followed by
workflow instructions. Use lowercase kebab-case for skill names, such as
`model-bringup`, and match the directory name to the frontmatter `name`.
Add `references/`, `scripts/`, or `assets/` inside a skill only when it needs
supporting documentation, executable helpers, or templates.

## Using the skills

- **Codex:** discovers repository skills in `.agents/skills/`.
- **Claude Code:** use the explicit file prompt or personal installation below.
- **GitHub Copilot:** supports `.agents/skills/` as a repository
  skill locations; availability depends on the Copilot client and version.

Ask your assistant to use a skill by name for the relevant task. These workflows
assume access to a MaxText checkout. To use a skill from another workspace, copy
the complete skill directory into your assistant's supported skill location and
provide the path to the MaxText checkout.

### Connecting Claude Code

For use without installation, ask Claude from the MaxText checkout:

```text
Read .agents/skills/model-bringup/SKILL.md and use it to help bring up my model.
```

For automatic discovery, optionally link a skill into your personal Claude skills
directory. Run this from the MaxText repository root:

```sh
mkdir -p ~/.claude/skills
ln -s "$PWD/.agents/skills/model-bringup" ~/.claude/skills/model-bringup
```

This keeps the repository's only skill directory at `.agents/skills/`. The
personal link makes the skill available across projects; provide the MaxText
checkout path when working elsewhere. If you move the checkout, update the link.

See the discovery documentation for
[Codex](https://developers.openai.com/codex/skills/),
[Claude Code](https://code.claude.com/docs/en/skills), and
[GitHub Copilot](https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/customize-cloud-agent/add-skills).

## Contributing a skill

1. Create `.agents/skills/<skill-name>/SKILL.md` with a description explaining when
   the workflow applies.
2. Keep instructions specific to MaxText and independent of assistant-specific
   commands or tools. Link to existing project documentation where possible.
3. Check the frontmatter and documentation links. Exercise any
   executable helpers before submitting them.

Resolve project file paths relative to the MaxText checkout, not the directory
where the skill happens to be installed. Keep repository-wide development
conventions separate from task-specific skill instructions.
