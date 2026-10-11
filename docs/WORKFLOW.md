# Daily Workflow

## One-time setup

```bash
git clone <your-repo-url> master-portfolio && cd master-portfolio
git config user.email "<the email attached to your GitHub account>"   # needed for the contribution graph
```

Commits only appear on your contribution graph when the author email matches a verified email on your
GitHub account and the commit lands on the default branch.

## Each day

1. Get the day's slice (code, tests, docs) and copy it into the project folder.
2. Read it, run it and make sure you can explain every design choice. Change anything you would do
   differently; the repo is yours, and being able to discuss it in an interview is the real goal.
3. Append a short entry to `projects/<NN>-<slug>/docs/devlog.md`.
4. From the repo root:

   ```bash
   tools/daily_commit.sh <slug> <day 1-7> "<short summary>"
   ```

   The script runs `ruff` and `pytest` for that project first, then commits and pushes.
5. On Day 7 also: tick the project off in the root `README.md` table, tag a release
   (`git tag <slug>-v0.1.0 && git push --tags`) and write the "what I learned" section in the devlog.

## Commit message format

```
feat(quantlab): day 3/7 - strategy interface and momentum strategies
fix(quantlab): handle zero-volume bars in resampling
docs(quantlab): add architecture diagram
```

## If you miss a day

Do not backfill with fake-dated commits. Shift the remaining slices of the project by a day (the
roadmap dates are a guide), or merge two slices into one day when you have the time. A genuine commit
history is worth more than a perfect streak.
