# Changelog

## 2026-10-21

The board lists stock templates. `optbench check` rejects a template file named `*-tuned`, a template that sets `tuned = true`, and any `tuning = "tuned"`, on every list the board writes. A parameter that is not what the code ships is not a template. The removed template files and campaign files stay in git history.
