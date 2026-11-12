`uv run optbench board render` draws the PNG and WebP board thumbnails from `results/board/index.json`.
Those files are left out of git.
The `board plots` job uploads them as the `board-plots` artifact on a pull request and on a push to `main`.
The pages build draws the same thumbnails into the published site, under `_static/board/`.
