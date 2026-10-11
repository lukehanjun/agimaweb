# Agima project website

Static project page for *Planning with Agentic Imagination* (Agima), served with GitHub Pages
from the repository root (`main` branch, `/`). No build step: open `index.html` locally, or run
`python -m http.server` in the repo root and visit http://localhost:8000.

## Layout

- `index.html`: the page.
- `assets/css/styles.css`: styles (layout adapted from the [RATs project page](https://playful-rats.github.io/)).
- `assets/js/main.js`: the interactive parts of the page. It renders:
  - the results charts and tables (data is inline at the top of the file);
  - the action-chunk (K) animation and the Sokoban K viewer;
  - the Figure 1 highlight animation;
  - the BibTeX copy button.
- `assets/images/`: paper figures (WebP), planning-trap boards, and the K viewer frames (`kchunks/`).
- `assets/gifs/env/`: train/test rollouts shown in the Environments section.
- `assets/gifs/rollout/`: imagined-vs-real rollouts. The page doesn't show them directly; the teaser video is built from them.
- `assets/video/`: the teaser video and its poster frame.
- `tools/teaser/`: the script that renders the teaser video, with its fonts and input frames. See its README.

## Before release

Fill in the placeholders marked `TODO` in `index.html`:
- the Code / Paper / arXiv links (marked `data-todo`, shown slightly faded until set);
- the venue and year in the BibTeX entry.

The `og:image` and `og:video` tags use relative paths. Most link-preview services need absolute
URLs, so set them to the page's final address before sharing the link.
