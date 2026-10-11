# Teaser video

Renders `assets/video/agima_teaser_v3.mp4`, the 46-second teaser on the project page.

```bash
pip install -r tools/teaser/requirements.txt   # plus ffmpeg on PATH
python tools/teaser/make_teaser_v3.py                 # writes assets/video/agima_teaser_v3.mp4
python tools/teaser/make_teaser_v3.py out.mp4         # or to another file
python tools/teaser/make_teaser_v3.py x stills DIR    # preview stills of every scene into DIR
```

A full render takes about 2–3 minutes. Every frame is drawn with Pillow at 2560×1440 and
downsampled to 1280×720, then piped to ffmpeg (H.264, 30 fps).

## Files

- `make_teaser_v3.py`: the current teaser. Scenes, in order:
  - wall of imagined frames
  - one omni model
  - Sokoban imagine / branch / act rollout
  - Pacman RL rollout, with the route-order trap panel
  - easy-to-hard results
  - imagined vs real wipe
  - end card
- `make_teaser_v2.py`, `make_teaser.py`: earlier cuts. v3 imports their data loading and drawing helpers, so all three are needed.
- `fonts/`: Gelasio and Inter (SIL Open Font License, from google/fonts).
- `inputs/`: the exported example frames the teaser reads. These are copies of
  `unified_world_model/artifacts/…`:
  - `pacman_near_far_food_vanilla_vs_forward_ep034_branch_start_20260924/`: the Pacman route-order episode: ground truth for both candidates, the linear rollout, and the forward-thinking imagined frames.
  - `sokoban_train_data_forward_branch_examples_20260925/`: the branching and forward-thinking training examples.
  - `sokoban_branching_checkpoint_examples_20260926/…/step_0016.png`: one solved board, used for the box-on-goal sprite.

Everything else comes from the site itself (`assets/images`, `assets/gifs`), so the teaser stays in
sync with the page's figures and GIFs.
