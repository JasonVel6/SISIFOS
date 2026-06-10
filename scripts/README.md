# scripts/

## render_phased.sh — phased / resumable rendering

Render a SISIFOS trajectory in bounded **banks** so you can stop (Ctrl-C or a full
shutdown) and resume later without losing work or re-rendering finished frames.

```bash
scripts/render_phased.sh <config.json> <tag> [bank_size]      # bank_size default 500
```
- `<config.json>` — a normal render config (as for `main.py --config_path`).
- `<tag>` — dataset name; output goes to `renders/_phased/<tag>/` and is symlinked
  to `$DATASETS/<tag>` (default `datasets/<tag>`).
- Re-run the **same command** to resume — it scans `images/` and renders only the
  missing frames. Run under `nohup`/`tmux` for long jobs; tail
  `renders/_phased/<tag>/render.log` for progress.

Env vars (optional): `SIS` (repo root, auto-detected), `DATASETS`, `BLENDER`.

### How it works
- Frames are written per-frame, so completed work always persists.
- The trajectory is deterministic in `(seed, tstep)`, so banks stitch exactly.
- `SISIFOS_OUTPUT_DIR` (honored in `main.py:run_sweep`) pins every bank to one
  folder; trajectory regen only `makedirs(exist_ok=True)`, so it never wipes frames.
- On completion it rebuilds the full-length `imgList.txt` from `camera_traj.csv`.

No `frames.mp4` is produced (the banked path skips video); make one after with
`ffmpeg -y -framerate 24 -i <agent>/images/frame_%04d.png -c:v libx264 -pix_fmt yuv420p <agent>/frames.mp4`.
