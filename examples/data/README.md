# Example data

This directory contains a deterministic 16-Gaussian scene for the runnable examples:

- `colmap/` is a three-view COLMAP text dataset with 16×16 PNG images.
- `splats.ply` is the corresponding Inria-layout Gaussian model.

Regenerate both assets from the repository root:

```bash
bundle exec ruby examples/generate_sample_data.rb
```

The scene is generated entirely by Gsplat and contains no third-party data.
