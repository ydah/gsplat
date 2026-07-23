# Migrating from Python gsplat

The Ruby API keeps upstream operation names and tensor layouts where Numo::NArray can represent them
directly. The main changes are Ruby keyword syntax, an explicit lightweight autograd wrapper, and a
CPU-only dense execution model.

## Rendering call

Python:

```python
rendered, alphas, meta = rasterization(
    means, quats, scales, opacities, colors, viewmats, Ks, width, height,
    render_mode="RGB", packed=False,
)
```

Ruby:

```ruby
rendered, alphas, meta = Gsplat.rasterization(
  means: means,
  quats: quats,
  scales: scales,
  opacities: opacities,
  colors: colors,
  viewmats: viewmats,
  ks: intrinsics,
  width: width,
  height: height,
  render_mode: "RGB",
  packed: false
)
```

`Ks` becomes the idiomatic keyword `ks`. Hash keys in metadata are symbols, for example
`meta[:means2d]`, `meta[:radii]`, and `meta[:isect_offsets]`.

## Tensor and autograd mapping

| PyTorch | Ruby |
|---|---|
| `torch.float32` / `torch.float64` | `Numo::SFloat` / `Numo::DFloat` |
| `tensor.requires_grad_(True)` | `Autograd::Variable.new(array, requires_grad: true)` |
| `tensor.detach()` | `variable.data` |
| `tensor.grad` | `variable.grad` |
| `optimizer.zero_grad()` | `optimizer.zero_grad!` or `variable.zero_grad!` |
| `with torch.no_grad()` | `Gsplat::Autograd.no_grad { ... }` |
| `loss.backward()` | `loss.backward` |
| `output.backward(weight)` | `output.backward(weight)` |

Numo arrays do not carry device or graph state. Only `Autograd::Variable` does. Mixed calls return
Variables when any differentiable input requires gradients, otherwise they return raw Numo arrays.

The package supports camera batches `[C,...]`, not arbitrary leading batch dimensions. It uses dense
`[C,N,...]` intermediates. `packed: true` is accepted for call compatibility but warns and executes
dense mode; sparse gradients are unavailable.

## Geometry and activation

- Quaternion order is `wxyz`.
- Rendering takes activated `scales > 0` and `opacities` in the 0–1 range.
- Trainer parameter hashes store `:scales` as log-scales and `:opacities` as logits, matching the
  optimization convention in the upstream examples.
- Direct `covars [N,3,3]` or packed symmetric `covars [N,6]` replace `quats`/`scales` on the normal
  2D raster path.
- Views are world-to-camera `[C,4,4]`; camera forward is positive Z.

## Render modes and metadata

The following modes are available: `RGB`, `D`, `ED`, `RGB+D`, `RGB+ED`, `d`, `Ed`, `RGB-d`, and
`RGB-Ed`. Uppercase depth uses projected Gaussian depth; lowercase depth is per-ray hit distance.

`absgrad: true` updates `meta[:means2d_absgrad]` after backward. World-space normals requested through
`with_eval3d: true, return_normals: true` are returned as `meta[:normals]`.

## Optimizers and strategies

An upstream parameter dictionary maps naturally to a Ruby symbol-keyed Hash:

```ruby
params = {
  means: means_variable,
  quats: quats_variable,
  scales: log_scales_variable,
  opacities: logits_variable,
  sh0: direct_sh_variable,
  shN: remaining_sh_variable
}
```

`Gsplat::Strategy::Default` and `Gsplat::Strategy::MCMC` use the same lifecycle around backward. All
structural edits must go through `Gsplat::Strategy::Ops` so parameter rows and Adam moments remain
aligned. `Gsplat::Optim::SelectiveAdam` accepts a first-axis visibility mask for visible-only updates.

## IO mapping

| Python workflow | Ruby API |
|---|---|
| COLMAP parser | `Gsplat::IO::Colmap.read` |
| Inria PLY export/import | `Gsplat::IO::Ply.write` / `.read` |
| training checkpoint | `Gsplat::IO::Checkpoint.save` / `.restore!` |
| NumPy fixture exchange | `Gsplat::IO::Npy.read`, `.write`, `.read_npz`, `.write_npz` |
| PNG parameter compression | `Gsplat::Compression::Png#compress` / `#decompress` |

NPY/NPZ supports the dtypes used by this project and C-order arrays. It intentionally does not load
arbitrary NumPy object arrays or Fortran-order payloads.

## Unsupported upstream options

There is no CUDA device backend, distributed rendering, arbitrary leading batch support, packed/sparse
execution, F-theta/lidar cameras, rolling shutter, custom rays, unscented-transform projection,
renderer configs, extra signals, or LPIPS in version 1.0.

The current 2DGS auxiliary buffers are compatible in shape but approximate upstream ray-splat
geometry. Eval3d is pinhole/classic only and uses a numerical geometry VJP. Check
`docs/DECISIONS.md` when exact cross-implementation parity is required.
