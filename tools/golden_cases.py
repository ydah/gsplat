"""Golden case definitions and generators for gsplat 1.5.3."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable


SEED = 42
GSPLAT_VERSION = (1, 5, 3)


@dataclass(frozen=True)
class Case:
    """A reproducible golden-data case."""

    name: str
    family: str
    options: dict[str, Any]
    requires_cuda: bool = False


def all_cases() -> list[Case]:
    """Return the complete, stable case matrix."""
    cases = [
        Case("quat_covar_full", "quat", {"triu": False}),
        Case("quat_covar_triu", "quat", {"triu": True}),
        Case("proj_covars_c1_n1000", "proj_covars", {"cameras": 1, "count": 1_000}),
        *(Case(f"sh_deg{degree}", "sh", {"degree": degree}) for degree in range(5)),
    ]
    for camera_model in ("pinhole", "ortho", "fisheye"):
        for cameras, count in ((1, 1_000), (3, 1_000), (1, 10_000), (3, 10_000)):
            name = f"proj_{camera_model}_c{cameras}_n{count}"
            cases.append(Case(name, "proj", {"camera_model": camera_model, "cameras": cameras, "count": count}))
    cases.extend(
        [
            Case("isect_c3_n1000", "isect", {"cameras": 3, "count": 1_000}),
            Case("isect_c1_n10000", "isect", {"cameras": 1, "count": 10_000}),
            Case("raster_rgb", "raster", {"channels": 3}, requires_cuda=True),
            Case("raster_features8", "raster", {"channels": 8}, requires_cuda=True),
            Case("raster_features32", "raster", {"channels": 32}, requires_cuda=True),
            Case("raster_features40", "raster", {"channels": 40}, requires_cuda=True),
            Case("render_fisheye_distorted", "fisheye_distorted", {}, requires_cuda=True),
            Case("distance_order", "distance_order", {}),
            Case("hit_distance_modes", "hit_distance", {}),
            Case("raster_2dgs", "raster_2dgs", {}, requires_cuda=True),
            Case("strategy_default_masks", "strategy", {}),
            Case("relocation_mcmc", "relocation", {}, requires_cuda=True),
            Case("ssim_rgb", "ssim", {}),
        ]
    )
    render_matrix = (
        ("RGB", "classic", None),
        ("D", "classic", None),
        ("ED", "classic", None),
        ("RGB+D", "classic", None),
        ("RGB+ED", "classic", None),
        ("RGB", "antialiased", None),
        ("RGB+D", "antialiased", None),
        ("RGB+ED", "antialiased", None),
        ("RGB", "classic", 0),
        ("RGB", "classic", 3),
        ("RGB", "antialiased", 3),
        ("ED", "antialiased", 3),
    )
    for index, (mode, rasterize_mode, sh_degree) in enumerate(render_matrix):
        cases.append(
            Case(
                f"render_{index:02d}_{mode.lower().replace('+', '_')}_{rasterize_mode}_sh{sh_degree}",
                "render",
                {"render_mode": mode, "rasterize_mode": rasterize_mode, "sh_degree": sh_degree},
                requires_cuda=True,
            )
        )
    return cases


def load_runtime(device_name: str) -> dict[str, Any]:
    """Import the pinned reference stack only for a real generation run."""
    import importlib.metadata

    import gsplat
    import numpy as np
    import torch
    from gsplat.cuda import _torch_impl as reference

    version = importlib.metadata.version("gsplat")
    if version != ".".join(map(str, GSPLAT_VERSION)):
        raise RuntimeError(f"gsplat {version} loaded; expected 1.5.3")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    return {"gsplat": gsplat, "np": np, "torch": torch, "reference": reference, "device": torch.device(device_name)}


def generate(case: Case, runtime: dict[str, Any]) -> dict[str, Any]:
    """Generate one case using the pinned upstream implementation."""
    generators: dict[str, Callable[[Case, dict[str, Any]], dict[str, Any]]] = {
        "quat": _quat_case,
        "sh": _sh_case,
        "proj": _projection_case,
        "proj_covars": _projection_covars_case,
        "isect": _isect_case,
        "raster": _raster_case,
        "fisheye_distorted": _fisheye_distorted_case,
        "distance_order": _distance_order_case,
        "hit_distance": _hit_distance_case,
        "raster_2dgs": _raster_2dgs_case,
        "render": _render_case,
        "strategy": _strategy_case,
        "relocation": _relocation_case,
        "ssim": _ssim_case,
    }
    payload = generators[case.family](case, runtime)
    payload.update(
        {
            "meta_gsplat_version": runtime["torch"].tensor(GSPLAT_VERSION, dtype=runtime["torch"].int32),
            "meta_seed": runtime["torch"].tensor([SEED], dtype=runtime["torch"].int64),
        }
    )
    return payload


def to_numpy(payload: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    """Move tensors to CPU NumPy arrays while retaining supported scalar dtypes."""
    torch = runtime["torch"]
    converted = {}
    for name, value in payload.items():
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            converted[name] = value.detach().cpu().contiguous().numpy()
        else:
            converted[name] = runtime["np"].asarray(value)
    return converted


def _scene(runtime: dict[str, Any], count: int, cameras: int = 1) -> dict[str, Any]:
    torch, device = runtime["torch"], runtime["device"]
    means = torch.randn(count, 3, device=device, dtype=torch.float32) * 0.35
    means[:, 2] = torch.rand(count, device=device) * 2.0 + 1.5
    quats = torch.randn(count, 4, device=device, dtype=torch.float32)
    scales = torch.rand(count, 3, device=device, dtype=torch.float32) * 0.08 + 0.02
    opacities = torch.rand(count, device=device, dtype=torch.float32) * 0.8 + 0.1
    viewmats = torch.eye(4, device=device, dtype=torch.float32).repeat(cameras, 1, 1)
    if cameras > 1:
        viewmats[:, 0, 3] = torch.linspace(-0.15, 0.15, cameras, device=device)
    width, height = 64, 48
    ks = torch.zeros(cameras, 3, 3, device=device, dtype=torch.float32)
    ks[:, 0, 0] = 52.0
    ks[:, 1, 1] = 52.0
    ks[:, 0, 2] = width / 2.0
    ks[:, 1, 2] = height / 2.0
    ks[:, 2, 2] = 1.0
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "viewmats": viewmats,
        "ks": ks,
        "width": width,
        "height": height,
    }


def _quat_case(case: Case, runtime: dict[str, Any]) -> dict[str, Any]:
    torch, reference, device = runtime["torch"], runtime["reference"], runtime["device"]
    quats = torch.randn(128, 4, device=device, requires_grad=True)
    scales = (torch.rand(128, 3, device=device) + 0.1).requires_grad_()
    covars, precis = reference._quat_scale_to_covar_preci(quats, scales, triu=case.options["triu"])
    weight_covar = torch.randn_like(covars)
    weight_preci = torch.randn_like(precis) * 0.01
    grad_quats, grad_scales = torch.autograd.grad(
        (covars * weight_covar).sum() + (precis * weight_preci).sum(), (quats, scales)
    )
    return locals_payload(locals(), "quats", "scales", "covars", "precis", "weight_covar", "weight_preci",
                          "grad_quats", "grad_scales")


def _sh_case(case: Case, runtime: dict[str, Any]) -> dict[str, Any]:
    torch, reference, device = runtime["torch"], runtime["reference"], runtime["device"]
    degree = case.options["degree"]
    dirs = torch.randn(256, 3, device=device, requires_grad=True)
    coeffs = torch.randn(256, 25, 3, device=device, requires_grad=True)
    colors = reference._spherical_harmonics(degree, dirs, coeffs)
    weight_colors = torch.randn_like(colors)
    grad_coeffs, grad_dirs = torch.autograd.grad(
        (colors * weight_colors).sum(), (coeffs, dirs), allow_unused=True
    )
    if grad_dirs is None:
        grad_dirs = torch.zeros_like(dirs)
    return locals_payload(locals(), "dirs", "coeffs", "colors", "weight_colors", "grad_coeffs", "grad_dirs")


def _projection_case(case: Case, runtime: dict[str, Any]) -> dict[str, Any]:
    torch, reference = runtime["torch"], runtime["reference"]
    scene = _scene(runtime, case.options["count"], case.options["cameras"])
    means = scene["means"].requires_grad_()
    quats = scene["quats"].requires_grad_()
    scales = scene["scales"].requires_grad_()
    covars, _ = reference._quat_scale_to_covar_preci(quats, scales, compute_preci=False)
    radii, means2d, depths, conics, compensations = reference._fully_fused_projection(
        means, covars, scene["viewmats"], scene["ks"], scene["width"], scene["height"],
        calc_compensations=True, camera_model=case.options["camera_model"]
    )
    weight_means2d, weight_depths = torch.randn_like(means2d), torch.randn_like(depths)
    weight_conics, weight_compensations = torch.randn_like(conics), torch.randn_like(compensations)
    loss = (
        (means2d * weight_means2d).sum()
        + (depths * weight_depths).sum()
        + (conics * weight_conics).sum()
        + (compensations * weight_compensations).sum()
    )
    grad_means, grad_quats, grad_scales = torch.autograd.grad(loss, (means, quats, scales))
    payload = {**scene, **locals_payload(
        locals(), "radii", "means2d", "depths", "conics", "compensations", "weight_means2d",
        "weight_depths", "weight_conics", "weight_compensations", "grad_means", "grad_quats", "grad_scales"
    )}
    return tensor_payload(payload)


def _projection_covars_case(case: Case, runtime: dict[str, Any]) -> dict[str, Any]:
    torch, reference = runtime["torch"], runtime["reference"]
    scene = _scene(runtime, case.options["count"], case.options["cameras"])
    means = scene["means"].requires_grad_()
    covars, _ = reference._quat_scale_to_covar_preci(
        scene["quats"], scene["scales"], compute_preci=False
    )
    covars = covars.detach().requires_grad_()
    radii, means2d, depths, conics, compensations = reference._fully_fused_projection(
        means, covars, scene["viewmats"], scene["ks"], scene["width"], scene["height"],
        calc_compensations=True, camera_model="pinhole"
    )
    weight_means2d, weight_depths = torch.randn_like(means2d), torch.randn_like(depths)
    weight_conics, weight_compensations = torch.randn_like(conics), torch.randn_like(compensations)
    loss = (
        (means2d * weight_means2d).sum()
        + (depths * weight_depths).sum()
        + (conics * weight_conics).sum()
        + (compensations * weight_compensations).sum()
    )
    grad_means, grad_covars = torch.autograd.grad(loss, (means, covars))
    payload = {**scene, **locals_payload(
        locals(), "covars", "radii", "means2d", "depths", "conics", "compensations",
        "weight_means2d", "weight_depths", "weight_conics", "weight_compensations",
        "grad_means", "grad_covars"
    )}
    return tensor_payload(payload)


def _isect_case(case: Case, runtime: dict[str, Any]) -> dict[str, Any]:
    torch, reference, device = runtime["torch"], runtime["reference"], runtime["device"]
    cameras, count = case.options["cameras"], case.options["count"]
    width, height, tile_size = 64, 48, 16
    means2d = torch.rand(cameras, count, 2, device=device) * torch.tensor([width, height], device=device)
    radii = torch.randint(0, 20, (cameras, count, 2), device=device, dtype=torch.int32)
    depths = torch.rand(cameras, count, device=device) * 4.0 + 0.1
    tile_width, tile_height = math.ceil(width / tile_size), math.ceil(height / tile_size)
    tiles_per_gauss, isect_ids, flatten_ids = reference._isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = reference._isect_offset_encode(isect_ids, cameras, tile_width, tile_height)
    return locals_payload(locals(), "means2d", "radii", "depths", "tiles_per_gauss", "isect_ids",
                          "flatten_ids", "isect_offsets")


def _raster_case(case: Case, runtime: dict[str, Any]) -> dict[str, Any]:
    torch, device = runtime["torch"], runtime["device"]
    from gsplat.cuda._wrapper import (
        fully_fused_projection,
        isect_offset_encode,
        isect_tiles,
        rasterize_to_pixels,
    )

    channels = case.options["channels"]
    scene = _scene(runtime, 96)
    radii, projected, depths, conics, _ = fully_fused_projection(
        scene["means"], None, scene["quats"], scene["scales"], scene["viewmats"], scene["ks"],
        scene["width"], scene["height"], packed=False
    )
    means2d = projected.detach().requires_grad_()
    conics = conics.detach().requires_grad_()
    colors = torch.rand(1, 96, channels, device=device, requires_grad=True)
    opacities = scene["opacities"][None].detach().requires_grad_()
    backgrounds = torch.rand(1, channels, device=device, requires_grad=True)
    tile_size = 16
    tile_width, tile_height = math.ceil(scene["width"] / tile_size), math.ceil(scene["height"] / tile_size)
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)
    render_colors, render_alphas = rasterize_to_pixels(
        means2d, conics, colors, opacities, scene["width"], scene["height"], tile_size,
        isect_offsets, flatten_ids, backgrounds=backgrounds, absgrad=True
    )
    weight_colors, weight_alphas = torch.randn_like(render_colors), torch.randn_like(render_alphas)
    loss = (render_colors * weight_colors).sum() + (render_alphas * weight_alphas).sum()
    loss.backward()
    return locals_payload(
        locals(), "means2d", "conics", "colors", "opacities", "backgrounds", "radii", "depths",
        "tiles_per_gauss", "isect_ids", "flatten_ids", "isect_offsets", "render_colors",
        "render_alphas", "weight_colors", "weight_alphas"
    ) | {
        "grad_means2d": means2d.grad,
        "means2d_absgrad": means2d.absgrad,
        "grad_conics": conics.grad,
        "grad_colors": colors.grad,
        "grad_opacities": opacities.grad,
        "grad_backgrounds": backgrounds.grad,
    }


def _fisheye_distorted_case(_case: Case, runtime: dict[str, Any]) -> dict[str, Any]:
    torch, device = runtime["torch"], runtime["device"]
    from gsplat import rasterization

    scene = _scene(runtime, 64)
    colors = torch.rand(64, 3, device=device)
    radial_coeffs = torch.tensor([[0.01, -0.002, 0.0003, 0.0]], device=device)
    render_colors, render_alphas, meta = rasterization(
        scene["means"], scene["quats"], scene["scales"], scene["opacities"], colors,
        scene["viewmats"], scene["ks"], scene["width"], scene["height"],
        camera_model="fisheye", radial_coeffs=radial_coeffs, with_ut=True
    )
    return tensor_payload({
        **scene,
        "colors": colors,
        "radial_coeffs": radial_coeffs,
        "render_colors": render_colors,
        "render_alphas": render_alphas,
        "radii": meta["radii"],
    })


def _distance_order_case(_case: Case, runtime: dict[str, Any]) -> dict[str, Any]:
    torch, device = runtime["torch"], runtime["device"]
    camera_means = torch.tensor([[[1.0, 0.0, 2.0], [0.0, 0.0, 2.1]]], device=device)
    z_depths = camera_means[..., 2]
    euclidean_depths = torch.linalg.vector_norm(camera_means, dim=-1)
    return locals_payload(locals(), "camera_means", "z_depths", "euclidean_depths")


def _hit_distance_case(_case: Case, runtime: dict[str, Any]) -> dict[str, Any]:
    torch, device = runtime["torch"], runtime["device"]
    means = torch.tensor([[0.0, 0.0, 2.0]], device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    scales = torch.tensor([[0.5, 0.5, 0.5]], device=device)
    opacities = torch.tensor([0.5], device=device)
    colors = torch.tensor([[0.2, 0.4, 0.6]], device=device)
    render_alphas = torch.tensor([[[[0.5]]]], device=device)
    render_d = torch.tensor([[[[1.0]]]], device=device)
    render_ed = torch.tensor([[[[2.0]]]], device=device)
    render_rgb_d = torch.tensor([[[[0.1, 0.2, 0.3, 1.0]]]], device=device)
    render_rgb_ed = torch.tensor([[[[0.1, 0.2, 0.3, 2.0]]]], device=device)
    return locals_payload(
        locals(), "means", "quats", "scales", "opacities", "colors", "render_alphas",
        "render_d", "render_ed", "render_rgb_d", "render_rgb_ed"
    )


def _raster_2dgs_case(_case: Case, runtime: dict[str, Any]) -> dict[str, Any]:
    torch, device = runtime["torch"], runtime["device"]
    from gsplat import rasterization_2dgs

    scene = _scene(runtime, 64)
    colors = torch.rand(64, 3, device=device)
    outputs = rasterization_2dgs(
        scene["means"], scene["quats"], scene["scales"], scene["opacities"], colors,
        scene["viewmats"], scene["ks"], scene["width"], scene["height"], packed=False
    )
    names = (
        "render_colors", "render_alphas", "render_normals", "render_surface_normals",
        "render_distort", "render_median"
    )
    return tensor_payload({**scene, "colors": colors, **dict(zip(names, outputs[:6]))})


def _render_case(case: Case, runtime: dict[str, Any]) -> dict[str, Any]:
    torch, device = runtime["torch"], runtime["device"]
    from gsplat import rasterization

    scene = _scene(runtime, 64)
    means = scene["means"].requires_grad_()
    quats = scene["quats"].requires_grad_()
    scales = scene["scales"].requires_grad_()
    opacities = scene["opacities"].requires_grad_()
    sh_degree = case.options["sh_degree"]
    color_shape = (64, 25, 3) if sh_degree is not None else (64, 3)
    colors = torch.rand(*color_shape, device=device, requires_grad=True)
    backgrounds = torch.rand(1, 3, device=device)
    render_colors, render_alphas, meta = rasterization(
        means, quats, scales, opacities, colors, scene["viewmats"], scene["ks"],
        scene["width"], scene["height"], packed=False, backgrounds=backgrounds,
        render_mode=case.options["render_mode"], rasterize_mode=case.options["rasterize_mode"],
        sh_degree=sh_degree, absgrad=True
    )
    weight_colors, weight_alphas = torch.randn_like(render_colors), torch.randn_like(render_alphas)
    ((render_colors * weight_colors).sum() + (render_alphas * weight_alphas).sum()).backward()
    payload = {
        **scene,
        "colors": colors,
        "backgrounds": backgrounds,
        "render_colors": render_colors,
        "render_alphas": render_alphas,
        "weight_colors": weight_colors,
        "weight_alphas": weight_alphas,
        "grad_means": means.grad,
        "grad_quats": quats.grad,
        "grad_scales": scales.grad,
        "grad_opacities": opacities.grad,
        "grad_colors": colors.grad,
    }
    for key in ("radii", "means2d", "depths", "conics", "opacities", "tiles_per_gauss",
                "isect_ids", "flatten_ids", "isect_offsets"):
        if key in meta and meta[key] is not None:
            payload[f"meta_{key}"] = meta[key]
    if hasattr(meta["means2d"], "absgrad"):
        payload["meta_means2d_absgrad"] = meta["means2d"].absgrad
    return tensor_payload(payload)


def _strategy_case(_case: Case, runtime: dict[str, Any]) -> dict[str, Any]:
    torch, device = runtime["torch"], runtime["device"]
    count = 128
    grad2d = torch.rand(count, device=device) * 0.0005
    observations = torch.randint(1, 5, (count,), device=device).float()
    scales = torch.linspace(math.log(0.001), math.log(0.2), count, device=device)[:, None].repeat(1, 3)
    opacities = torch.linspace(-8.0, 4.0, count, device=device)
    mean_grad = grad2d / observations.clamp_min(1)
    high = mean_grad > 0.0002
    small = torch.exp(scales).amax(dim=-1) <= 0.01
    duplicate_mask = high & small
    split_mask = high & ~small
    prune_mask = torch.sigmoid(opacities) < 0.005
    return locals_payload(locals(), "grad2d", "observations", "scales", "opacities", "duplicate_mask",
                          "split_mask", "prune_mask")


def _relocation_case(_case: Case, runtime: dict[str, Any]) -> dict[str, Any]:
    torch, device = runtime["torch"], runtime["device"]
    from gsplat.relocation import compute_relocation

    opacities = torch.linspace(0.01, 0.9, 32, device=device)
    scales = torch.rand(32, 3, device=device) * 0.2 + 0.01
    ratios = torch.arange(1, 33, device=device, dtype=torch.float32).remainder(8).add(1)
    binoms = torch.zeros(51, 51, device=device)
    for n in range(51):
        for k in range(n + 1):
            binoms[n, k] = math.comb(n, k)
    new_opacities, new_scales = compute_relocation(opacities, scales, ratios, binoms)
    return locals_payload(locals(), "opacities", "scales", "ratios", "binoms", "new_opacities", "new_scales")


def _ssim_case(_case: Case, runtime: dict[str, Any]) -> dict[str, Any]:
    torch, device = runtime["torch"], runtime["device"]
    import torch.nn.functional as functional

    image_a = torch.rand(1, 3, 24, 24, device=device, requires_grad=True)
    image_b = torch.rand(1, 3, 24, 24, device=device, requires_grad=True)
    coordinates = torch.arange(11, device=device, dtype=torch.float32) - 5
    kernel = torch.exp(-(coordinates ** 2) / (2 * 1.5 ** 2))
    kernel = kernel / kernel.sum()
    window = (kernel[:, None] * kernel[None, :]).expand(3, 1, 11, 11)
    mu_a = functional.conv2d(image_a, window, padding=5, groups=3)
    mu_b = functional.conv2d(image_b, window, padding=5, groups=3)
    sigma_a = functional.conv2d(image_a * image_a, window, padding=5, groups=3) - mu_a.square()
    sigma_b = functional.conv2d(image_b * image_b, window, padding=5, groups=3) - mu_b.square()
    sigma_ab = functional.conv2d(image_a * image_b, window, padding=5, groups=3) - mu_a * mu_b
    ssim_map = ((2 * mu_a * mu_b + 0.01 ** 2) * (2 * sigma_ab + 0.03 ** 2)) / (
        (mu_a.square() + mu_b.square() + 0.01 ** 2) * (sigma_a + sigma_b + 0.03 ** 2)
    )
    ssim = ssim_map.mean()
    grad_image_a, grad_image_b = torch.autograd.grad(ssim, (image_a, image_b))
    return locals_payload(locals(), "image_a", "image_b", "window", "ssim_map", "ssim",
                          "grad_image_a", "grad_image_b")


def locals_payload(values: dict[str, Any], *names: str) -> dict[str, Any]:
    """Select tensor locals by stable output names."""
    return {name: values[name] for name in names}


def tensor_payload(values: dict[str, Any]) -> dict[str, Any]:
    """Drop non-tensor scene metadata from a payload."""
    return {name: value for name, value in values.items() if hasattr(value, "detach")}
