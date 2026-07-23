#ifndef GSPLAT_RASTERIZATION_H
#define GSPLAT_RASTERIZATION_H

#include <stddef.h>
#include <stdint.h>

#define GS_ALPHA_CLAMP 0.999f
#define GS_ALPHA_SKIP (1.0f / 255.0f)
#define GS_TRANSMITTANCE_STOP 1.0e-4f

typedef struct {
    const float *means;
    const float *conics;
    const float *colors;
    const float *opacities;
    const float *backgrounds;
    const int32_t *masks;
    const int32_t *offsets;
    const int32_t *flatten_ids;
    float *render_colors;
    float *render_alphas;
    int32_t *last_ids;
    size_t cameras;
    size_t gaussians;
    size_t channels;
    size_t intersection_count;
    int width;
    int height;
    int tile_size;
    int tile_width;
    int tile_height;
} gs_raster_forward_args;

typedef struct {
    gs_raster_forward_args forward;
    const float *render_alphas;
    const int32_t *last_ids;
    const float *grad_render_colors;
    const float *grad_render_alphas;
    float *grad_means;
    float *grad_conics;
    float *grad_colors;
    float *grad_opacities;
    float *grad_backgrounds;
    float *grad_means_abs;
} gs_raster_backward_args;

void *gs_raster_forward_without_gvl(void *opaque);
void *gs_raster_backward_without_gvl(void *opaque);

#endif
