#include <math.h>
#include "rasterization.h"

#ifdef GSPLAT_OPENMP
#define GS_ATOMIC_ADD(target, value) do { \
    _Pragma("omp atomic update") \
    (target) += (value); \
} while (0)
#else
#define GS_ATOMIC_ADD(target, value) ((target) += (value))
#endif

static void
gs_raster_backward_pixel(
    gs_raster_backward_args *args,
    size_t camera,
    int x,
    int y,
    size_t range_start,
    size_t range_end
)
{
    gs_raster_forward_args *forward = &args->forward;
    size_t pixel = (camera * forward->height + (size_t)y) * forward->width + (size_t)x;
    const float *pixel_color_grad = args->grad_render_colors + pixel * forward->channels;
    float current_transmittance = 1.0f - args->render_alphas[pixel];
    float after_transmittance = 1.0f;
    float remaining_color[forward->channels];
    for (size_t channel = 0; channel < forward->channels; ++channel) {
        remaining_color[channel] = forward->backgrounds == NULL ?
            0.0f : forward->backgrounds[camera * forward->channels + channel];
        if (args->grad_backgrounds != NULL) {
            GS_ATOMIC_ADD(
                args->grad_backgrounds[camera * forward->channels + channel],
                pixel_color_grad[channel] * current_transmittance
            );
        }
    }
    size_t tile = (camera * forward->tile_height + y / forward->tile_size) *
        forward->tile_width + x / forward->tile_size;
    if (forward->masks != NULL && forward->masks[tile] == 0) return;
    for (size_t cursor = range_end; cursor > range_start; --cursor) {
        size_t intersection = cursor - 1;
        int32_t gaussian = forward->flatten_ids[intersection];
        float delta_x = forward->means[gaussian * 2] - ((float)x + 0.5f);
        float delta_y = forward->means[gaussian * 2 + 1] - ((float)y + 0.5f);
        const float *conic = forward->conics + gaussian * 3;
        float sigma = 0.5f * (
            conic[0] * delta_x * delta_x + conic[2] * delta_y * delta_y
        ) + conic[1] * delta_x * delta_y;
        float response = expf(-sigma);
        float raw_alpha = forward->opacities[gaussian] * response;
        float alpha = raw_alpha > GS_ALPHA_CLAMP ? GS_ALPHA_CLAMP : raw_alpha;
        int valid = args->render_alphas[pixel] > 0.0f && sigma >= 0.0f &&
            alpha >= GS_ALPHA_SKIP && args->last_ids[pixel] >= (int32_t)intersection &&
            current_transmittance < 1.0f && alpha < 1.0f;
        if (!valid) continue;
        float transmittance_before = current_transmittance / (1.0f - alpha);
        float visibility = transmittance_before * alpha;
        const float *color = forward->colors + gaussian * forward->channels;
        float color_dot = 0.0f;
        for (size_t channel = 0; channel < forward->channels; ++channel) {
            GS_ATOMIC_ADD(
                args->grad_colors[gaussian * forward->channels + channel],
                pixel_color_grad[channel] * visibility
            );
            color_dot += pixel_color_grad[channel] * (color[channel] - remaining_color[channel]);
        }
        float grad_alpha = transmittance_before * (
            color_dot + args->grad_render_alphas[pixel] * after_transmittance
        );
        float grad_raw_alpha = raw_alpha < GS_ALPHA_CLAMP ? grad_alpha : 0.0f;
        GS_ATOMIC_ADD(args->grad_opacities[gaussian], grad_raw_alpha * response);
        float grad_sigma = -grad_raw_alpha * raw_alpha;
        float mean_x = grad_sigma * (conic[0] * delta_x + conic[1] * delta_y);
        float mean_y = grad_sigma * (conic[2] * delta_y + conic[1] * delta_x);
        GS_ATOMIC_ADD(args->grad_means[gaussian * 2], mean_x);
        GS_ATOMIC_ADD(args->grad_means[gaussian * 2 + 1], mean_y);
        if (args->grad_means_abs != NULL) {
            GS_ATOMIC_ADD(args->grad_means_abs[gaussian * 2], fabsf(mean_x));
            GS_ATOMIC_ADD(args->grad_means_abs[gaussian * 2 + 1], fabsf(mean_y));
        }
        GS_ATOMIC_ADD(args->grad_conics[gaussian * 3], grad_sigma * 0.5f * delta_x * delta_x);
        GS_ATOMIC_ADD(args->grad_conics[gaussian * 3 + 1], grad_sigma * delta_x * delta_y);
        GS_ATOMIC_ADD(args->grad_conics[gaussian * 3 + 2], grad_sigma * 0.5f * delta_y * delta_y);
        for (size_t channel = 0; channel < forward->channels; ++channel) {
            remaining_color[channel] =
                alpha * color[channel] + (1.0f - alpha) * remaining_color[channel];
        }
        after_transmittance *= 1.0f - alpha;
        current_transmittance = transmittance_before;
    }
}

void *
gs_raster_backward_without_gvl(void *opaque)
{
    gs_raster_backward_args *args = (gs_raster_backward_args *)opaque;
    gs_raster_forward_args *forward = &args->forward;
    size_t tiles_per_camera = (size_t)forward->tile_width * forward->tile_height;
    size_t total_tiles = forward->cameras * tiles_per_camera;
#ifdef GSPLAT_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (size_t tile = 0; tile < total_tiles; ++tile) {
        size_t camera = tile / tiles_per_camera;
        size_t local_tile = tile % tiles_per_camera;
        int tile_y = (int)(local_tile / forward->tile_width);
        int tile_x = (int)(local_tile % forward->tile_width);
        int x_start = tile_x * forward->tile_size;
        int y_start = tile_y * forward->tile_size;
        int x_end = x_start + forward->tile_size;
        int y_end = y_start + forward->tile_size;
        if (x_end > forward->width) x_end = forward->width;
        if (y_end > forward->height) y_end = forward->height;
        if (x_start >= forward->width || y_start >= forward->height) continue;
        size_t range_start = (size_t)forward->offsets[tile];
        size_t range_end = tile + 1 < total_tiles ?
            (size_t)forward->offsets[tile + 1] : forward->intersection_count;
        int masked = forward->masks != NULL && forward->masks[tile] == 0;
        if ((masked || range_start == range_end) && args->grad_backgrounds == NULL) continue;
        for (int y = y_start; y < y_end; ++y) {
            for (int x = x_start; x < x_end; ++x) {
                gs_raster_backward_pixel(args, camera, x, y, range_start, range_end);
            }
        }
    }
    return NULL;
}
