#include <math.h>
#include <string.h>
#include "common.h"
#include "rasterization.h"

void gs_init_rasterization_backward(VALUE native);

static size_t
gs_pixel_index(const gs_raster_forward_args *args, size_t camera, int x, int y)
{
    return (camera * args->height + (size_t)y) * args->width + (size_t)x;
}

static void
gs_raster_pixel(
    const gs_raster_forward_args *args,
    size_t camera,
    int x,
    int y,
    size_t range_start,
    size_t range_end
)
{
    size_t pixel = gs_pixel_index(args, camera, x, y);
    float *output = args->render_colors + pixel * args->channels;
    float transmittance = 1.0f;
    for (size_t intersection = range_start; intersection < range_end; ++intersection) {
        int32_t gaussian = args->flatten_ids[intersection];
        float delta_x = args->means[gaussian * 2] - ((float)x + 0.5f);
        float delta_y = args->means[gaussian * 2 + 1] - ((float)y + 0.5f);
        const float *conic = args->conics + gaussian * 3;
        float sigma = 0.5f * (
            conic[0] * delta_x * delta_x + conic[2] * delta_y * delta_y
        ) + conic[1] * delta_x * delta_y;
        float alpha = args->opacities[gaussian] * expf(-sigma);
        if (alpha > GS_ALPHA_CLAMP) alpha = GS_ALPHA_CLAMP;
        if (sigma < 0.0f || alpha < GS_ALPHA_SKIP) continue;
        float next_transmittance = transmittance * (1.0f - alpha);
        if (next_transmittance <= GS_TRANSMITTANCE_STOP) break;
        float visibility = alpha * transmittance;
        const float *color = args->colors + gaussian * args->channels;
        for (size_t channel = 0; channel < args->channels; ++channel) {
            output[channel] += visibility * color[channel];
        }
        transmittance = next_transmittance;
        args->last_ids[pixel] = (int32_t)intersection;
    }
    if (args->backgrounds != NULL) {
        const float *background = args->backgrounds + camera * args->channels;
        for (size_t channel = 0; channel < args->channels; ++channel) {
            output[channel] += transmittance * background[channel];
        }
    }
    args->render_alphas[pixel] = 1.0f - transmittance;
}

void *
gs_raster_forward_without_gvl(void *opaque)
{
    gs_raster_forward_args *args = (gs_raster_forward_args *)opaque;
    size_t tiles_per_camera = (size_t)args->tile_width * args->tile_height;
    size_t total_tiles = args->cameras * tiles_per_camera;
#ifdef GSPLAT_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (size_t tile = 0; tile < total_tiles; ++tile) {
        size_t camera = tile / tiles_per_camera;
        size_t local_tile = tile % tiles_per_camera;
        int tile_y = (int)(local_tile / args->tile_width);
        int tile_x = (int)(local_tile % args->tile_width);
        int x_start = tile_x * args->tile_size;
        int y_start = tile_y * args->tile_size;
        int x_end = x_start + args->tile_size;
        int y_end = y_start + args->tile_size;
        if (x_end > args->width) x_end = args->width;
        if (y_end > args->height) y_end = args->height;
        if (x_start >= args->width || y_start >= args->height) continue;
        size_t range_start = (size_t)args->offsets[tile];
        size_t range_end = tile + 1 < total_tiles ?
            (size_t)args->offsets[tile + 1] : args->intersection_count;
        int masked = args->masks != NULL && args->masks[tile] == 0;
        if ((masked || range_start == range_end) && args->backgrounds == NULL) continue;
        for (int y = y_start; y < y_end; ++y) {
            for (int x = x_start; x < x_end; ++x) {
                if (masked) {
                    size_t pixel = gs_pixel_index(args, camera, x, y);
                    if (args->backgrounds != NULL) {
                        memcpy(
                            args->render_colors + pixel * args->channels,
                            args->backgrounds + camera * args->channels,
                            args->channels * sizeof(float)
                        );
                    }
                } else {
                    gs_raster_pixel(args, camera, x, y, range_start, range_end);
                }
            }
        }
    }
    return NULL;
}

static VALUE
gs_raster_forward(int argc, VALUE *argv, VALUE self)
{
    (void)self;
    if (argc != 11) rb_raise(rb_eArgError, "expected 11 raster forward arguments");
    gs_require_sfloat(argv[0], "means2d");
    gs_require_sfloat(argv[1], "conics");
    gs_require_sfloat(argv[2], "colors");
    gs_require_sfloat(argv[3], "opacities");
    if (!NIL_P(argv[4])) gs_require_sfloat(argv[4], "backgrounds");
    if (!NIL_P(argv[5])) gs_require_int32(argv[5], "masks");
    gs_require_int32(argv[9], "isect_offsets");
    gs_require_int32(argv[10], "flatten_ids");
    narray_t *means = gs_narray(argv[0]);
    narray_t *colors = gs_narray(argv[2]);
    narray_t *offsets = gs_narray(argv[9]);
    size_t color_shape[4] = {
        means->shape[0], (size_t)NUM2INT(argv[7]), (size_t)NUM2INT(argv[6]), colors->shape[2]
    };
    size_t alpha_shape[4] = {means->shape[0], color_shape[1], color_shape[2], 1};
    size_t last_shape[3] = {means->shape[0], color_shape[1], color_shape[2]};
    VALUE render_colors = nary_new(numo_cSFloat, 4, color_shape);
    VALUE render_alphas = nary_new(numo_cSFloat, 4, alpha_shape);
    VALUE last_ids = nary_new(numo_cInt32, 3, last_shape);
    memset(GS_SFLOAT_WRITE(render_colors), 0, gs_narray(render_colors)->size * sizeof(float));
    memset(GS_SFLOAT_WRITE(render_alphas), 0, gs_narray(render_alphas)->size * sizeof(float));
    memset(GS_INT32_WRITE(last_ids), 0, gs_narray(last_ids)->size * sizeof(int32_t));
    gs_raster_forward_args args = {
        GS_SFLOAT_READ(argv[0]), GS_SFLOAT_READ(argv[1]), GS_SFLOAT_READ(argv[2]),
        GS_SFLOAT_READ(argv[3]), NIL_P(argv[4]) ? NULL : GS_SFLOAT_READ(argv[4]),
        NIL_P(argv[5]) ? NULL : GS_INT32_READ(argv[5]), GS_INT32_READ(argv[9]),
        GS_INT32_READ(argv[10]), GS_SFLOAT_WRITE(render_colors), GS_SFLOAT_WRITE(render_alphas),
        GS_INT32_WRITE(last_ids), means->shape[0], means->shape[1], colors->shape[2],
        gs_narray(argv[10])->size, NUM2INT(argv[6]), NUM2INT(argv[7]), NUM2INT(argv[8]),
        (int)offsets->shape[2], (int)offsets->shape[1]
    };
    rb_thread_call_without_gvl(gs_raster_forward_without_gvl, &args, RUBY_UBF_IO, NULL);
    RB_GC_GUARD(argv[0]);
    RB_GC_GUARD(render_colors);
    return rb_ary_new_from_args(3, render_colors, render_alphas, last_ids);
}

void
gs_init_rasterization(VALUE native)
{
    rb_define_singleton_method(native, "rasterize_forward_sfloat", gs_raster_forward, -1);
    gs_init_rasterization_backward(native);
}
