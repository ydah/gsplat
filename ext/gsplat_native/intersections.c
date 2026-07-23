#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"

typedef struct {
    uint64_t *keys;
    int32_t *flatten_ids;
    size_t count;
} gs_sort_args;

static int
gs_tile_bits(int width, int height)
{
    uint64_t count = (uint64_t)width * (uint64_t)height;
    int bits = 0;
    uint64_t capacity = 1;
    while (capacity < count) {
        capacity <<= 1;
        bits += 1;
    }
    return bits;
}

static int
gs_clamp_tile(int value, int maximum)
{
    if (value < 0) return 0;
    if (value > maximum) return maximum;
    return value;
}

static void
gs_bounds(
    const float *means,
    const float *radii,
    int elliptical,
    int tile_size,
    int tile_width,
    int tile_height,
    int *bounds
)
{
    float radius_x = radii[0];
    float radius_y = elliptical ? radii[1] : radius_x;
    if (radius_x <= 0.0f || radius_y <= 0.0f) {
        memset(bounds, 0, 4 * sizeof(int));
        return;
    }
    bounds[0] = gs_clamp_tile((int)floorf((means[0] - radius_x) / tile_size), tile_width);
    bounds[1] = gs_clamp_tile((int)floorf((means[1] - radius_y) / tile_size), tile_height);
    bounds[2] = gs_clamp_tile((int)ceilf((means[0] + radius_x) / tile_size), tile_width);
    bounds[3] = gs_clamp_tile((int)ceilf((means[1] + radius_y) / tile_size), tile_height);
}

static void *
gs_radix_sort_without_gvl(void *opaque)
{
    gs_sort_args *args = (gs_sort_args *)opaque;
    uint64_t *temporary_keys = malloc(args->count * sizeof(uint64_t));
    int32_t *temporary_ids = malloc(args->count * sizeof(int32_t));
    if (temporary_keys == NULL || temporary_ids == NULL) {
        free(temporary_keys);
        free(temporary_ids);
        return (void *)1;
    }
    uint64_t *source_keys = args->keys;
    uint64_t *destination_keys = temporary_keys;
    int32_t *source_ids = args->flatten_ids;
    int32_t *destination_ids = temporary_ids;
    for (int pass = 0; pass < 8; ++pass) {
        size_t counts[256] = {0};
        int shift = pass * 8;
        for (size_t index = 0; index < args->count; ++index) {
            counts[(source_keys[index] >> shift) & 0xffu] += 1;
        }
        size_t running = 0;
        for (int bucket = 0; bucket < 256; ++bucket) {
            size_t value = counts[bucket];
            counts[bucket] = running;
            running += value;
        }
        for (size_t index = 0; index < args->count; ++index) {
            int bucket = (source_keys[index] >> shift) & 0xffu;
            size_t output_index = counts[bucket]++;
            destination_keys[output_index] = source_keys[index];
            destination_ids[output_index] = source_ids[index];
        }
        uint64_t *key_swap = source_keys;
        source_keys = destination_keys;
        destination_keys = key_swap;
        int32_t *id_swap = source_ids;
        source_ids = destination_ids;
        destination_ids = id_swap;
    }
    free(temporary_keys);
    free(temporary_ids);
    return NULL;
}

static VALUE
gs_isect_tiles(int argc, VALUE *argv, VALUE self)
{
    (void)self;
    if (argc != 7) rb_raise(rb_eArgError, "expected 7 intersection arguments");
    VALUE means2d = argv[0], radii = argv[1], depths = argv[2];
    gs_require_sfloat(means2d, "means2d");
    gs_require_sfloat(radii, "radii");
    gs_require_sfloat(depths, "depths");
    int tile_size = NUM2INT(argv[3]), tile_width = NUM2INT(argv[4]), tile_height = NUM2INT(argv[5]);
    int sort = RTEST(argv[6]);
    narray_t *mean_array = gs_narray(means2d);
    narray_t *radius_array = gs_narray(radii);
    size_t cameras = mean_array->shape[0], gaussians = mean_array->shape[1];
    int elliptical = radius_array->ndim == 3;
    size_t shape_cn[2] = {cameras, gaussians};
    VALUE tiles_per = nary_new(numo_cInt32, 2, shape_cn);
    int32_t *tile_counts = (int32_t *)na_get_pointer_for_write(tiles_per);
    const float *mean_values = GS_SFLOAT_READ(means2d);
    const float *radius_values = GS_SFLOAT_READ(radii);
    size_t total = 0;
    for (size_t index = 0; index < cameras * gaussians; ++index) {
        int bounds[4];
        gs_bounds(
            mean_values + index * 2,
            radius_values + index * (elliptical ? 2 : 1),
            elliptical, tile_size, tile_width, tile_height, bounds
        );
        int count = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]);
        tile_counts[index] = count;
        total += count;
    }
    size_t shape_total[1] = {total};
    VALUE keys = nary_new(numo_cInt64, 1, shape_total);
    VALUE flatten_ids = nary_new(numo_cInt32, 1, shape_total);
    uint64_t *key_values = (uint64_t *)na_get_pointer_for_write(keys);
    int32_t *id_values = (int32_t *)na_get_pointer_for_write(flatten_ids);
    const float *depth_values = GS_SFLOAT_READ(depths);
    int tile_bits = gs_tile_bits(tile_width, tile_height);
    size_t write_index = 0;
    for (size_t camera = 0; camera < cameras; ++camera) {
        for (size_t gaussian = 0; gaussian < gaussians; ++gaussian) {
            size_t index = camera * gaussians + gaussian;
            if (tile_counts[index] == 0) continue;
            int bounds[4];
            gs_bounds(
                mean_values + index * 2,
                radius_values + index * (elliptical ? 2 : 1),
                elliptical, tile_size, tile_width, tile_height, bounds
            );
            union { float value; uint32_t bits; } depth = {depth_values[index]};
            uint64_t upper_camera = camera << tile_bits;
            for (int tile_y = bounds[1]; tile_y < bounds[3]; ++tile_y) {
                for (int tile_x = bounds[0]; tile_x < bounds[2]; ++tile_x) {
                    uint64_t tile_id = (uint64_t)tile_y * tile_width + tile_x;
                    key_values[write_index] = ((upper_camera | tile_id) << 32) | depth.bits;
                    id_values[write_index] = (int32_t)index;
                    write_index += 1;
                }
            }
        }
    }
    if (sort && total > 1) {
        gs_sort_args args = {key_values, id_values, total};
        void *failed = rb_thread_call_without_gvl(gs_radix_sort_without_gvl, &args, RUBY_UBF_IO, NULL);
        if (failed != NULL) rb_raise(rb_eNoMemError, "native radix sort allocation failed");
    }
    return rb_ary_new_from_args(3, tiles_per, keys, flatten_ids);
}

static VALUE
gs_offset_encode(VALUE self, VALUE keys, VALUE cameras_value, VALUE width_value, VALUE height_value)
{
    (void)self;
    int cameras = NUM2INT(cameras_value), width = NUM2INT(width_value), height = NUM2INT(height_value);
    int tile_bits = gs_tile_bits(width, height);
    uint64_t tile_mask = tile_bits == 0 ? 0 : ((1ULL << tile_bits) - 1);
    size_t tile_count = (size_t)cameras * width * height;
    size_t *counts = ALLOC_N(size_t, tile_count);
    memset(counts, 0, tile_count * sizeof(size_t));
    narray_t *key_array = gs_narray(keys);
    const uint64_t *key_values = (const uint64_t *)na_get_pointer_for_read(keys);
    for (size_t index = 0; index < key_array->size; ++index) {
        uint64_t upper = key_values[index] >> 32;
        uint64_t camera = upper >> tile_bits;
        uint64_t tile = upper & tile_mask;
        if (camera >= (uint64_t)cameras || tile >= (uint64_t)(width * height)) {
            xfree(counts);
            rb_raise(rb_eArgError, "intersection key is outside the tile grid");
        }
        counts[camera * width * height + tile] += 1;
    }
    size_t shape[3] = {(size_t)cameras, (size_t)height, (size_t)width};
    VALUE offsets = nary_new(numo_cInt32, 3, shape);
    int32_t *offset_values = (int32_t *)na_get_pointer_for_write(offsets);
    size_t running = 0;
    for (size_t index = 0; index < tile_count; ++index) {
        offset_values[index] = (int32_t)running;
        running += counts[index];
    }
    xfree(counts);
    return offsets;
}

void
gs_init_intersections(VALUE native)
{
    rb_define_singleton_method(native, "isect_tiles_sfloat", gs_isect_tiles, -1);
    rb_define_singleton_method(native, "isect_offset_encode_int64", gs_offset_encode, 4);
}
