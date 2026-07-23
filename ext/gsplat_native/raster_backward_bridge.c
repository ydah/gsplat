#include <string.h>
#include "common.h"
#include "rasterization.h"

static VALUE
gs_zero_sfloat(int dimensions, size_t *shape)
{
    VALUE output = nary_new(numo_cSFloat, dimensions, shape);
    memset(GS_SFLOAT_WRITE(output), 0, gs_narray(output)->size * sizeof(float));
    return output;
}

static VALUE
gs_raster_backward(int argc, VALUE *argv, VALUE self)
{
    (void)self;
    if (argc != 16) rb_raise(rb_eArgError, "expected 16 raster backward arguments");
    int sfloat_indices[] = {0, 1, 2, 3, 11, 13, 14};
    for (size_t index = 0; index < sizeof(sfloat_indices) / sizeof(int); ++index) {
        gs_require_sfloat(argv[sfloat_indices[index]], "raster input");
    }
    if (!NIL_P(argv[4])) gs_require_sfloat(argv[4], "backgrounds");
    if (!NIL_P(argv[5])) gs_require_int32(argv[5], "masks");
    gs_require_int32(argv[9], "isect_offsets");
    gs_require_int32(argv[10], "flatten_ids");
    gs_require_int32(argv[12], "last_ids");
    narray_t *means = gs_narray(argv[0]);
    narray_t *colors = gs_narray(argv[2]);
    narray_t *offsets = gs_narray(argv[9]);
    size_t means_shape[3] = {means->shape[0], means->shape[1], 2};
    size_t conic_shape[3] = {means->shape[0], means->shape[1], 3};
    size_t color_shape[3] = {means->shape[0], means->shape[1], colors->shape[2]};
    size_t opacity_shape[2] = {means->shape[0], means->shape[1]};
    size_t background_shape[2] = {means->shape[0], colors->shape[2]};
    VALUE grad_means = gs_zero_sfloat(3, means_shape);
    VALUE grad_conics = gs_zero_sfloat(3, conic_shape);
    VALUE grad_colors = gs_zero_sfloat(3, color_shape);
    VALUE grad_opacities = gs_zero_sfloat(2, opacity_shape);
    VALUE grad_backgrounds = NIL_P(argv[4]) ? Qnil : gs_zero_sfloat(2, background_shape);
    VALUE grad_means_abs = RTEST(argv[15]) ? gs_zero_sfloat(3, means_shape) : Qnil;
    gs_raster_backward_args args = {
        {
            GS_SFLOAT_READ(argv[0]), GS_SFLOAT_READ(argv[1]), GS_SFLOAT_READ(argv[2]),
            GS_SFLOAT_READ(argv[3]), NIL_P(argv[4]) ? NULL : GS_SFLOAT_READ(argv[4]),
            NIL_P(argv[5]) ? NULL : GS_INT32_READ(argv[5]), GS_INT32_READ(argv[9]),
            GS_INT32_READ(argv[10]), NULL, NULL, NULL, means->shape[0], means->shape[1],
            colors->shape[2], gs_narray(argv[10])->size, NUM2INT(argv[6]), NUM2INT(argv[7]),
            NUM2INT(argv[8]), (int)offsets->shape[2], (int)offsets->shape[1]
        },
        GS_SFLOAT_READ(argv[11]), GS_INT32_READ(argv[12]), GS_SFLOAT_READ(argv[13]),
        GS_SFLOAT_READ(argv[14]), GS_SFLOAT_WRITE(grad_means), GS_SFLOAT_WRITE(grad_conics),
        GS_SFLOAT_WRITE(grad_colors), GS_SFLOAT_WRITE(grad_opacities),
        NIL_P(grad_backgrounds) ? NULL : GS_SFLOAT_WRITE(grad_backgrounds),
        NIL_P(grad_means_abs) ? NULL : GS_SFLOAT_WRITE(grad_means_abs)
    };
    rb_thread_call_without_gvl(gs_raster_backward_without_gvl, &args, RUBY_UBF_IO, NULL);
    VALUE gradients = rb_ary_new_from_args(
        5, grad_means, grad_conics, grad_colors, grad_opacities, grad_backgrounds
    );
    RB_GC_GUARD(argv[0]);
    RB_GC_GUARD(gradients);
    return rb_ary_new_from_args(2, gradients, grad_means_abs);
}

void
gs_init_rasterization_backward(VALUE native)
{
    rb_define_singleton_method(native, "rasterize_backward_sfloat", gs_raster_backward, -1);
}
