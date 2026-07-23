#include "common.h"

typedef struct {
    const float *left;
    const float *right;
    float *output;
    size_t count;
} gs_add_args;

void gs_init_spherical_harmonics(VALUE native);
void gs_init_projection(VALUE native);

static void *
gs_add_without_gvl(void *opaque)
{
    gs_add_args *args = (gs_add_args *)opaque;
#ifdef GSPLAT_OPENMP
#pragma omp parallel for
#endif
    for (size_t index = 0; index < args->count; ++index) {
        args->output[index] = args->left[index] + args->right[index];
    }
    return NULL;
}

static VALUE
gs_native_add(VALUE self, VALUE left, VALUE right)
{
    (void)self;
    gs_require_sfloat(left, "left");
    gs_require_sfloat(right, "right");
    gs_require_same_shape(left, right);

    VALUE output = rb_funcall(left, rb_intern("dup"), 0);
    gs_add_args args = {
        GS_SFLOAT_READ(left),
        GS_SFLOAT_READ(right),
        GS_SFLOAT_WRITE(output),
        gs_narray(left)->size
    };
    rb_thread_call_without_gvl(gs_add_without_gvl, &args, RUBY_UBF_IO, NULL);
    RB_GC_GUARD(left);
    RB_GC_GUARD(right);
    RB_GC_GUARD(output);
    return output;
}

void
Init_gsplat_native(void)
{
    VALUE gsplat = rb_const_get(rb_cObject, rb_intern("Gsplat"));
    VALUE native = rb_define_module_under(gsplat, "Native");
    rb_define_singleton_method(native, "add", gs_native_add, 2);
    gs_init_spherical_harmonics(native);
    gs_init_projection(native);
}
