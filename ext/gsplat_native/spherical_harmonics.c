#include <math.h>
#include "common.h"

typedef struct {
    int degree;
    size_t count;
    size_t basis_count;
    size_t channels;
    const float *directions;
    const float *coefficients;
    float *output;
} gs_sh_args;

static void
gs_sh_basis(int degree, float x, float y, float z, float *basis)
{
    const float c0 = 0.2820947917738781f;
    const float c1 = 0.48860251190292f;
    basis[0] = c0;
    if (degree < 1) return;
    basis[1] = -c1 * y;
    basis[2] = c1 * z;
    basis[3] = -c1 * x;
    if (degree < 2) return;
    basis[4] = 1.092548430592079f * x * y;
    basis[5] = -1.092548430592079f * y * z;
    basis[6] = 0.3153915652525201f * (3.0f * z * z - 1.0f);
    basis[7] = -1.092548430592079f * x * z;
    basis[8] = 0.5462742152960395f * (x * x - y * y);
    if (degree < 3) return;
    basis[9] = -0.5900435899266435f * y * (3.0f * x * x - y * y);
    basis[10] = 2.890611442640554f * x * y * z;
    basis[11] = -0.4570457994644658f * y * (5.0f * z * z - 1.0f);
    basis[12] = 0.3731763325901154f * z * (5.0f * z * z - 3.0f);
    basis[13] = -0.4570457994644658f * x * (5.0f * z * z - 1.0f);
    basis[14] = 1.445305721320277f * z * (x * x - y * y);
    basis[15] = -0.5900435899266435f * x * (x * x - 3.0f * y * y);
    if (degree < 4) return;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    basis[16] = 2.5033429417967046f * x * y * (x2 - y2);
    basis[17] = -1.770130769779931f * y * z * (3.0f * x2 - y2);
    basis[18] = 0.9461746957575601f * x * y * (7.0f * z2 - 1.0f);
    basis[19] = -0.6690465435572892f * y * z * (7.0f * z2 - 3.0f);
    basis[20] = 0.1057855469152043f * (z2 * (35.0f * z2 - 30.0f) + 3.0f);
    basis[21] = -0.6690465435572892f * x * z * (7.0f * z2 - 3.0f);
    basis[22] = 0.47308734787878f * (x2 - y2) * (7.0f * z2 - 1.0f);
    basis[23] = -1.770130769779931f * x * z * (x2 - 3.0f * y2);
    basis[24] = 0.6258357354491763f * (x2 * x2 - 6.0f * x2 * y2 + y2 * y2);
}

static void *
gs_sh_without_gvl(void *opaque)
{
    gs_sh_args *args = (gs_sh_args *)opaque;
#ifdef GSPLAT_OPENMP
#pragma omp parallel for
#endif
    for (size_t index = 0; index < args->count; ++index) {
        const float *direction = args->directions + index * 3;
        float norm = sqrtf(
            direction[0] * direction[0] +
            direction[1] * direction[1] +
            direction[2] * direction[2]
        );
        if (norm < 1e-12f) norm = 1e-12f;
        float basis[25] = {0};
        gs_sh_basis(
            args->degree,
            direction[0] / norm,
            direction[1] / norm,
            direction[2] / norm,
            basis
        );
        for (size_t channel = 0; channel < args->channels; ++channel) {
            float color = 0.0f;
            for (size_t coefficient = 0; coefficient < args->basis_count; ++coefficient) {
                size_t offset = (index * args->basis_count + coefficient) * args->channels + channel;
                color += basis[coefficient] * args->coefficients[offset];
            }
            args->output[index * args->channels + channel] = color;
        }
    }
    return NULL;
}

static VALUE
gs_sh_forward(VALUE self, VALUE degree_value, VALUE directions, VALUE coefficients)
{
    (void)self;
    gs_require_sfloat(directions, "directions");
    gs_require_sfloat(coefficients, "coefficients");
    int degree = NUM2INT(degree_value);
    narray_t *direction_array = gs_narray(directions);
    narray_t *coefficient_array = gs_narray(coefficients);
    if (degree < 0 || degree > 4 || direction_array->shape[direction_array->ndim - 1] != 3) {
        rb_raise(rb_eArgError, "invalid spherical harmonics inputs");
    }
    size_t count = direction_array->size / 3;
    size_t basis_count = coefficient_array->shape[coefficient_array->ndim - 2];
    size_t channels = coefficient_array->shape[coefficient_array->ndim - 1];
    if (basis_count > 25 || basis_count < (size_t)((degree + 1) * (degree + 1)) ||
        coefficient_array->size != count * basis_count * channels) {
        rb_raise(rb_eArgError, "invalid spherical harmonics coefficient shape");
    }

    size_t *shape = ALLOCA_N(size_t, direction_array->ndim);
    for (int axis = 0; axis < direction_array->ndim - 1; ++axis) {
        shape[axis] = direction_array->shape[axis];
    }
    shape[direction_array->ndim - 1] = channels;
    VALUE output = nary_new(numo_cSFloat, direction_array->ndim, shape);
    gs_sh_args args = {
        degree, count, basis_count, channels,
        GS_SFLOAT_READ(directions),
        GS_SFLOAT_READ(coefficients),
        GS_SFLOAT_WRITE(output)
    };
    rb_thread_call_without_gvl(gs_sh_without_gvl, &args, RUBY_UBF_IO, NULL);
    RB_GC_GUARD(directions);
    RB_GC_GUARD(coefficients);
    RB_GC_GUARD(output);
    return output;
}

void
gs_init_spherical_harmonics(VALUE native)
{
    rb_define_singleton_method(native, "spherical_harmonics_forward_sfloat", gs_sh_forward, 3);
}
