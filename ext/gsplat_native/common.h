#ifndef GSPLAT_COMMON_H
#define GSPLAT_COMMON_H

#include <ruby.h>
#include <ruby/thread.h>
#include <numo/narray.h>
#include <numo/intern.h>

static inline narray_t *
gs_narray(VALUE value)
{
    narray_t *array;
    GetNArray(value, array);
    return array;
}

static inline void
gs_require_sfloat(VALUE value, const char *name)
{
    if (!rb_obj_is_kind_of(value, numo_cSFloat)) {
        rb_raise(rb_eTypeError, "%s must be Numo::SFloat", name);
    }
}

static inline void
gs_require_same_shape(VALUE left, VALUE right)
{
    narray_t *left_array = gs_narray(left);
    narray_t *right_array = gs_narray(right);
    if (left_array->ndim != right_array->ndim ||
        left_array->size != right_array->size) {
        rb_raise(rb_eArgError, "NArray shapes differ");
    }
    for (int axis = 0; axis < left_array->ndim; ++axis) {
        if (left_array->shape[axis] != right_array->shape[axis]) {
            rb_raise(rb_eArgError, "NArray shapes differ");
        }
    }
}

#define GS_SFLOAT_READ(value) ((const float *)na_get_pointer_for_read(value))
#define GS_SFLOAT_WRITE(value) ((float *)na_get_pointer_for_write(value))

#endif
