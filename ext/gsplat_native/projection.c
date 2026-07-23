#include <math.h>
#include <string.h>
#include "common.h"

typedef struct {
    size_t cameras;
    size_t gaussians;
    int width;
    int height;
    int orthographic;
    int compensation;
    float eps2d;
    float near_plane;
    float far_plane;
    float radius_clip;
    const float *means;
    const float *covars;
    const float *views;
    const float *intrinsics;
    int32_t *radii;
    float *means2d;
    float *depths;
    float *conics;
    float *compensations;
} gs_projection_args;

static void
gs_mat3_product(const float *left, const float *right, float *output)
{
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            float value = 0.0f;
            for (int inner = 0; inner < 3; ++inner) {
                value += left[row * 3 + inner] * right[inner * 3 + column];
            }
            output[row * 3 + column] = value;
        }
    }
}

static void
gs_camera_covar(const float *rotation, const float *covar, float *output)
{
    float intermediate[9], transpose[9];
    gs_mat3_product(rotation, covar, intermediate);
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            transpose[row * 3 + column] = rotation[column * 3 + row];
        }
    }
    gs_mat3_product(intermediate, transpose, output);
}

static float
gs_clip(float value, float minimum, float maximum)
{
    return fminf(fmaxf(value, minimum), maximum);
}

static void
gs_project_one(gs_projection_args *args, size_t camera, size_t gaussian)
{
    const float *view = args->views + camera * 16;
    const float *intrinsic = args->intrinsics + camera * 9;
    const float *mean = args->means + gaussian * 3;
    const float *covar = args->covars + gaussian * 9;
    float rotation[9] = {
        view[0], view[1], view[2],
        view[4], view[5], view[6],
        view[8], view[9], view[10]
    };
    float camera_mean[3];
    for (int row = 0; row < 3; ++row) {
        camera_mean[row] =
            rotation[row * 3] * mean[0] +
            rotation[row * 3 + 1] * mean[1] +
            rotation[row * 3 + 2] * mean[2] +
            view[row * 4 + 3];
    }
    float camera_covar[9];
    gs_camera_covar(rotation, covar, camera_covar);
    float depth = camera_mean[2];
    float safe_z = (depth <= args->near_plane || depth >= args->far_plane) ? 1.0f : depth;
    float jacobian[6] = {0};
    float projected[2];
    if (args->orthographic) {
        jacobian[0] = intrinsic[0];
        jacobian[4] = intrinsic[4];
        projected[0] = camera_mean[0] * intrinsic[0] + intrinsic[2];
        projected[1] = camera_mean[1] * intrinsic[4] + intrinsic[5];
    } else {
        float minimum_x = -(intrinsic[2] / intrinsic[0] + 0.15f * args->width / intrinsic[0]);
        float maximum_x = (args->width - intrinsic[2]) / intrinsic[0] + 0.15f * args->width / intrinsic[0];
        float minimum_y = -(intrinsic[5] / intrinsic[4] + 0.15f * args->height / intrinsic[4]);
        float maximum_y = (args->height - intrinsic[5]) / intrinsic[4] + 0.15f * args->height / intrinsic[4];
        float clipped_x = gs_clip(camera_mean[0] / safe_z, minimum_x, maximum_x) * safe_z;
        float clipped_y = gs_clip(camera_mean[1] / safe_z, minimum_y, maximum_y) * safe_z;
        jacobian[0] = intrinsic[0] / safe_z;
        jacobian[2] = -intrinsic[0] * clipped_x / (safe_z * safe_z);
        jacobian[4] = intrinsic[4] / safe_z;
        jacobian[5] = -intrinsic[4] * clipped_y / (safe_z * safe_z);
        projected[0] =
            (intrinsic[0] * camera_mean[0] + intrinsic[1] * camera_mean[1] + intrinsic[2] * safe_z) / safe_z;
        projected[1] =
            (intrinsic[3] * camera_mean[0] + intrinsic[4] * camera_mean[1] + intrinsic[5] * safe_z) / safe_z;
    }
    float covar2d[4] = {0};
    for (int row = 0; row < 2; ++row) {
        for (int column = 0; column < 2; ++column) {
            for (int left = 0; left < 3; ++left) {
                for (int right = 0; right < 3; ++right) {
                    covar2d[row * 2 + column] +=
                        jacobian[row * 3 + left] * camera_covar[left * 3 + right] *
                        jacobian[column * 3 + right];
                }
            }
        }
    }
    float original_det = covar2d[0] * covar2d[3] - covar2d[1] * covar2d[2];
    covar2d[0] += args->eps2d;
    covar2d[3] += args->eps2d;
    float determinant = covar2d[0] * covar2d[3] - covar2d[1] * covar2d[2];
    float safe_det = determinant <= 0.0f ? 1e-10f : determinant;
    int32_t radius_x = (int32_t)ceilf(3.33f * sqrtf(fmaxf(covar2d[0], 0.0f)));
    int32_t radius_y = (int32_t)ceilf(3.33f * sqrtf(fmaxf(covar2d[3], 0.0f)));
    int visible = determinant > 0.0f && depth > args->near_plane && depth < args->far_plane &&
                  (radius_x > args->radius_clip || radius_y > args->radius_clip) &&
                  projected[0] + radius_x > 0.0f && projected[0] - radius_x < args->width &&
                  projected[1] + radius_y > 0.0f && projected[1] - radius_y < args->height;
    size_t index = camera * args->gaussians + gaussian;
    args->radii[index * 2] = visible ? radius_x : 0;
    args->radii[index * 2 + 1] = visible ? radius_y : 0;
    args->means2d[index * 2] = projected[0];
    args->means2d[index * 2 + 1] = projected[1];
    args->depths[index] = depth;
    args->conics[index * 3] = covar2d[3] / safe_det;
    args->conics[index * 3 + 1] = -(covar2d[1] + covar2d[2]) / (2.0f * safe_det);
    args->conics[index * 3 + 2] = covar2d[0] / safe_det;
    if (args->compensation) {
        args->compensations[index] = sqrtf(fmaxf(original_det / safe_det, 0.0f));
    }
}

static void *
gs_projection_without_gvl(void *opaque)
{
    gs_projection_args *args = (gs_projection_args *)opaque;
    size_t count = args->cameras * args->gaussians;
#ifdef GSPLAT_OPENMP
#pragma omp parallel for
#endif
    for (size_t index = 0; index < count; ++index) {
        gs_project_one(args, index / args->gaussians, index % args->gaussians);
    }
    return NULL;
}

static VALUE
gs_projection_forward(int argc, VALUE *argv, VALUE self)
{
    (void)self;
    if (argc != 12) rb_raise(rb_eArgError, "expected 12 projection arguments");
    VALUE means = argv[0], covars = argv[1], views = argv[2], intrinsics = argv[3];
    VALUE width = argv[4], height = argv[5], eps2d = argv[6], near_plane = argv[7];
    VALUE far_plane = argv[8], radius_clip = argv[9], compensation = argv[10], orthographic = argv[11];
    gs_require_sfloat(means, "means");
    gs_require_sfloat(covars, "covars");
    gs_require_sfloat(views, "viewmats");
    gs_require_sfloat(intrinsics, "intrinsics");
    narray_t *mean_array = gs_narray(means);
    narray_t *view_array = gs_narray(views);
    size_t cameras = view_array->shape[0], gaussians = mean_array->shape[0];
    size_t shape_cn[2] = {cameras, gaussians};
    size_t shape_cn2[3] = {cameras, gaussians, 2};
    size_t shape_cn3[3] = {cameras, gaussians, 3};
    VALUE radii = nary_new(numo_cInt32, 3, shape_cn2);
    VALUE means2d = nary_new(numo_cSFloat, 3, shape_cn2);
    VALUE depths = nary_new(numo_cSFloat, 2, shape_cn);
    VALUE conics = nary_new(numo_cSFloat, 3, shape_cn3);
    VALUE compensations = RTEST(compensation) ? nary_new(numo_cSFloat, 2, shape_cn) : Qnil;
    gs_projection_args args = {
        cameras, gaussians, NUM2INT(width), NUM2INT(height),
        RTEST(orthographic), RTEST(compensation),
        NUM2DBL(eps2d), NUM2DBL(near_plane), NUM2DBL(far_plane), NUM2DBL(radius_clip),
        GS_SFLOAT_READ(means), GS_SFLOAT_READ(covars), GS_SFLOAT_READ(views),
        GS_SFLOAT_READ(intrinsics),
        (int32_t *)na_get_pointer_for_write(radii),
        GS_SFLOAT_WRITE(means2d), GS_SFLOAT_WRITE(depths), GS_SFLOAT_WRITE(conics),
        NIL_P(compensations) ? NULL : GS_SFLOAT_WRITE(compensations)
    };
    rb_thread_call_without_gvl(gs_projection_without_gvl, &args, RUBY_UBF_IO, NULL);
    return rb_ary_new_from_args(5, radii, means2d, depths, conics, compensations);
}

void
gs_init_projection(VALUE native)
{
    rb_define_singleton_method(native, "projection_forward_sfloat", gs_projection_forward, -1);
}
