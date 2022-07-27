/**
 * @file kernel_cuda.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-07-24
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include "kernel/kernel_cuda.h"
#include "kernel/launch_prediction.cuh"

#define CONSTRUCT_LORENZOI(Tliteral, Eliteral, FPliteral, T, E, FP)                                            \
    cusz_error_status launch_construct_LorenzoI_T##Tliteral##_E##Eliteral##_FP##FPliteral(                     \
        bool NO_R_SEPARATE, T* const data, dim3 const len3, T* const anchor, dim3 const placeholder_1,         \
        E* const errctrl, dim3 const placeholder_2, double const eb, int const radius, float* time_elapsed,    \
        cudaStream_t stream)                                                                                   \
    {                                                                                                          \
        if (NO_R_SEPARATE)                                                                                     \
            launch_construct_LorenzoI<T, E, FP, true>(                                                         \
                data, len3, anchor, placeholder_1, errctrl, placeholder_2, eb, radius, *time_elapsed, stream); \
        else                                                                                                   \
            launch_construct_LorenzoI<T, E, FP, false>(                                                        \
                data, len3, anchor, placeholder_1, errctrl, placeholder_2, eb, radius, *time_elapsed, stream); \
        return CUSZ_SUCCESS;                                                                                   \
    }

CONSTRUCT_LORENZOI(fp32, ui8, fp32, float, uint8_t, float);
CONSTRUCT_LORENZOI(fp32, ui16, fp32, float, uint16_t, float);
CONSTRUCT_LORENZOI(fp32, ui32, fp32, float, uint32_t, float);
CONSTRUCT_LORENZOI(fp32, fp32, fp32, float, float, float);

#undef CONSTRUCT_LORENZOI

#define RECONSTRUCT_LORENZOI(Tliteral, Eliteral, FPliteral, T, E, FP)                                         \
    cusz_error_status launch_reconstruct_LorenzoI_T##Tliteral##_E##Eliteral##_FP##FPliteral(                  \
        T* xdata, dim3 const len3, T* anchor, dim3 const placeholder_1, E* errctrl, dim3 const placeholder_2, \
        double const eb, int const radius, float* time_elapsed, cudaStream_t stream)                          \
    {                                                                                                         \
        launch_reconstruct_LorenzoI<T, E, FP>(                                                                \
            xdata, len3, anchor, placeholder_1, errctrl, placeholder_2, eb, radius, *time_elapsed, stream);   \
        return CUSZ_SUCCESS;                                                                                  \
    }

RECONSTRUCT_LORENZOI(fp32, ui8, fp32, float, uint8_t, float);
RECONSTRUCT_LORENZOI(fp32, ui16, fp32, float, uint16_t, float);
RECONSTRUCT_LORENZOI(fp32, ui32, fp32, float, uint32_t, float);
RECONSTRUCT_LORENZOI(fp32, fp32, fp32, float, float, float);

#undef RECONSTRUCT_LORENZOI

#define CONSTRUCT_SPLINE3(Tliteral, Eliteral, FPliteral, T, E, FP)                                                   \
    cusz_error_status launch_construct_Spline3_T##Tliteral##_E##Eliteral##_FP##FPliteral(                            \
        bool NO_R_SEPARATE, T* data, dim3 const len3, T* anchor, dim3 const an_len3, E* errctrl, dim3 const ec_len3, \
        double const eb, int const radius, float* time_elapsed, cudaStream_t stream)                                 \
    {                                                                                                                \
        if (NO_R_SEPARATE)                                                                                           \
            launch_construct_Spline3<T, E, FP, true>(                                                                \
                data, len3, anchor, an_len3, errctrl, ec_len3, eb, radius, *time_elapsed, stream);                   \
        else                                                                                                         \
            launch_construct_Spline3<T, E, FP, false>(                                                               \
                data, len3, anchor, an_len3, errctrl, ec_len3, eb, radius, *time_elapsed, stream);                   \
        return CUSZ_SUCCESS;                                                                                         \
    }

CONSTRUCT_SPLINE3(fp32, ui8, fp32, float, uint8_t, float);
CONSTRUCT_SPLINE3(fp32, ui16, fp32, float, uint16_t, float);
CONSTRUCT_SPLINE3(fp32, ui32, fp32, float, uint32_t, float);
CONSTRUCT_SPLINE3(fp32, fp32, fp32, float, float, float);

#undef CONSTRUCT_SPLINE3

#define RECONSTRUCT_SPLINE3(Tliteral, Eliteral, FPliteral, T, E, FP)                                               \
    cusz_error_status launch_reconstruct_Spline3_T##Tliteral##_E##Eliteral##_FP##FPliteral(                        \
        T* xdata, dim3 const len3, T* anchor, dim3 const an_len3, E* errctrl, dim3 const ec_len3, double const eb, \
        int const radius, float* time_elapsed, cudaStream_t stream)                                                \
    {                                                                                                              \
        launch_reconstruct_Spline3<T, E, FP>(                                                                      \
            xdata, len3, anchor, an_len3, errctrl, ec_len3, eb, radius, *time_elapsed, stream);                    \
        return CUSZ_SUCCESS;                                                                                       \
    }

RECONSTRUCT_SPLINE3(fp32, ui8, fp32, float, uint8_t, float);
RECONSTRUCT_SPLINE3(fp32, ui16, fp32, float, uint16_t, float);
RECONSTRUCT_SPLINE3(fp32, ui32, fp32, float, uint32_t, float);
RECONSTRUCT_SPLINE3(fp32, fp32, fp32, float, float, float);

#undef RECONSTRUCT_SPLINE3
