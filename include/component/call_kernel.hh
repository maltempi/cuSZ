/**
 * @file call_kernel.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-07-27
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef COMPONENT_CALL_KERNEL_HH
#define COMPONENT_CALL_KERNEL_HH

#include "../kernel/kernel_cuda.h"

namespace cusz {

template <typename T, typename E, typename FP>
cusz_error_status cpplaunch_construct_LorenzoI(
    bool         NO_R_SEPARATE,
    T* const     data,
    dim3 const   len3,
    T* const     anchor,
    dim3 const   placeholder_1,
    E* const     errctrl,
    dim3 const   placeholder_2,
    double const eb,
    int const    radius,
    float*       time_elapsed,
    cudaStream_t stream);

template <typename T, typename E, typename FP>
cusz_error_status cpplaunch_reconstruct_LorenzoI(
    T*           xdata,
    dim3 const   len3,
    T*           anchor,
    dim3 const   placeholder_1,
    E*           errctrl,
    dim3 const   placeholder_2,
    double const eb,
    int const    radius,
    float*       time_elapsed,
    cudaStream_t stream);

template <typename T, typename E, typename FP>
cusz_error_status cpplaunch_construct_Spline3(
    bool         NO_R_SEPARATE,
    T*           data,
    dim3 const   len3,
    T*           anchor,
    dim3 const   an_len3,
    E*           errctrl,
    dim3 const   ec_len3,
    double const eb,
    int const    radius,
    float*       time_elapsed,
    cudaStream_t stream);

template <typename T, typename E, typename FP>
cusz_error_status cpplaunch_reconstruct_Spline3(
    T*           xdata,
    dim3 const   len3,
    T*           anchor,
    dim3 const   an_len3,
    E*           errctrl,
    dim3 const   ec_len3,
    double const eb,
    int const    radius,
    float*       time_elapsed,
    cudaStream_t stream);

}  // namespace cusz

#define CPP_CONSTRUCT_LORENZOI(Tliteral, Eliteral, FPliteral, T, E, FP)                                         \
    template <>                                                                                                 \
    cusz_error_status cusz::cpplaunch_construct_LorenzoI<T, E, FP>(                                             \
        bool NO_R_SEPARATE, T* const data, dim3 const len3, T* const anchor, dim3 const placeholder_1,          \
        E* const errctrl, dim3 const placeholder_2, double const eb, int const radius, float* time_elapsed,     \
        cudaStream_t stream)                                                                                    \
    {                                                                                                           \
        return launch_construct_LorenzoI_T##Tliteral##_E##Eliteral##_FP##FPliteral(                             \
            NO_R_SEPARATE, data, len3, anchor, placeholder_1, errctrl, placeholder_2, eb, radius, time_elapsed, \
            stream);                                                                                            \
    }

CPP_CONSTRUCT_LORENZOI(fp32, ui8, fp32, float, uint8_t, float);
CPP_CONSTRUCT_LORENZOI(fp32, ui16, fp32, float, uint16_t, float);
CPP_CONSTRUCT_LORENZOI(fp32, ui32, fp32, float, uint32_t, float);
CPP_CONSTRUCT_LORENZOI(fp32, fp32, fp32, float, float, float);

#undef CPP_CONSTRUCT_LORENZOI

#define CPP_RECONSTRUCT_LORENZOI(Tliteral, Eliteral, FPliteral, T, E, FP)                                      \
    template <>                                                                                                \
    cusz_error_status cusz::cpplaunch_reconstruct_LorenzoI<T, E, FP>(                                          \
        T * xdata, dim3 const len3, T* anchor, dim3 const placeholder_1, E* errctrl, dim3 const placeholder_2, \
        double const eb, int const radius, float* time_elapsed, cudaStream_t stream)                           \
    {                                                                                                          \
        return launch_reconstruct_LorenzoI_T##Tliteral##_E##Eliteral##_FP##FPliteral(                          \
            xdata, len3, anchor, placeholder_1, errctrl, placeholder_2, eb, radius, time_elapsed, stream);     \
    }

CPP_RECONSTRUCT_LORENZOI(fp32, ui8, fp32, float, uint8_t, float);
CPP_RECONSTRUCT_LORENZOI(fp32, ui16, fp32, float, uint16_t, float);
CPP_RECONSTRUCT_LORENZOI(fp32, ui32, fp32, float, uint32_t, float);
CPP_RECONSTRUCT_LORENZOI(fp32, fp32, fp32, float, float, float);

#undef CPP_RECONSTRUCT_LORENZOI

#define CPP_CONSTRUCT_SPLINE3(Tliteral, Eliteral, FPliteral, T, E, FP)                                               \
    template <>                                                                                                      \
    cusz_error_status cusz::cpplaunch_construct_Spline3<T, E, FP>(                                                   \
        bool NO_R_SEPARATE, T* data, dim3 const len3, T* anchor, dim3 const an_len3, E* errctrl, dim3 const ec_len3, \
        double const eb, int const radius, float* time_elapsed, cudaStream_t stream)                                 \
    {                                                                                                                \
        return launch_construct_Spline3_T##Tliteral##_E##Eliteral##_FP##FPliteral(                                   \
            NO_R_SEPARATE, data, len3, anchor, an_len3, errctrl, ec_len3, eb, radius, time_elapsed, stream);         \
    }

CPP_CONSTRUCT_SPLINE3(fp32, ui8, fp32, float, uint8_t, float);
CPP_CONSTRUCT_SPLINE3(fp32, ui16, fp32, float, uint16_t, float);
CPP_CONSTRUCT_SPLINE3(fp32, ui32, fp32, float, uint32_t, float);
CPP_CONSTRUCT_SPLINE3(fp32, fp32, fp32, float, float, float);

#undef CPP_CONSTRUCT_SPLINE3

#define CPP_RECONSTRUCT_SPLINE3(Tliteral, Eliteral, FPliteral, T, E, FP)                                            \
    template <>                                                                                                     \
    cusz_error_status cusz::cpplaunch_reconstruct_Spline3<T, E, FP>(                                                \
        T * xdata, dim3 const len3, T* anchor, dim3 const an_len3, E* errctrl, dim3 const ec_len3, double const eb, \
        int const radius, float* time_elapsed, cudaStream_t stream)                                                 \
    {                                                                                                               \
        return launch_reconstruct_Spline3_T##Tliteral##_E##Eliteral##_FP##FPliteral(                                \
            xdata, len3, anchor, an_len3, errctrl, ec_len3, eb, radius, time_elapsed, stream);                      \
    }

CPP_RECONSTRUCT_SPLINE3(fp32, ui8, fp32, float, uint8_t, float);
CPP_RECONSTRUCT_SPLINE3(fp32, ui16, fp32, float, uint16_t, float);
CPP_RECONSTRUCT_SPLINE3(fp32, ui32, fp32, float, uint32_t, float);
CPP_RECONSTRUCT_SPLINE3(fp32, fp32, fp32, float, float, float);

#undef CPP_RECONSTRUCT_SPLINE3

#endif
