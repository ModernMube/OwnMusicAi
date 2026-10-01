//! `x · wᵀ` for the projections. Candle runs a gemv for a single row on Metal, but from two
//! rows on it switches to a full gemm that streams the weights at well under half the
//! bandwidth - and a CFG decode step is exactly two rows. So 2-4 rows get a small kernel of
//! our own that reads every weight row once for all of them.

use candle_core::Tensor;

use crate::Result;

/// `x` is [rows, k], `w` a [n, k] weight, the result [rows, n].
pub fn matmul_t(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "metal")]
    if metal::fits(x, w) {
        return Ok(x.apply_op2_no_bwd(w, &metal::GemvRows)?);
    }
    Ok(x.matmul(&w.t()?)?)
}

#[cfg(feature = "metal")]
mod metal {
    use std::sync::OnceLock;

    use candle_core::{
        CpuStorage, CustomOp2, DType, Layout, MetalDevice, MetalStorage, Shape, Tensor, backend::BackendStorage, bail,
    };
    use candle_metal_kernels::{
        MetalKernelError,
        metal::{ComputeCommandEncoder, ComputePipeline, Device},
    };
    use objc2_metal::MTLSize;

    const MAX_ROWS: usize = 4;
    const THREADS: usize = 256;
    /// 8 simdgroups x 4 weight rows each
    const OUTPUTS_PER_GROUP: usize = 32;

    const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr uint MAX_ROWS = 4;
constant constexpr uint PER_SIMD = 4;

// out[r, n] = dot(x[r, :], w[n, :]); each simdgroup owns PER_SIMD weight rows, each lane
// walks k sixteen values at a time, the rows of x share every weight load
template <typename T>
void gemv_rows(device const T* w, device const T* x, device T* out, uint n_out, uint k_in, uint rows,
               uint tg, uint sg, uint lane)
{
    const uint first = (tg * 8 + sg) * PER_SIMD;
    float acc[PER_SIMD][MAX_ROWS] = {};

    for (uint k = lane * 16; k < k_in; k += 512) {
        float4 xs[MAX_ROWS][4];
        for (uint r = 0; r < MAX_ROWS; ++r) {
            if (r < rows) {
                device const vec<T, 4>* p = (device const vec<T, 4>*)(x + r * k_in + k);
                for (uint q = 0; q < 4; ++q) xs[r][q] = float4(p[q]);
            }
        }
        for (uint j = 0; j < PER_SIMD; ++j) {
            const uint n = first + j;
            if (n >= n_out) break;
            device const vec<T, 4>* p = (device const vec<T, 4>*)(w + ulong(n) * k_in + k);
            float4 ws[4];
            for (uint q = 0; q < 4; ++q) ws[q] = float4(p[q]);
            for (uint r = 0; r < MAX_ROWS; ++r)
                if (r < rows) acc[j][r] += dot(ws[0], xs[r][0]) + dot(ws[1], xs[r][1]) + dot(ws[2], xs[r][2]) + dot(ws[3], xs[r][3]);
        }
    }

    for (uint j = 0; j < PER_SIMD; ++j) {
        const uint n = first + j;
        for (uint r = 0; r < MAX_ROWS; ++r) {
            const float s = simd_sum(acc[j][r]);
            if (lane == 0 && n < n_out && r < rows) out[r * n_out + n] = T(s);
        }
    }
}

#define GEMV_ROWS(NAME, T)                                                                     \
kernel void NAME(device const T* w [[buffer(0)]], device const T* x [[buffer(1)]],            \
                 device T* out [[buffer(2)]], constant uint& n_out [[buffer(3)]],              \
                 constant uint& k_in [[buffer(4)]], constant uint& rows [[buffer(5)]],         \
                 uint tg [[threadgroup_position_in_grid]],                                     \
                 uint sg [[simdgroup_index_in_threadgroup]],                                   \
                 uint lane [[thread_index_in_simdgroup]])                                      \
{                                                                                              \
    gemv_rows<T>(w, x, out, n_out, k_in, rows, tg, sg, lane);                                  \
}

GEMV_ROWS(gemv_rows_f16, half)
GEMV_ROWS(gemv_rows_f32, float)
"#;

    /// f16 at 0, f32 at 1; compiled once on first use, there is only ever one Metal device here.
    static PIPELINES: OnceLock<Result<[ComputePipeline; 2], String>> = OnceLock::new();

    /// Vector loads need 4-element aligned starts and k in steps of 16; anything else (and every
    /// other row count) stays on candle's own matmul.
    pub(super) fn fits(x: &Tensor, w: &Tensor) -> bool {
        let (Ok((rows, k)), Ok((_, wk))) = (x.dims2(), w.dims2()) else {
            return false;
        };
        let aligned = |t: &Tensor| t.is_contiguous() && t.layout().start_offset() % 4 == 0;
        x.device().is_metal()
            && (2..=MAX_ROWS).contains(&rows)
            && k == wk
            && k % 16 == 0
            && matches!(x.dtype(), DType::F16 | DType::F32)
            && x.dtype() == w.dtype()
            && aligned(x)
            && aligned(w)
    }

    fn pipelines(device: &MetalDevice) -> candle_core::Result<&'static [ComputePipeline; 2]> {
        PIPELINES
            .get_or_init(|| compile(device.metal_device()).map_err(|e| e.to_string()))
            .as_ref()
            .map_err(|e| candle_core::Error::Msg(format!("gemv_rows kernel: {e}")))
    }

    fn compile(device: &Device) -> Result<[ComputePipeline; 2], MetalKernelError> {
        let library = device.new_library_with_source(SOURCE, None)?;
        let load = |name| device.new_compute_pipeline_state_with_function(&library.get_function(name, None)?);
        Ok([load("gemv_rows_f16")?, load("gemv_rows_f32")?])
    }

    pub(super) struct GemvRows;

    impl CustomOp2 for GemvRows {
        fn name(&self) -> &'static str {
            "gemv-rows"
        }

        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
            bail!("gemv-rows only runs on Metal")
        }

        fn metal_fwd(
            &self,
            x: &MetalStorage,
            x_layout: &Layout,
            w: &MetalStorage,
            w_layout: &Layout,
        ) -> candle_core::Result<(MetalStorage, Shape)> {
            let (rows, k) = x_layout.shape().dims2()?;
            let n = w_layout.dims()[0];
            let dtype = x.dtype();
            let device = x.device();
            let pipeline = &pipelines(device)?[usize::from(dtype == DType::F32)];
            let out = device.new_buffer(rows * n, dtype, "gemv-rows")?;

            let size = dtype.size_in_bytes();
            let guard = device.command_encoder()?;
            let encoder: &ComputeCommandEncoder = guard.as_ref();
            encoder.set_label("gemv-rows");
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_input_buffer(0, Some(w.buffer()), w_layout.start_offset() * size);
            encoder.set_input_buffer(1, Some(x.buffer()), x_layout.start_offset() * size);
            encoder.set_output_buffer(2, Some(&out), 0);
            encoder.set_bytes(3, &(n as u32));
            encoder.set_bytes(4, &(k as u32));
            encoder.set_bytes(5, &(rows as u32));
            encoder.dispatch_thread_groups(
                MTLSize { width: n.div_ceil(OUTPUTS_PER_GROUP), height: 1, depth: 1 },
                MTLSize { width: THREADS, height: 1, depth: 1 },
            );
            drop(guard);

            Ok((MetalStorage::new(out, device.clone(), rows * n, dtype), Shape::from((rows, n))))
        }
    }
}

