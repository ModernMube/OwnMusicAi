//! SnakeBeta, `x + 1/beta * sin(alpha * x)^2`. Candle runs it as five kernels, two of them
//! slow broadcast ones, which made it ~40% of VAE decoding on Metal - so there it is one pass.

use candle_core::Tensor;

use crate::Result;

/// `x` is [b, c, l], `alpha` and `inv_beta` are [1, c, 1].
pub fn snake(x: &Tensor, alpha: &Tensor, inv_beta: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "metal")]
    if metal::fits(x, alpha, inv_beta) {
        return Ok(x.contiguous()?.apply_op3_no_bwd(alpha, inv_beta, &metal::Snake)?);
    }
    Ok((x + x.broadcast_mul(alpha)?.sin()?.sqr()?.broadcast_mul(inv_beta)?)?)
}

#[cfg(feature = "metal")]
mod metal {
    use std::sync::OnceLock;

    use candle_core::{
        CpuStorage, CustomOp3, DType, Layout, MetalDevice, MetalStorage, Shape, Tensor, backend::BackendStorage, bail,
    };
    use candle_metal_kernels::{
        MetalKernelError,
        metal::{ComputeCommandEncoder, ComputePipeline, Device},
    };
    use objc2_metal::MTLSize;

    const THREADS: usize = 256;

    const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void snake_f32(device const float* x [[buffer(0)]], device const float* alpha [[buffer(1)]],
                      device const float* inv_beta [[buffer(2)]], device float* out [[buffer(3)]],
                      constant uint& channels [[buffer(4)]], constant uint& len [[buffer(5)]],
                      constant uint& count [[buffer(6)]], uint i [[thread_position_in_grid]])
{
    if (i >= count) return;
    const uint c = (i / len) % channels;
    // precise: alpha * x gets large enough for fast-math sin to drift
    const float s = precise::sin(alpha[c] * x[i]);
    out[i] = x[i] + inv_beta[c] * s * s;
}
"#;

    static PIPELINE: OnceLock<Result<ComputePipeline, String>> = OnceLock::new();

    pub(super) fn fits(x: &Tensor, alpha: &Tensor, inv_beta: &Tensor) -> bool {
        let params = |t: &Tensor| t.dtype() == DType::F32 && t.is_contiguous() && t.elem_count() == x.dim(1).unwrap_or(0);
        x.device().is_metal() && x.rank() == 3 && x.dtype() == DType::F32 && params(alpha) && params(inv_beta)
    }

    fn pipeline(device: &MetalDevice) -> candle_core::Result<&'static ComputePipeline> {
        PIPELINE
            .get_or_init(|| compile(device.metal_device()).map_err(|e| e.to_string()))
            .as_ref()
            .map_err(|e| candle_core::Error::Msg(format!("snake kernel: {e}")))
    }

    fn compile(device: &Device) -> Result<ComputePipeline, MetalKernelError> {
        let library = device.new_library_with_source(SOURCE, None)?;
        device.new_compute_pipeline_state_with_function(&library.get_function("snake_f32", None)?)
    }

    pub(super) struct Snake;

    impl CustomOp3 for Snake {
        fn name(&self) -> &'static str {
            "snake"
        }

        fn cpu_fwd(
            &self,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> candle_core::Result<(CpuStorage, Shape)> {
            bail!("the fused snake only runs on Metal")
        }

        fn metal_fwd(
            &self,
            x: &MetalStorage,
            x_layout: &Layout,
            alpha: &MetalStorage,
            alpha_layout: &Layout,
            inv_beta: &MetalStorage,
            inv_beta_layout: &Layout,
        ) -> candle_core::Result<(MetalStorage, Shape)> {
            let (_, channels, len) = x_layout.shape().dims3()?;
            let count = x_layout.shape().elem_count();
            let device = x.device();
            let pipeline = pipeline(device)?;
            let out = device.new_buffer(count, DType::F32, "snake")?;

            let guard = device.command_encoder()?;
            let encoder: &ComputeCommandEncoder = guard.as_ref();
            encoder.set_label("snake");
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_input_buffer(0, Some(x.buffer()), x_layout.start_offset() * 4);
            encoder.set_input_buffer(1, Some(alpha.buffer()), alpha_layout.start_offset() * 4);
            encoder.set_input_buffer(2, Some(inv_beta.buffer()), inv_beta_layout.start_offset() * 4);
            encoder.set_output_buffer(3, Some(&out), 0);
            encoder.set_bytes(4, &(channels as u32));
            encoder.set_bytes(5, &(len as u32));
            encoder.set_bytes(6, &(count as u32));
            encoder.dispatch_thread_groups(
                MTLSize { width: count.div_ceil(THREADS), height: 1, depth: 1 },
                MTLSize { width: THREADS, height: 1, depth: 1 },
            );
            drop(guard);

            Ok((MetalStorage::new(out, device.clone(), count, DType::F32), x_layout.shape().clone()))
        }
    }
}
