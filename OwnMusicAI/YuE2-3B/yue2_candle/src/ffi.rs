//! C ABI for the C# host. Every call returns 0 on success and -1 on failure, with the message
//! waiting in [`yue2_last_error`] on the same thread. Panics never cross the boundary.

use std::{
    cell::RefCell,
    ffi::{CStr, CString, c_char},
    panic::{self, AssertUnwindSafe},
    path::PathBuf,
    slice,
};

use candle_core::DType;

use crate::{Backend, Cache, EngineError, Result, VaeDecoder, Yue2Model};

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

/// What the host gets back from `yue2_engine_create`.
pub struct Engine {
    model: Yue2Model,
    vae: Option<VaeDecoder>,
    backend: Backend,
}

#[repr(C)]
pub struct Yue2Info {
    pub vocab_size: i32,
    pub latent_dim: i32,
    pub layers: i32,
    pub context: i32,
    pub sample_rate: i32,
    pub downsampling_ratio: i32,
    /// 1 = f32, 2 = f16, 3 = bf16
    pub dtype: i32,
    /// 0 = cpu, 1 = metal, 2 = cuda
    pub backend: i32,
}

fn guarded(body: impl FnOnce() -> Result<()>) -> i32 {
    let failure = match panic::catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => return 0,
        Ok(Err(e)) => e.to_string(),
        Err(payload) => payload
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "panic inside yue2_engine".into()),
    };
    LAST_ERROR.with(|slot| *slot.borrow_mut() = CString::new(failure.replace('\0', " ")).unwrap_or_default());
    -1
}

fn invalid(message: &str) -> EngineError {
    EngineError::Invalid(message.into())
}

fn dir_arg(p: *const c_char) -> Result<Option<PathBuf>> {
    if p.is_null() {
        return Ok(None);
    }
    // SAFETY: host hands us a NUL-terminated UTF-8 string that outlives the call.
    let text = unsafe { CStr::from_ptr(p) }.to_str().map_err(|_| invalid("path is not UTF-8"))?;
    Ok(Some(PathBuf::from(text)))
}

/// Message of the last failed call on this thread; valid until the next failing call.
#[unsafe(no_mangle)]
pub extern "C" fn yue2_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}

/// Loads YuE2-3B (and optionally the VAE decoder). `backend`: 0 cpu, 1 metal, 2 cuda;
/// `dtype`: 0 = backend default, 1 f32, 2 f16, 3 bf16. The VAE always runs in f32.
///
/// # Safety
/// `model_dir` must be a valid C string, `vae_dir` a valid C string or null, `out` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn yue2_engine_create(
    model_dir: *const c_char,
    vae_dir: *const c_char,
    backend: i32,
    dtype: i32,
    out: *mut *mut Engine,
) -> i32 {
    guarded(|| {
        let backend = match backend {
            0 => Backend::Cpu,
            1 => Backend::Metal,
            2 => Backend::Cuda,
            _ => return Err(invalid("backend must be 0 (cpu), 1 (metal) or 2 (cuda)")),
        };
        let dtype = match dtype {
            0 => backend.default_dtype(),
            1 => DType::F32,
            2 => DType::F16,
            3 => DType::BF16,
            _ => return Err(invalid("dtype must be 0..3")),
        };
        let model_dir = dir_arg(model_dir)?.ok_or_else(|| invalid("model_dir is null"))?;
        let device = backend.device()?;
        let model = Yue2Model::load(&model_dir, &device, dtype)?;
        let vae = dir_arg(vae_dir)?.map(|dir| VaeDecoder::load(&dir, &device)).transpose()?;
        let engine = Box::into_raw(Box::new(Engine { model, vae, backend }));
        // SAFETY: caller promised `out` is writable.
        unsafe { *out = engine };
        Ok(())
    })
}

/// # Safety
/// `engine` must come from `yue2_engine_create` (or be null) and not be used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn yue2_engine_destroy(engine: *mut Engine) {
    if !engine.is_null() {
        // SAFETY: we handed this Box out in yue2_engine_create.
        drop(unsafe { Box::from_raw(engine) });
    }
}

/// # Safety
/// `engine` valid, `info` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn yue2_engine_info(engine: *const Engine, info: *mut Yue2Info) -> i32 {
    guarded(|| {
        // SAFETY: pointers checked by the caller contract above.
        let (engine, info) = unsafe { (engine.as_ref().ok_or_else(|| invalid("engine is null"))?, &mut *info) };
        let cfg = engine.model.config();
        let vae = engine.vae.as_ref().map(|v| v.config());
        *info = Yue2Info {
            vocab_size: cfg.vocab_size as i32,
            latent_dim: cfg.latent_dim as i32,
            layers: cfg.num_hidden_layers as i32,
            context: cfg.max_position_embeddings as i32,
            sample_rate: vae.map_or(0, |v| v.sample_rate as i32),
            downsampling_ratio: vae.map_or(0, |v| v.downsampling_ratio as i32),
            dtype: match engine.model.dtype() {
                DType::F16 => 2,
                DType::BF16 => 3,
                _ => 1,
            },
            backend: engine.backend as i32,
        };
        Ok(())
    })
}

/// New empty KV cache; `capacity` tokens are reserved on first use (it grows past that).
///
/// # Safety
/// `engine` valid, `out` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn yue2_cache_create(engine: *const Engine, capacity: i32, out: *mut *mut Cache) -> i32 {
    guarded(|| {
        // SAFETY: caller contract.
        let engine = unsafe { engine.as_ref() }.ok_or_else(|| invalid("engine is null"))?;
        let cache = Box::into_raw(Box::new(engine.model.new_cache(capacity.max(1) as usize)));
        // SAFETY: caller contract.
        unsafe { *out = cache };
        Ok(())
    })
}

/// # Safety
/// `cache` from `yue2_cache_create` or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn yue2_cache_destroy(cache: *mut Cache) {
    if !cache.is_null() {
        // SAFETY: Box handed out by yue2_cache_create.
        drop(unsafe { Box::from_raw(cache) });
    }
}

/// Tokens already in the cache, -1 for a null handle.
///
/// # Safety
/// `cache` valid or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn yue2_cache_len(cache: *const Cache) -> i32 {
    // SAFETY: caller contract.
    unsafe { cache.as_ref() }.map_or(-1, |c| c.len() as i32)
}

/// AR step: appends `count` tokens to `cache`, last-position logits land in `logits` (vocab_size).
///
/// # Safety
/// All pointers valid for the given lengths; `cache` not shared with another running call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn yue2_ar_forward(
    engine: *const Engine,
    cache: *mut Cache,
    tokens: *const u32,
    count: i32,
    logits: *mut f32,
    logits_len: i32,
) -> i32 {
    guarded(|| {
        if tokens.is_null() || logits.is_null() || count <= 0 || logits_len <= 0 {
            return Err(invalid("ar_forward got an empty buffer"));
        }
        // SAFETY: caller contract, lengths checked positive above.
        let (engine, cache, tokens, logits) = unsafe {
            (
                engine.as_ref().ok_or_else(|| invalid("engine is null"))?,
                cache.as_mut().ok_or_else(|| invalid("cache is null"))?,
                slice::from_raw_parts(tokens, count as usize),
                slice::from_raw_parts_mut(logits, logits_len as usize),
            )
        };
        engine.model.forward(cache, tokens, logits)
    })
}

/// NAR velocity for one chunk. `latents`/`velocity` are [frames * latent_dim] row-major.
///
/// # Safety
/// All pointers valid for `frames * latent_dim` floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn yue2_nar_velocity(
    engine: *const Engine,
    prefix: *const Cache,
    latents: *const f32,
    frames: i32,
    t_logit: f32,
    velocity: *mut f32,
) -> i32 {
    guarded(|| {
        // SAFETY: caller contract.
        let engine = unsafe { engine.as_ref() }.ok_or_else(|| invalid("engine is null"))?;
        let n = frames.max(0) as usize * engine.model.config().latent_dim;
        if latents.is_null() || velocity.is_null() || n == 0 {
            return Err(invalid("nar_velocity got an empty buffer"));
        }
        // SAFETY: caller contract, n > 0.
        let (prefix, latents, velocity) = unsafe {
            (
                prefix.as_ref().ok_or_else(|| invalid("prefix cache is null"))?,
                slice::from_raw_parts(latents, n),
                slice::from_raw_parts_mut(velocity, n),
            )
        };
        engine.model.velocity(prefix, latents, t_logit, velocity)
    })
}

/// Samples the VAE produces for `frames` latent frames (1920 * frames - 64), -1 without a VAE.
///
/// # Safety
/// `engine` valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn yue2_vae_output_len(engine: *const Engine, frames: i32) -> i64 {
    // SAFETY: caller contract.
    unsafe { engine.as_ref() }
        .and_then(|e| e.vae.as_ref())
        .map_or(-1, |vae| vae.output_len(frames.max(0) as usize) as i64)
}

/// Decodes [frames * 64] latents into planar stereo (`2 * output_len` floats, left then right).
///
/// # Safety
/// `latents` valid for `frames * 64`, `audio` for `audio_len` floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn yue2_vae_decode(
    engine: *const Engine,
    latents: *const f32,
    frames: i32,
    audio: *mut f32,
    audio_len: i64,
) -> i32 {
    guarded(|| {
        // SAFETY: caller contract.
        let engine = unsafe { engine.as_ref() }.ok_or_else(|| invalid("engine is null"))?;
        let vae = engine.vae.as_ref().ok_or_else(|| invalid("engine was created without a VAE"))?;
        let n = frames.max(0) as usize * vae.config().latent_dim;
        if latents.is_null() || audio.is_null() || n == 0 || audio_len <= 0 {
            return Err(invalid("vae_decode got an empty buffer"));
        }
        // SAFETY: caller contract, sizes positive.
        let (latents, audio) =
            unsafe { (slice::from_raw_parts(latents, n), slice::from_raw_parts_mut(audio, audio_len as usize)) };
        vae.decode(latents, audio)
    })
}
