//! C ABI for the C# host. Every call returns 0 on success and -1 on failure, with the message
//! waiting in [`ss2_last_error`] on the same thread. Panics never cross the boundary.

use std::{
    cell::RefCell,
    ffi::{CStr, CString, c_char},
    panic::{self, AssertUnwindSafe},
    path::PathBuf,
    slice,
};

use candle_core::DType;

use crate::{Backend, DecoderCache, EngineError, Memory, Result, SheetSage2};

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

/// What the host gets back from `ss2_engine_create`.
pub struct Engine {
    model: SheetSage2,
    backend: Backend,
}

#[repr(C)]
pub struct Ss2Info {
    pub vocab_size: i32,
    pub sample_rate: i32,
    /// samples in one model window (300 s)
    pub window_samples: i32,
    /// shortest window the encoder accepts
    pub min_samples: i32,
    pub encoder_frames: i32,
    pub hidden_size: i32,
    /// decoder context, prompt prefix included
    pub max_tokens: i32,
    pub time_hz: i32,
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
            .unwrap_or_else(|| "panic inside sheetsage2_engine".into()),
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
pub extern "C" fn ss2_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}

/// Loads SheetSage2. `mert_dir` may be null for a merged checkpoint. `backend`: 0 cpu, 1 metal,
/// 2 cuda; `dtype`: 0 = backend default, 1 f32, 2 f16, 3 bf16.
///
/// # Safety
/// `model_dir` a valid C string, `mert_dir` a valid C string or null, `out` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ss2_engine_create(
    model_dir: *const c_char,
    mert_dir: *const c_char,
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
        let mert_dir = dir_arg(mert_dir)?;
        let model = SheetSage2::load(&model_dir, mert_dir.as_deref(), &backend.device()?, dtype)?;
        let engine = Box::into_raw(Box::new(Engine { model, backend }));
        // SAFETY: caller promised `out` is writable.
        unsafe { *out = engine };
        Ok(())
    })
}

/// # Safety
/// `engine` from `ss2_engine_create` (or null), not used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ss2_engine_destroy(engine: *mut Engine) {
    if !engine.is_null() {
        // SAFETY: we handed this Box out in ss2_engine_create.
        drop(unsafe { Box::from_raw(engine) });
    }
}

/// # Safety
/// `engine` valid, `info` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ss2_engine_info(engine: *const Engine, info: *mut Ss2Info) -> i32 {
    guarded(|| {
        // SAFETY: caller contract.
        let (engine, info) = unsafe { (engine.as_ref().ok_or_else(|| invalid("engine is null"))?, &mut *info) };
        let cfg = engine.model.config();
        let window = cfg.window_samples();
        *info = Ss2Info {
            vocab_size: cfg.vocab_size as i32,
            sample_rate: cfg.sampling_rate as i32,
            window_samples: window as i32,
            min_samples: (cfg.backbone_config.n_fft / 2 + 1) as i32,
            encoder_frames: (window / cfg.backbone_config.samples_per_frame()) as i32,
            hidden_size: cfg.hidden_size as i32,
            max_tokens: cfg.max_output_seq_len as i32,
            time_hz: cfg.time_hz as i32,
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

/// Encodes one mono window (`count` samples at the model rate, zero-padded to the window).
///
/// # Safety
/// `samples` valid for `count` floats, `out` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ss2_encode(engine: *const Engine, samples: *const f32, count: i64, out: *mut *mut Memory) -> i32 {
    guarded(|| {
        if samples.is_null() || count <= 0 {
            return Err(invalid("encode got an empty buffer"));
        }
        // SAFETY: caller contract, count checked positive.
        let (engine, samples) =
            unsafe { (engine.as_ref().ok_or_else(|| invalid("engine is null"))?, slice::from_raw_parts(samples, count as usize)) };
        let memory = Box::into_raw(Box::new(engine.model.encode(samples)?));
        // SAFETY: caller contract.
        unsafe { *out = memory };
        Ok(())
    })
}

/// # Safety
/// `memory` from `ss2_encode` or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ss2_memory_destroy(memory: *mut Memory) {
    if !memory.is_null() {
        // SAFETY: Box handed out by ss2_encode.
        drop(unsafe { Box::from_raw(memory) });
    }
}

/// Copies the encoder output (`encoder_frames * hidden_size` floats, row-major) - the
/// `encoder_last_hidden_state` embeddings of the window.
///
/// # Safety
/// `memory` valid, `features` valid for `len` floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ss2_memory_features(memory: *const Memory, features: *mut f32, len: i64) -> i32 {
    guarded(|| {
        // SAFETY: caller contract.
        let memory = unsafe { memory.as_ref() }.ok_or_else(|| invalid("memory is null"))?;
        let values = memory.features()?;
        if features.is_null() || len as usize != values.len() {
            return Err(EngineError::Invalid(format!("features buffer must hold {} floats", values.len())));
        }
        // SAFETY: length checked against the tensor above.
        unsafe { slice::from_raw_parts_mut(features, values.len()) }.copy_from_slice(&values);
        Ok(())
    })
}

/// New empty decoder cache; `capacity` tokens are reserved up front (it grows past that).
///
/// # Safety
/// `engine` valid, `out` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ss2_cache_create(engine: *const Engine, capacity: i32, out: *mut *mut DecoderCache) -> i32 {
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
/// `cache` from `ss2_cache_create` or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ss2_cache_destroy(cache: *mut DecoderCache) {
    if !cache.is_null() {
        // SAFETY: Box handed out by ss2_cache_create.
        drop(unsafe { Box::from_raw(cache) });
    }
}

/// Tokens already in the cache, -1 for a null handle.
///
/// # Safety
/// `cache` valid or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ss2_cache_len(cache: *const DecoderCache) -> i32 {
    // SAFETY: caller contract.
    unsafe { cache.as_ref() }.map_or(-1, |c| c.len() as i32)
}

/// Decoder step: appends `count` tokens to `cache` (attending to `memory`), last-position logits
/// go to `logits` (vocab_size floats).
///
/// # Safety
/// All pointers valid for the given lengths; `cache` not shared with another running call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ss2_decode(
    engine: *const Engine,
    memory: *const Memory,
    cache: *mut DecoderCache,
    tokens: *const u32,
    count: i32,
    logits: *mut f32,
    logits_len: i32,
) -> i32 {
    guarded(|| {
        if tokens.is_null() || logits.is_null() || count <= 0 || logits_len <= 0 {
            return Err(invalid("decode got an empty buffer"));
        }
        // SAFETY: caller contract, lengths checked positive above.
        let (engine, memory, cache, tokens, logits) = unsafe {
            (
                engine.as_ref().ok_or_else(|| invalid("engine is null"))?,
                memory.as_ref().ok_or_else(|| invalid("memory is null"))?,
                cache.as_mut().ok_or_else(|| invalid("cache is null"))?,
                slice::from_raw_parts(tokens, count as usize),
                slice::from_raw_parts_mut(logits, logits_len as usize),
            )
        };
        engine.model.decode(memory, cache, tokens, logits)
    })
}
