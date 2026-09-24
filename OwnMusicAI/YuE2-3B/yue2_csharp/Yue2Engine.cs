using System.Runtime.InteropServices;

namespace YuE2.Song;

internal enum EngineBackend { Cpu = 0, Metal = 1, Cuda = 2 }

internal enum EngineDType { Auto = 0, F32 = 1, F16 = 2, BF16 = 3 }

/// <summary>
/// One sequence's KV cache living inside the engine (GPU memory on Metal/CUDA).
/// </summary>
internal sealed class KvCache : IDisposable
{
    internal readonly CacheHandle Handle;

    internal KvCache(CacheHandle handle)
    {
        Handle = handle;
    }

    public int Length => Native.yue2_cache_len(Handle);

    public void Dispose() => Handle.Dispose();
}

/// <summary>
/// Managed side of the Rust/Candle engine. Owns the loaded YuE2 weights (+ VAE decoder),
/// everything above raw tensor math is done by the caller.
/// </summary>
internal sealed unsafe class Yue2Engine : IDisposable
{
    readonly EngineHandle _handle;

    public Yue2Info Info { get; }

    public Yue2Engine(string modelDir, string? vaeDir, EngineBackend backend, EngineDType dtype)
    {
        _check(Native.yue2_engine_create(modelDir, vaeDir, (int)backend, (int)dtype, out _handle));
        _check(Native.yue2_engine_info(_handle, out var _info));
        Info = _info;
    }

    /// <summary>capacity = tokens reserved up front, it still grows if you go past it.</summary>
    public KvCache NewCache(int capacity)
    {
        _check(Native.yue2_cache_create(_handle, capacity, out var _cache));
        return new KvCache(_cache);
    }

    /// <summary>
    /// Appends tokens to the cache, last-position logits go to <paramref name="logits"/> (vocab sized).
    /// </summary>
    public void Forward(KvCache cache, ReadOnlySpan<int> tokens, Span<float> logits)
    {
        fixed (int* t = tokens)
        fixed (float* l = logits)
        {
            _check(Native.yue2_ar_forward(_handle, cache.Handle, (uint*)t, tokens.Length, l, logits.Length));
        }
    }

    public void Velocity(KvCache prefix, ReadOnlySpan<float> latents, float tLogit, Span<float> velocity)
    {
        int _frames = latents.Length / Info.LatentDim;
        fixed (float* x = latents)
        fixed (float* v = velocity)
        {
            _check(Native.yue2_nar_velocity(_handle, prefix.Handle, x, _frames, tLogit, v));
        }
    }

    public long VaeOutputLength(int frames) => Native.yue2_vae_output_len(_handle, frames);

    /// <summary>Planar output: left samples first, then right.</summary>
    public void DecodeVae(ReadOnlySpan<float> latents, Span<float> planarAudio)
    {
        int _frames = latents.Length / Info.LatentDim;
        fixed (float* x = latents)
        fixed (float* a = planarAudio)
        {
            _check(Native.yue2_vae_decode(_handle, x, _frames, a, planarAudio.Length));
        }
    }

    public void Dispose() => _handle.Dispose();

    static void _check(int status)
    {
        if (status != 0) throw new InvalidOperationException("yue2_engine: " + Marshal.PtrToStringUTF8(Native.yue2_last_error()));
    }
}
