using System.Runtime.InteropServices;
using System.Text.Json;

namespace SheetSage2;

public enum ComputeBackend { Cpu = 0, Metal = 1, Cuda = 2 }

public enum ComputeDType { Auto = 0, F32 = 1, F16 = 2, BF16 = 3 }

/// <summary>
/// Encoder output of one 300 s window, lives inside the engine (GPU memory on Metal/CUDA).
/// </summary>
public sealed class EncoderMemory : IDisposable
{
    internal readonly MemoryHandle Handle;

    internal EncoderMemory(MemoryHandle handle)
    {
        Handle = handle;
    }

    public void Dispose() => Handle.Dispose();
}

/// <summary>
/// Decoder self-attention cache of one token sequence.
/// </summary>
public sealed class DecoderCache : IDisposable
{
    internal readonly CacheHandle Handle;

    internal DecoderCache(CacheHandle handle)
    {
        Handle = handle;
    }

    public int Length => Native.ss2_cache_len(Handle);

    public void Dispose() => Handle.Dispose();
}

/// <summary>
/// Managed side of the Rust/Candle engine: loads SheetSage2 (+ the MERT-v2 parent it merges),
/// encodes windows and runs decoder steps. Grammar, windowing and exports are the caller's job.
/// </summary>
public sealed unsafe class SheetSage2Engine : IDisposable
{
    readonly EngineHandle _handle;

    public EngineInfo Info { get; }

    /// <summary>mertDir can stay null for a merged (save_pretrained) checkpoint.</summary>
    public SheetSage2Engine(string modelDir, string? mertDir, ComputeBackend backend, ComputeDType dtype)
    {
        _check(Native.ss2_engine_create(modelDir, mertDir, (int)backend, (int)dtype, out _handle));
        _check(Native.ss2_engine_info(_handle, out var _info));
        Info = _info;
    }

    /// <summary>
    /// Mono samples at Info.SampleRate, at most one window; the engine pads the rest with silence.
    /// </summary>
    public EncoderMemory Encode(ReadOnlySpan<float> samples)
    {
        fixed (float* s = samples)
        {
            _check(Native.ss2_encode(_handle, s, samples.Length, out var _memory));
            return new EncoderMemory(_memory);
        }
    }

    /// <summary>encoder_last_hidden_state, [EncoderFrames * HiddenSize] row-major.</summary>
    public float[] Features(EncoderMemory memory)
    {
        var _features = new float[(long)Info.EncoderFrames * Info.HiddenSize];
        fixed (float* f = _features)
        {
            _check(Native.ss2_memory_features(memory.Handle, f, _features.Length));
        }
        return _features;
    }

    public DecoderCache NewCache(int capacity)
    {
        _check(Native.ss2_cache_create(_handle, capacity, out var _cache));
        return new DecoderCache(_cache);
    }

    /// <summary>
    /// Appends tokens to the cache, last-position logits go to <paramref name="logits"/> (vocab sized).
    /// </summary>
    public void Decode(EncoderMemory memory, DecoderCache cache, ReadOnlySpan<int> tokens, Span<float> logits)
    {
        fixed (int* t = tokens)
        fixed (float* l = logits)
        {
            _check(Native.ss2_decode(_handle, memory.Handle, cache.Handle, (uint*)t, tokens.Length, l, logits.Length));
        }
    }

    public void Dispose() => _handle.Dispose();

    /// <summary>
    /// The pinned m-a-p/MERT-v2-FullSong snapshot in the Hugging Face cache (revision from the
    /// model's config.json), or any snapshot holding weights if that one is missing.
    /// </summary>
    public static string? FindMert(string modelDir)
    {
        string _home = Environment.GetEnvironmentVariable("HF_HOME")
            ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface");
        string _snapshots = Path.Combine(_home, "hub", "models--m-a-p--MERT-v2-FullSong", "snapshots");
        if (!Directory.Exists(_snapshots)) return null;

        using (var _doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(modelDir, "config.json"))))
        {
            if (_doc.RootElement.TryGetProperty("base_model_revision", out var _rev))
            {
                string _pinned = Path.Combine(_snapshots, _rev.GetString()!);
                if (File.Exists(Path.Combine(_pinned, "model.safetensors"))) return _pinned;
            }
        }
        return Directory.GetDirectories(_snapshots).FirstOrDefault(d => File.Exists(Path.Combine(d, "model.safetensors")));
    }

    static void _check(int status)
    {
        if (status != 0) throw new InvalidOperationException("sheetsage2_engine: " + Marshal.PtrToStringUTF8(Native.ss2_last_error()));
    }
}
