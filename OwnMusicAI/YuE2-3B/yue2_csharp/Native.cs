using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace YuE2.Song;

/// <summary>
/// Mirror of Yue2Info in yue2_candle/src/ffi.rs - keep the field order in sync.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct Yue2Info
{
    public int VocabSize;
    public int LatentDim;
    public int Layers;
    public int Context;
    public int SampleRate;
    public int DownsamplingRatio;
    public int DType;
    public int Backend;
}

internal sealed class EngineHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public EngineHandle() : base(true) { }

    protected override bool ReleaseHandle()
    {
        Native.yue2_engine_destroy(handle);
        return true;
    }
}

internal sealed class CacheHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public CacheHandle() : base(true) { }

    protected override bool ReleaseHandle()
    {
        Native.yue2_cache_destroy(handle);
        return true;
    }
}

/// <summary>
/// Raw C ABI of libyue2_engine. Status 0 = ok, -1 = see yue2_last_error.
/// </summary>
internal static unsafe class Native
{
    const string Lib = "yue2_engine";

    [DllImport(Lib)] public static extern IntPtr yue2_last_error();

    [DllImport(Lib)]
    public static extern int yue2_engine_create([MarshalAs(UnmanagedType.LPUTF8Str)] string modelDir,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? vaeDir, int backend, int dtype, out EngineHandle engine);

    [DllImport(Lib)] public static extern void yue2_engine_destroy(IntPtr engine);
    [DllImport(Lib)] public static extern int yue2_engine_info(EngineHandle engine, out Yue2Info info);

    [DllImport(Lib)] public static extern int yue2_cache_create(EngineHandle engine, int capacity, out CacheHandle cache);
    [DllImport(Lib)] public static extern void yue2_cache_destroy(IntPtr cache);
    [DllImport(Lib)] public static extern int yue2_cache_len(CacheHandle cache);

    [DllImport(Lib)]
    public static extern int yue2_ar_forward(EngineHandle engine, CacheHandle cache, uint* tokens, int count, float* logits, int logitsLen);

    [DllImport(Lib)]
    public static extern int yue2_nar_velocity(EngineHandle engine, CacheHandle prefix, float* latents, int frames, float tLogit, float* velocity);

    [DllImport(Lib)] public static extern long yue2_vae_output_len(EngineHandle engine, int frames);

    [DllImport(Lib)]
    public static extern int yue2_vae_decode(EngineHandle engine, float* latents, int frames, float* audio, long audioLen);
}
