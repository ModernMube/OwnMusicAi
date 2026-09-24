using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace SheetSage2;

/// <summary>
/// Mirror of Ss2Info in sheetsage2_candle/src/ffi.rs - keep the field order in sync.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct EngineInfo
{
    public int VocabSize;
    public int SampleRate;
    public int WindowSamples;
    public int MinSamples;
    public int EncoderFrames;
    public int HiddenSize;
    public int MaxTokens;
    public int TimeHz;
    public int DType;
    public int Backend;
}

internal sealed class EngineHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public EngineHandle() : base(true) { }

    protected override bool ReleaseHandle()
    {
        Native.ss2_engine_destroy(handle);
        return true;
    }
}

internal sealed class MemoryHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public MemoryHandle() : base(true) { }

    protected override bool ReleaseHandle()
    {
        Native.ss2_memory_destroy(handle);
        return true;
    }
}

internal sealed class CacheHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public CacheHandle() : base(true) { }

    protected override bool ReleaseHandle()
    {
        Native.ss2_cache_destroy(handle);
        return true;
    }
}

/// <summary>
/// Raw C ABI of libsheetsage2_engine. Status 0 = ok, -1 = see ss2_last_error.
/// </summary>
internal static unsafe class Native
{
    const string Lib = "sheetsage2_engine";

    [DllImport(Lib)] public static extern IntPtr ss2_last_error();

    [DllImport(Lib)]
    public static extern int ss2_engine_create([MarshalAs(UnmanagedType.LPUTF8Str)] string modelDir,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? mertDir, int backend, int dtype, out EngineHandle engine);

    [DllImport(Lib)] public static extern void ss2_engine_destroy(IntPtr engine);
    [DllImport(Lib)] public static extern int ss2_engine_info(EngineHandle engine, out EngineInfo info);

    [DllImport(Lib)] public static extern int ss2_encode(EngineHandle engine, float* samples, long count, out MemoryHandle memory);
    [DllImport(Lib)] public static extern void ss2_memory_destroy(IntPtr memory);
    [DllImport(Lib)] public static extern int ss2_memory_features(MemoryHandle memory, float* features, long len);

    [DllImport(Lib)] public static extern int ss2_cache_create(EngineHandle engine, int capacity, out CacheHandle cache);
    [DllImport(Lib)] public static extern void ss2_cache_destroy(IntPtr cache);
    [DllImport(Lib)] public static extern int ss2_cache_len(CacheHandle cache);

    [DllImport(Lib)]
    public static extern int ss2_decode(EngineHandle engine, MemoryHandle memory, CacheHandle cache, uint* tokens, int count, float* logits, int logitsLen);
}
