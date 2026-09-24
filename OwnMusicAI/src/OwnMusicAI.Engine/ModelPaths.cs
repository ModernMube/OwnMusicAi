using SheetSage2;

namespace OwnMusicAI.Engine;

/// <summary>
/// Where the weights live and what runs them. Discover() fills it from the repo layout
/// and the Hugging Face cache.
/// </summary>
public sealed class ModelPaths
{
    public string YuE2Dir { get; set; } = "";

    /// <summary>m-a-p/YuE2-Vae, null = look it up in the HF cache.</summary>
    public string? VaeDir { get; set; }

    public string SheetSageDir { get; set; } = "";

    public ComputeBackend Backend { get; set; } = OperatingSystem.IsMacOS() ? ComputeBackend.Metal : ComputeBackend.Cpu;

    public ComputeDType DType { get; set; } = ComputeDType.Auto;

    /// <summary>
    /// Walks up from the exe until it finds the folder holding YuE2-3B and SheetSage2.
    /// Only the folder counts, the weights may not be downloaded yet.
    /// </summary>
    public static ModelPaths Discover()
    {
        var _paths = new ModelPaths { VaeDir = FindVae() };
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
        {
            if (!Directory.Exists(Path.Combine(dir.FullName, "YuE2-3B"))) continue;
            _paths.YuE2Dir = Path.Combine(dir.FullName, "YuE2-3B");
            _paths.SheetSageDir = Path.Combine(dir.FullName, "SheetSage2");
            break;
        }
        return _paths;
    }

    public static string? FindVae()
    {
        string _snapshots = Path.Combine(HfHome(), "hub", "models--m-a-p--YuE2-Vae", "snapshots");
        if (!Directory.Exists(_snapshots)) return null;
        return Directory.GetDirectories(_snapshots).FirstOrDefault(d => File.Exists(Path.Combine(d, "model.safetensors")));
    }

    /// <summary>What is missing for a song, null when everything is in place.</summary>
    public string? SongProblem()
    {
        if (_noEngine("yue2_engine")) return "libyue2_engine is missing next to the app: run `cargo build --release --features metal` in YuE2-3B/yue2_candle, then rebuild";
        if (!File.Exists(Path.Combine(YuE2Dir, "config.json"))) return $"YuE2-3B not found at '{YuE2Dir}' (Settings › Download models)";
        if (!File.Exists(Path.Combine(YuE2Dir, "model.safetensors"))) return "The YuE2-3B weights are missing (Settings › Download models)";
        if (!File.Exists(Path.Combine(YuE2Dir, "qwen.tiktoken"))) return "qwen.tiktoken is missing from the YuE2-3B folder (Settings › Download models)";
        if ((VaeDir ?? FindVae()) == null) return "YuE2-Vae not found (Settings › Download models)";
        return null;
    }

    /// <summary>Same for the reference song path (SheetSage2 + MERT).</summary>
    public string? AnalysisProblem()
    {
        if (_noEngine("sheetsage2_engine")) return "libsheetsage2_engine is missing next to the app: run `cargo build --release --features metal` in SheetSage2/sheetsage2_candle, then rebuild";
        if (!File.Exists(Path.Combine(SheetSageDir, "config.json"))) return $"SheetSage2 not found at '{SheetSageDir}' (Settings › Download models)";
        if (!File.Exists(Path.Combine(SheetSageDir, "model.safetensors"))) return "The SheetSage2 weights are missing (Settings › Download models)";
        if (SheetSage2Engine.FindMert(SheetSageDir) == null) return "MERT-v2-FullSong not found (Settings › Download models)";
        return null;
    }

    /// <summary>The Rust dylib the build should have copied next to the exe.</summary>
    static bool _noEngine(string name) =>
        !new[] { $"lib{name}.dylib", $"lib{name}.so", $"{name}.dll" }.Any(f => File.Exists(Path.Combine(AppContext.BaseDirectory, f)));

    /// <summary>HF_HOME, or ~/.cache/huggingface like the hf CLI.</summary>
    public static string HfHome() => Environment.GetEnvironmentVariable("HF_HOME")
        ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface");
}
