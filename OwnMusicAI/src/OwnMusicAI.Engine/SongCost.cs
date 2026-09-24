using SheetSage2;
using YuE2.Song;

namespace OwnMusicAI.Engine;

public enum CostLevel { Light, Heavy, TooMuch }

/// <summary>
/// Back-of-the-envelope memory + time check of a request before it goes into the queue.
/// Numbers come from the YuE2-3B shape (28 layers, 8 KV heads x 128), good to ~20%, not more.
/// </summary>
public sealed record SongCost(double PeakGb, double MachineGb, string PeakPhase, CostLevel Level, IReadOnlyList<string> Warnings)
{
    const double Params = 3.63e9;
    const double VaeBytes = 0.5e9;
    const int Layers = 28, KvHeads = 8, HeadDim = 128, Heads = 16, Hidden = 2048, Inter = 6144;
    const int MaxContext = 24576;
    const int SamplesPerFrame = 1920;
    const double Gb = 1024.0 * 1024 * 1024;

    public string Summary => $"~{PeakGb:F1} GB peak ({PeakPhase}) of {MachineGb:F0} GB";

    /// <summary>What this machine has, the GC reports the physical RAM (or the container limit).</summary>
    public static double MachineMemoryGb => GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / Gb;

    public static SongCost Estimate(SongRequest r, ComputeBackend backend, ComputeDType dtype)
    {
        int _bytes = dtype switch
        {
            ComputeDType.F32 => 4,
            ComputeDType.Auto => backend == ComputeBackend.Cpu ? 4 : 2,
            _ => 2
        };
        double _kvToken = (double)Layers * 2 * KvHeads * HeadDim * _bytes;
        bool _metal = backend == ComputeBackend.Metal;

        int _text = (r.Style.Length + r.Lyrics.Length) / 3 + 60;
        int _abc = r.Mode == ScoreMode.Off ? 0 : r.Abc != null ? r.Abc.Length / 2 : Math.Min(r.MaxAbcTokens ?? 4096, 2048);
        int _prefix = _text + _abc;
        int _music = Math.Max(0, Math.Min(r.MaxSeconds is double sec ? (int)(sec * 25) : 9000, MaxContext - _prefix));

        double _defaultCfg = SongRequest.DefaultCfg(r.Mode);
        bool _cfg = (r.Cfg ?? _defaultCfg) != 1.0;
        bool _extraCfg = _cfg && _defaultCfg == 1.0;

        //non-Metal attention builds [heads, 512, keys] score blocks
        double _attn(int keys) => _metal ? 0 : (double)Heads * 512 * keys * 4;

        double _ar = (_prefix + _music) * _kvToken * (_cfg ? 2 : 1) + _attn(_prefix + _music);

        int _chunk = Math.Min(Math.Max(1, (r.Context - _prefix - 3) / 2), Math.Max(1, _music));
        int _keys = _prefix + 2 * _chunk + 3;
        double _nar = _keys * _kvToken
            + (_chunk + 2.0) * (Hidden * 6 + Inter * 3) * _bytes
            + 2.0 * _keys * KvHeads * HeadDim * _bytes
            + _attn(_keys);

        double _vae = (Math.Min(r.VaeTile, Math.Max(1, _music)) + 32.0) * SamplesPerFrame * 64 * 7 * 4;
        double _host = _music * (double)SamplesPerFrame * 2 * 4 * 2;

        string _phase = "music generation";
        double _top = _ar;
        if (_nar > _top) { _top = _nar; _phase = "audio synthesis"; }
        if (_vae > _top) { _top = _vae; _phase = "VAE decoding"; }

        double _peak = (Params * _bytes + VaeBytes + _top + _host) / Gb;
        double _machine = MachineMemoryGb;
        //Apple Silicon lets Metal have ~75% of the unified memory
        double _limit = _machine * (_metal ? 0.75 : 0.85);

        var _warnings = new List<string>();
        var _level = CostLevel.Light;

        if (_peak > _limit)
        {
            _level = CostLevel.TooMuch;
            _warnings.Add($"Needs ~{_peak:F1} GB, more than the ~{_limit:F0} GB this machine can give the engine - expect heavy swapping or an out-of-memory crash. {_fix(_phase)}");
        }
        else if (_peak > _machine * 0.5)
        {
            _level = CostLevel.Heavy;
            _warnings.Add($"Needs ~{_peak:F1} GB of {_machine:F0} GB - close other big apps. {_fix(_phase)}");
        }

        if (_extraCfg)
            _warnings.Add($"Style influence above 0 runs every music token twice: ~2× longer music generation, +{_music * _kvToken / Gb:F1} GB.");
        if (r.OdeSteps > Protocol.OdeSteps)
            _warnings.Add($"{r.OdeSteps} flow-matching steps: audio synthesis takes ~{r.OdeSteps / (double)Protocol.OdeSteps:F1}× longer.");
        if (_bytes == 4 && backend != ComputeBackend.Cpu)
            _warnings.Add("F32 weights: twice the memory and about half the speed of F16/BF16.");
        if (backend == ComputeBackend.Cpu)
            _warnings.Add("CPU backend: every core is busy, a full song can take hours.");

        if (_level == CostLevel.Light && _warnings.Count > 0) _level = CostLevel.Heavy;
        return new SongCost(_peak, _machine, _phase, _level, _warnings);
    }

    static string _fix(string phase) => phase switch
    {
        "music generation" => "Shorten the song or lower Style influence.",
        "audio synthesis" => "Lower the acoustic context.",
        _ => "Lower the VAE tile."
    };
}
