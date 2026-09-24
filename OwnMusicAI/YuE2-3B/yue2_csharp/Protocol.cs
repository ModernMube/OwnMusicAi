namespace YuE2.Song;

/// <summary>
/// Sampling knobs for one AR phase (same fields as yue2.protocol.Sampling).
/// </summary>
internal sealed record SamplingParams(double Temperature, double TopP, int TopK, double RepetitionPenalty,
    int PenaltyWindow, int MinTokens, int MaxTokens);

/// <summary>
/// yue2-native-v1 protocol, copied from yue2/protocol.py of the 0.1.5 wheel.
/// </summary>
internal static class Protocol
{
    public const int Eod = 151643;
    public const int AbcStart = 151847;
    public const int AbcEnd = 151848;
    public const int MusicStart = 151851;
    public const int MusicEnd = 151852;
    public const int CodecOffset = 151853;
    public const int CodecSize = 32768;
    public const int Context = 24576;
    public const int OdeSteps = 32;

    public static readonly Dictionary<string, string> Instructions = new Dictionary<string, string>
    {
        { "off", "Generate music with codec tokens from the given conditions." },
        { "melody", "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions." },
        { "full", "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions." }
    };

    public static readonly SamplingParams Abc = new SamplingParams(0.7, 0.9, 30, 1.005, 100, 32, 4096);
    public static readonly SamplingParams Semantic = new SamplingParams(1.0, 0.95, 100, 1.2, 50, 200, 9000);

    /// <summary>Split regex of tokenization_yue2.py (Qwen style).</summary>
    public const string SplitPattern = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    public static double DefaultCfg(string cot) => cot == "off" ? 1.01 : 1.0;

    public static string PromptText(string cot, string style, string lyrics) =>
        $"{Instructions[cot]}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n";
}
