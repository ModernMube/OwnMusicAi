using YuE2.Song;

namespace OwnMusicAI.Engine;

/// <summary>
/// How YuE2 gets its score: none, a melody-only one, or one with chord symbols.
/// </summary>
public enum ScoreMode { Off, Melody, Full }

/// <summary>
/// One AR phase's sampling knobs, the part of Protocol worth exposing.
/// </summary>
public sealed record SamplingSettings(double Temperature, double TopP, int TopK, double RepetitionPenalty)
{
    public static SamplingSettings ScoreDefaults => _from(Protocol.Abc);

    public static SamplingSettings MusicDefaults => _from(Protocol.Semantic);

    internal SamplingParams Apply(SamplingParams p) =>
        p with { Temperature = Temperature, TopP = TopP, TopK = TopK, RepetitionPenalty = RepetitionPenalty };

    static SamplingSettings _from(SamplingParams p) => new SamplingSettings(p.Temperature, p.TopP, p.TopK, p.RepetitionPenalty);
}

/// <summary>
/// Everything one song needs. Saved next to the song as job.json, so it doubles as the recipe
/// for a remix.
/// </summary>
public sealed class SongRequest
{
    public string Title { get; set; } = "";
    public string Style { get; set; } = "";
    public string Lyrics { get; set; } = "";
    public long Seed { get; set; } = 831001;
    public ScoreMode Mode { get; set; } = ScoreMode.Full;

    /// <summary>Fixed ABC score (reference song or hand edited). Null lets YuE2 plan its own.</summary>
    public string? Abc { get; set; }

    /// <summary>The recording the score came from - only kept for the library card.</summary>
    public string? ReferenceAudio { get; set; }

    public double? Cfg { get; set; }
    public double? MaxSeconds { get; set; }
    public int? MaxAbcTokens { get; set; }
    public int OdeSteps { get; set; } = Protocol.OdeSteps;
    public int Context { get; set; } = Protocol.Context;
    public int VaeTile { get; set; } = 256;
    public SamplingSettings? ScoreSampling { get; set; }
    public SamplingSettings? MusicSampling { get; set; }

    internal string Cot => Mode.ToString().ToLowerInvariant();

    public static double DefaultCfg(ScoreMode mode) => Protocol.DefaultCfg(mode.ToString().ToLowerInvariant());

    public SongRequest Clone() => (SongRequest)MemberwiseClone();
}

public enum GenerationStage { Waiting, Transcribing, LoadingModel, PlanningScore, GeneratingMusic, SynthesizingAudio, DecodingAudio, Finished }

/// <summary>
/// Fraction is 0..1 inside the current stage.
/// </summary>
public readonly record struct GenerationProgress(GenerationStage Stage, double Fraction, string Detail);
