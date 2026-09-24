namespace OwnMusicAI.Engine;

/// <summary>
/// Suno-style macro sliders (0..100) mapped onto the real knobs. 50 / 0 / 50 is exactly the model's
/// own defaults, so nothing changes until you touch a slider.
/// </summary>
public sealed record CreativeMix(double Weirdness, double StyleInfluence, double Variety)
{
    public static CreativeMix Default => new CreativeMix(50, 0, 50);

    /// <summary>Weirdness: temperature, top-p, top-k. Variety: repetition penalty.</summary>
    public SamplingSettings Music
    {
        get
        {
            double w = _centered(Weirdness);
            return new SamplingSettings(
                Math.Round(1.0 + w * 0.6, 2),
                Math.Round(Math.Clamp(0.95 + w * 0.08, 0.05, 1), 2),
                (int)Math.Round(100 * Math.Pow(2, w * 4)),
                Math.Round(1.2 + _centered(Variety) * 0.3, 3));
        }
    }

    /// <summary>The score gets the same push, just gentler - a wild ABC stops being valid ABC fast.</summary>
    public SamplingSettings Score
    {
        get
        {
            double w = _centered(Weirdness);
            return new SamplingSettings(
                Math.Round(0.7 + w * 0.6, 2),
                Math.Round(0.9 + w * 0.16, 2),
                (int)Math.Round(30 * Math.Pow(2, w * 2)),
                Math.Round(Math.Max(1.0, 1.005 + _centered(Variety) * 0.03), 3));
        }
    }

    /// <summary>
    /// CFG 1..3, null at 0 = the mode's auto value. Anything above auto runs a second, negative
    /// pass on every music token.
    /// </summary>
    public double? Cfg(ScoreMode mode) =>
        StyleInfluence <= 0 ? null : Math.Round(Math.Max(SongRequest.DefaultCfg(mode), 1 + StyleInfluence / 50), 2);

    public void ApplyTo(SongRequest r)
    {
        r.MusicSampling = Music;
        r.ScoreSampling = Score;
        r.Cfg = Cfg(r.Mode);
    }

    /// <summary>
    /// Rough way back from a recipe (library reuse), the music knobs decide.
    /// </summary>
    public static CreativeMix From(SongRequest r)
    {
        var m = r.MusicSampling ?? SamplingSettings.MusicDefaults;
        double _style = r.Cfg is double cfg && cfg > SongRequest.DefaultCfg(r.Mode) ? (cfg - 1) * 50 : 0;
        return new CreativeMix(_slider(50 + (m.Temperature - 1.0) / 0.6 * 100), _slider(_style), _slider(50 + (m.RepetitionPenalty - 1.2) / 0.3 * 100));
    }

    static double _centered(double slider) => Math.Clamp(slider, 0, 100) / 100 - 0.5;

    static double _slider(double v) => Math.Round(Math.Clamp(v, 0, 100));
}
