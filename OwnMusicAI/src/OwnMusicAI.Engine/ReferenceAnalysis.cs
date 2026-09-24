using System.Globalization;
using SheetSage2;

namespace OwnMusicAI.Engine;

public readonly record struct TimedLabel(double Start, double End, string Label);

/// <summary>
/// What SheetSage2 heard in a reference song, boiled down for the Advanced page: the ABC score
/// YuE2 will follow, plus key / tempo / chords / sections to show next to it.
/// </summary>
public sealed class ReferenceAnalysis
{
    public string? Abc { get; init; }
    public string? AbcError { get; init; }
    public int Measures { get; init; }
    public double DurationSeconds { get; init; }
    public string? Key { get; init; }
    public double? Bpm { get; init; }
    public string? Meter { get; init; }
    public IReadOnlyList<TimedLabel> Chords { get; init; } = Array.Empty<TimedLabel>();
    public IReadOnlyList<TimedLabel> Sections { get; init; } = Array.Empty<TimedLabel>();
    public IReadOnlyList<string> Warnings { get; init; } = Array.Empty<string>();
    public double ElapsedSeconds { get; init; }

    internal static ReferenceAnalysis From(TranscriptionResult r)
    {
        var _beats = _table(r, "beat.lab");
        double? _bpm = null;
        string? _meter = null;
        if (_beats.Count > 1)
        {
            double _first = _num(_beats[0][0]), _last = _num(_beats[^1][0]);
            if (_last > _first) _bpm = Math.Round(60.0 * (_beats.Count - 1) / (_last - _first), 1);
            if (_beats[0].Length >= 4) _meter = $"{_beats[0][2]}/{_beats[0][3]}";
        }

        //The key that covers the most time wins, modulations just show up as runners-up
        string? _key = _intervals(r, "key.lab")
            .GroupBy(k => k.Label)
            .OrderByDescending(g => g.Sum(k => k.End - k.Start))
            .Select(g => g.Key)
            .FirstOrDefault();

        return new ReferenceAnalysis
        {
            Abc = r.Abc,
            AbcError = r.AbcError,
            Measures = r.AbcMeasures,
            DurationSeconds = r.DurationSeconds,
            Key = _key,
            Bpm = _bpm,
            Meter = _meter,
            Chords = _intervals(r, "chord.lab"),
            Sections = _intervals(r, "structure.lab"),
            Warnings = r.Warnings.ToList(),
            ElapsedSeconds = r.ElapsedSeconds
        };
    }

    static List<TimedLabel> _intervals(TranscriptionResult r, string lab) =>
        _table(r, lab).Where(c => c.Length >= 3).Select(c => new TimedLabel(_num(c[0]), _num(c[1]), c[2])).ToList();

    static List<string[]> _table(TranscriptionResult r, string lab)
    {
        if (!r.Labs.TryGetValue(lab, out var _text)) return new List<string[]>();
        return _text.Split('\n', StringSplitOptions.RemoveEmptyEntries).Select(l => l.Split('\t')).ToList();
    }

    static double _num(string s) => double.Parse(s, CultureInfo.InvariantCulture);
}
