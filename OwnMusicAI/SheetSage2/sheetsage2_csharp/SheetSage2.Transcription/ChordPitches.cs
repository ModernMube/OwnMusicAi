using SheetSage2.Notation;

namespace SheetSage2;

/// <summary>
/// Chord label -> MIDI pitches for the playback track, following mir_eval's encode: the chord
/// tones in the octave from C3, with the inversion bass an octave below.
/// </summary>
internal static class ChordPitches
{
    static readonly Dictionary<string, int[]> QualityIntervals = new Dictionary<string, int[]>
    {
        { "maj", new[] { 0, 4, 7 } },
        { "min", new[] { 0, 3, 7 } },
        { "dim", new[] { 0, 3, 6 } },
        { "aug", new[] { 0, 4, 8 } },
        { "maj7", new[] { 0, 4, 7, 11 } },
        { "min7", new[] { 0, 3, 7, 10 } },
        { "7", new[] { 0, 4, 7, 10 } },
        { "hdim7", new[] { 0, 3, 6, 10 } },
        { "dim7", new[] { 0, 3, 6, 9 } },
        { "minmaj7", new[] { 0, 3, 7, 11 } },
        { "sus2", new[] { 0, 2, 7 } },
        { "sus4", new[] { 0, 5, 7 } },
        { "sus4(b7)", new[] { 0, 5, 7, 10 } },
        { "maj6", new[] { 0, 4, 7, 9 } },
        { "min6", new[] { 0, 3, 7, 9 } }
    };

    static readonly Dictionary<string, int> BassDegrees = new Dictionary<string, int>
    {
        { "2", 2 }, { "b3", 3 }, { "3", 4 }, { "5", 7 }, { "b7", 10 }, { "7", 11 }
    };

    public static List<int> Of(string label)
    {
        label = label.Trim();
        if (label is "N" or "X" or "?" || !label.Contains(':')) return new List<int>();

        string _root = label[..label.IndexOf(':')];
        string _descriptor = label[(label.IndexOf(':') + 1)..];
        string _quality = _descriptor.Contains('/') ? _descriptor[.._descriptor.IndexOf('/')] : _descriptor;
        string? _bass = _descriptor.Contains('/') ? _descriptor[(_descriptor.IndexOf('/') + 1)..] : null;
        if (!QualityIntervals.TryGetValue(_quality, out var intervals))
            throw new ChordSymbolException($"Chord playback does not know quality '{_quality}'");

        int _rootClass = MusicSymbols.Spell(_root).PitchClass;
        int _bassSemitone = 0;
        if (_bass != null && !BassDegrees.TryGetValue(_bass, out _bassSemitone))
            throw new ChordSymbolException($"Chord playback does not know bass degree '{_bass}'");

        var _semitones = intervals.Append(_bassSemitone % 12).Distinct().Order();
        var _pitches = _semitones.Select(s => 48 + _rootClass + s).Append(36 + (_rootClass + _bassSemitone) % 12);
        return _pitches.Distinct().Order().ToList();
    }
}
