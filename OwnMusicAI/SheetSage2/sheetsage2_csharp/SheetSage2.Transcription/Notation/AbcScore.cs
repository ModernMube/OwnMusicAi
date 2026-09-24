namespace SheetSage2.Notation;

public readonly record struct BeatEvent(double Time, int BeatId, int DeclaredNumerator, int Denominator, int LineNo);

/// <summary>An interval annotation: chord, key or structure label over [Start, End).</summary>
public readonly record struct Interval(double Start, double End, string Label);

/// <summary>A note as the notation side sees it: seconds, MIDI pitch, voice (0 vocal, 1 ins).</summary>
public readonly record struct TimedNote(double Start, double End, int Pitch, int Track);

/// <summary>
/// One measure between two downbeats. Numerator/Denominator is what the beats actually show,
/// the notated pair is what ABC writes when a partial bar gets padded with rests.
/// </summary>
public sealed record Measure(int Index, int StartBeat, int EndBeat, int Numerator, int Denominator,
    bool Pickup = false, bool Partial = false, bool Inferred = false,
    int? NotatedNumerator = null, int? NotatedDenominator = null, bool PadBefore = false)
{
    public int StartT => StartBeat * AbcScore.SubbeatDivision;
    public int EndT => EndBeat * AbcScore.SubbeatDivision;
    public int AbcNumerator => NotatedNumerator ?? Numerator;
    public int AbcDenominator => NotatedDenominator ?? Denominator;
}

/// <summary>
/// The quantized score behind the ABC: beat grid, per-subbeat key/chord/voice arrays and the
/// measures inferred from the decoded downbeats.
/// </summary>
public sealed class AbcScore
{
    public const int SubbeatDivision = 4;
    public static readonly string[] VoiceIds = { "Vocal", "Ins" };

    public List<BeatEvent> Beats = new List<BeatEvent>();
    public List<Measure> Measures = new List<Measure>();
    public double[] SubbeatTimes = Array.Empty<double>();
    public double[] SubbeatQuarters = Array.Empty<double>();
    public int[] SubbeatDenominators = Array.Empty<int>();
    public string[] KeyArr = Array.Empty<string>();
    public string[] ChordArr = Array.Empty<string>();
    public List<(int T, string Label)> StructureEvents = new List<(int, string)>();
    public Dictionary<string, int[]> VoiceArrs = new Dictionary<string, int[]>();
    public List<string> Diagnostics = new List<string>();

    /// <summary>
    /// Builds the score from decoded beats, interval annotations and the (already quantized)
    /// melody notes. melodyOnly drops chords, which also removes chord-only render boundaries.
    /// </summary>
    public static AbcScore Build(IReadOnlyList<BeatEvent> beats, IReadOnlyList<Interval> chords, IReadOnlyList<Interval> keys,
        IReadOnlyList<Interval> structures, IReadOnlyList<TimedNote> notes, bool melodyOnly)
    {
        var _beats = ParseBeats(beats);
        var _keys = ParseIntervals(keys, "keys", key => MusicSymbols.KeySymbolToAbc(key));
        if (_keys.Count == 0) throw new AbcRebuildException("keys: at least one key interval is required");
        var _structures = ParseIntervals(structures, "structures", label => label);
        var _chords = melodyOnly ? new List<Interval>() : ParseIntervals(chords, "chords", chord =>
        {
            MusicSymbols.ChordSymbolToAbc(chord);
            return chord;
        });

        var (measures, diagnostics) = InferMeasures(_beats);
        var _score = new AbcScore { Beats = _beats, Measures = measures, Diagnostics = diagnostics };
        _score._buildGrid();

        foreach (var voice in VoiceIds)
            _score.VoiceArrs[voice] = _score._notesToArray(notes.Where(n => n.Track == Array.IndexOf(VoiceIds, voice)), voice);

        _score.KeyArr = _score._fillIntervals(_keys, _keys[0].Label);
        _score.ChordArr = melodyOnly
            ? Enumerable.Repeat("N", _score.SubbeatTimes.Length).ToArray()
            : _score._fillIntervals(_chords, "N");
        _score.StructureEvents = _structures
            .Select(row => (Math.Clamp(_score.QuantizeTime(row.Start), 0, _score.SubbeatTimes.Length - 1), row.Label))
            .ToList();
        return _score;
    }

    public static List<BeatEvent> ParseBeats(IReadOnlyList<BeatEvent> rows)
    {
        var _beats = new List<BeatEvent>();
        foreach (var beat in rows)
        {
            if (beat.BeatId < 1) throw new BeatGridException($"beats:{beat.LineNo}: beat ID must be positive");
            if (beat.DeclaredNumerator < 1) throw new BeatGridException($"beats:{beat.LineNo}: meter numerator must be positive");
            if (beat.Denominator < 1 || (beat.Denominator & (beat.Denominator - 1)) != 0)
                throw new BeatGridException($"beats:{beat.LineNo}: meter denominator must be a positive power of two");
            if (_beats.Count > 0 && beat.Time <= _beats[^1].Time)
                throw new BeatGridException($"beats:{beat.LineNo}: beat times must be strictly increasing");
            _beats.Add(beat);
        }
        if (_beats.Count < 2) throw new BeatGridException("beats: at least two beat events are required");
        return _beats;
    }

    static List<Interval> ParseIntervals(IReadOnlyList<Interval> rows, string source, Func<string, string> normalize)
    {
        var _rows = new List<Interval>();
        double? _previousEnd = null;
        int _line = 0;
        foreach (var row in rows)
        {
            _line++;
            if (row.End <= row.Start) throw new AbcRebuildException($"{source}:{_line}: interval end must be after start");
            if (_previousEnd is double previous && row.Start < previous - 1e-6)
                throw new AbcRebuildException($"{source}:{_line}: overlapping intervals");
            _rows.Add(row with { Label = normalize(row.Label.Trim()) });
            _previousEnd = row.End;
        }
        return _rows;
    }

    /// <summary>Measures between actual downbeats; declared meters only break ties.</summary>
    public static (List<Measure>, List<string>) InferMeasures(IReadOnlyList<BeatEvent> beats)
    {
        var _downbeats = Enumerable.Range(0, beats.Count).Where(i => beats[i].BeatId == 1).ToList();
        if (_downbeats.Count == 0) throw new BeatGridException("No downbeat (beat ID 1) exists in the beat lab");

        var _spans = new List<(int Start, int End, bool Pickup, bool Partial)>();
        if (_downbeats[0] > 0) _spans.Add((0, _downbeats[0], true, false));
        for (int i = 0; i + 1 < _downbeats.Count; i++) _spans.Add((_downbeats[i], _downbeats[i + 1], false, false));
        //an exported beat lab ends on a boundary row; a non-downbeat last row means a truncated bar
        if (_downbeats[^1] < beats.Count - 1) _spans.Add((_downbeats[^1], beats.Count - 1, false, true));
        if (_spans.Count == 0) throw new BeatGridException("No positive-length measure exists between downbeats");

        var _diagnostics = new List<string>();
        var _measures = new List<Measure>();
        for (int index = 0; index < _spans.Count; index++)
        {
            var (start, end, pickup, partial) = _spans[index];
            var _events = beats.Skip(start).Take(end - start).ToList();
            int _count = _events.Count;
            if (_count < 1) throw new BeatGridException($"Measure {index}: empty downbeat span");
            var _ids = _events.Select(e => e.BeatId).ToList();
            if (!_ids.SequenceEqual(Enumerable.Range(_ids[0], _count)))
                throw new BeatGridException($"Measure {index} (beat rows {_events[0].LineNo}-{_events[^1].LineNo}): " +
                    $"non-consecutive beat IDs [{string.Join(", ", _ids)}]");
            if (!pickup && _ids[0] != 1) throw new BeatGridException($"Measure {index}: full measure does not start at beat ID 1");

            var _denominators = _events.Select(e => e.Denominator).ToList();
            int _denominator = _mode(_denominators);
            var _numerators = _events.Select(e => e.DeclaredNumerator).ToList();
            int _declared = _mode(_numerators);
            bool _numeratorConflict = _numerators.Any(v => v != _count);
            bool _denominatorConflict = _denominators.Any(v => v != _denominator);
            bool _padFinal = partial && _numerators.Distinct().Count() == 1 && !_denominatorConflict && _declared >= _count;
            bool _inferred = pickup || partial || _numeratorConflict || _denominatorConflict;

            if (_padFinal && _declared > _count)
                _diagnostics.Add($"measure {index}: padded final {_count}/{_denominator} span to declared {_declared}/{_denominator} with trailing rest");
            else if (_numeratorConflict)
                _diagnostics.Add($"measure {index}: inferred {_count}/{_denominator} from downbeat span; declared numerators were [{string.Join(", ", _numerators)}]");
            if (_denominatorConflict)
                _diagnostics.Add($"measure {index}: placed denominator {_denominator} at the measure boundary; row declarations were [{string.Join(", ", _denominators)}]");

            _measures.Add(new Measure(index, start, end, _count, _denominator, pickup, partial, _inferred,
                NotatedNumerator: _padFinal ? _declared : _count));
        }

        if (_measures.Count >= 2)
        {
            var _first = _measures[0];
            var _next = _measures[1];
            if (_first.Numerator / (double)_first.Denominator < _next.AbcNumerator / (double)_next.AbcDenominator)
            {
                _measures[0] = _first with
                {
                    Inferred = true,
                    NotatedNumerator = _next.AbcNumerator,
                    NotatedDenominator = _next.AbcDenominator,
                    PadBefore = true
                };
                _diagnostics.Add($"measure 0: padded leading {_first.Numerator}/{_first.Denominator} span " +
                    $"to {_next.AbcNumerator}/{_next.AbcDenominator} with preceding rest");
            }
        }
        return (_measures, _diagnostics);
    }

    /// <summary>Subbeat index of a time, snapping on the midpoints between subbeats.</summary>
    public int QuantizeTime(double time)
    {
        int _index = Array.BinarySearch(_boundaries, time);
        return _index >= 0 ? _index : ~_index;
    }

    double[] _boundaries = Array.Empty<double>();

    void _buildGrid()
    {
        var _denominators = new int[Beats.Count - 1];
        foreach (var measure in Measures)
            for (int i = measure.StartBeat; i < measure.EndBeat; i++) _denominators[i] = measure.Denominator;
        if (_denominators.Any(d => d == 0)) throw new BeatGridException("Downbeat spans do not cover every beat interval");

        var _times = new List<double>();
        var _subbeatDenominators = new List<int>();
        var _quarters = new List<double> { 0.0 };
        double _quarter = 0.0;
        for (int i = 0; i < Beats.Count - 1; i++)
        {
            double _start = Beats[i].Time;
            double _step = (Beats[i + 1].Time - _start) / SubbeatDivision;
            for (int s = 0; s < SubbeatDivision; s++)
            {
                _times.Add(s * _step + _start);
                _subbeatDenominators.Add(_denominators[i]);
                _quarter += 4.0 / _denominators[i] / SubbeatDivision;
                _quarters.Add(_quarter);
            }
        }
        _times.Add(Beats[^1].Time);
        _subbeatDenominators.Add(_denominators[^1]);

        SubbeatTimes = _times.ToArray();
        SubbeatQuarters = _quarters.ToArray();
        SubbeatDenominators = _subbeatDenominators.ToArray();
        _boundaries = Enumerable.Range(0, SubbeatTimes.Length - 1).Select(i => (SubbeatTimes[i] + SubbeatTimes[i + 1]) / 2).ToArray();
    }

    string[] _fillIntervals(IReadOnlyList<Interval> rows, string fallback)
    {
        var _result = Enumerable.Repeat(fallback, SubbeatTimes.Length).ToArray();
        foreach (var row in rows)
        {
            int _start = Math.Clamp(QuantizeTime(row.Start), 0, _result.Length - 1);
            int _end = Math.Clamp(QuantizeTime(row.End), 0, _result.Length - 1);
            if (_end <= _start)
                throw new AbcRebuildException(FormattableString.Invariant(
                    $"Interval {row.Start:F6}-{row.End:F6} ({row.Label}) is shorter than the ABC subbeat grid"));
            for (int t = _start; t < _end; t++) _result[t] = row.Label;
        }
        if (_result.Length > 1) _result[^1] = _result[^2];
        return _result;
    }

    /// <summary>Per-subbeat voice track: 0 rest, pitch*2+2 sustain, +1 on the onset subbeat.</summary>
    int[] _notesToArray(IEnumerable<TimedNote> notes, string voiceId)
    {
        var _result = new int[SubbeatTimes.Length];
        foreach (var note in notes.OrderBy(n => n.Start).ThenBy(n => n.End).ThenBy(n => n.Pitch))
        {
            int _start = Math.Clamp(QuantizeTime(note.Start), 0, _result.Length - 1);
            int _end = Math.Clamp(QuantizeTime(note.End), 0, _result.Length - 1);
            if (_end <= _start)
                throw new MelodyVoiceException(FormattableString.Invariant(
                    $"{voiceId}: MIDI note pitch={note.Pitch} at {note.Start:F6}-{note.End:F6} cannot be represented on the decoded subbeat grid"));
            for (int t = _start; t < _end; t++)
                if (_result[t] != 0) throw new MelodyVoiceException($"{voiceId}: overlapping quantized melody notes at subbeats {_start}:{_end}");

            int _sustain = note.Pitch * 2 + 2;
            for (int t = _start; t < _end; t++) _result[t] = _sustain;
            _result[_start] = _sustain + 1;
        }
        return _result;
    }

    static int _mode(IReadOnlyList<int> values)
    {
        var _counts = values.GroupBy(v => v).ToDictionary(g => g.Key, g => g.Count());
        int _max = _counts.Values.Max();
        return values.First(v => _counts[v] == _max);
    }
}
