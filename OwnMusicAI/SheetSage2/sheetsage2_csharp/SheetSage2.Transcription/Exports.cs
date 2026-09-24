using System.Globalization;
using System.Text;
using System.Text.Json;
using SheetSage2.Notation;

namespace SheetSage2;

/// <summary>Everything a transcription produced, in memory. Files are optional.</summary>
public sealed class TranscriptionResult
{
    public string Audio = "";
    public double DurationSeconds;
    public IReadOnlyList<string> Prompts = Array.Empty<string>();
    public bool MelodyOnly;
    public string? Abc;
    public string? AbcError;
    public int AbcMeasures;
    public List<DecodedEvent> Events = new List<DecodedEvent>();
    public Dictionary<string, string> Labs = new Dictionary<string, string>();
    public Dictionary<string, byte[]> Midis = new Dictionary<string, byte[]>();
    public List<string> Warnings = new List<string>();
    public List<string> Diagnostics = new List<string>();
    public List<List<int>> WindowTokens = new List<List<int>>();
    public List<int> WindowPrefixLengths = new List<int>();
    public double ElapsedSeconds;

    /// <summary>Melody + chord playback, the transcription.mid of the reference.</summary>
    public byte[]? Midi => Midis.TryGetValue("transcription", out var bytes) ? bytes : null;
}

/// <summary>
/// Turns stitched events into the reference's output set: raw annotations (LAB/TSV/JSON), the
/// independently validated ABC score, and MIDI playback.
/// </summary>
internal static class Exports
{
    public static void Build(TranscriptionResult result, SheetSage2Tokenizer tokenizer, bool melodyOnly)
    {
        var _events = result.Events;
        double _duration = result.DurationSeconds;

        var _notes = new List<TimedNote>();
        foreach (var item in _events)
        {
            double _start = item.Time!.Value;
            foreach (var note in item.Values.Melody ?? Enumerable.Empty<MelodyNote>())
            {
                double _end = Math.Min(_duration, note.EndTime ?? _start);
                if (_end > _start) _notes.Add(new TimedNote(_start, _end, note.Pitch, note.Track));
            }
        }
        _notes = _notes.OrderBy(n => n.Start).ThenBy(n => n.End).ThenBy(n => n.Pitch).ThenBy(n => n.Track).ToList();

        result.Labs["events.tsv"] = _rows(new[] { new object[] { "time", "global_subbeat", "fields" } }
            .Concat(_events.Select(e => new object[] { e.Time!.Value, e.GlobalSubbeat, _fieldText(e) })));
        result.Labs["melody_full.lab"] = _rows(_notes.Select(n => new object[] { n.Start, n.End, n.Pitch, n.Track }));
        foreach (var (track, name) in new[] { (0, "vocal"), (1, "instrumental") })
        {
            result.Labs[$"melody_{name}.lab"] = _rows(_notes.Where(n => n.Track == track).Select(n => new object[] { n.Start, n.End, n.Pitch }));
            result.Midis[$"melody_{name}"] = MidiExport.Write(_melodyTracks(_notes.Where(n => n.Track == track).ToList()));
        }
        result.Midis["melody"] = MidiExport.Write(_melodyTracks(_notes));

        var _intervals = new Dictionary<EventField, List<Interval>>();
        foreach (var (field, name) in new[] { (EventField.Chord, "chord"), (EventField.Key, "key"), (EventField.Structure, "structure") })
        {
            _intervals[field] = IntervalRows(_events, field, _duration);
            result.Labs[$"{name}.lab"] = _rows(_intervals[field].Select(r => new object[] { r.Start, r.End, r.Label }));
        }
        result.Labs["rhythm_events.lab"] = _rows(_events
            .Where(e => e.Values.Rhythm != null || e.Values.Timestamp != null)
            .Select(e => new object[] { e.Time!.Value, _rhythmJson(e.Values.Rhythm) }));

        AbcScore? _score = null;
        try
        {
            var _beats = BeatRows(_events);
            result.Labs["beat.lab"] = _rows(_beats.Select(b => new object[] { b.Time, b.BeatId, b.DeclaredNumerator, b.Denominator }));
            result.Labs["downbeat.lab"] = _rows(_beats.Where(b => b.BeatId == 1).Select(b => new object[] { b.Time }));
            if (_beats.Count < 2) throw new AbcRebuildException("At least two decoded beats are required for ABC");

            var _abcBeats = _extendBeats(_beats, _duration, _notes);
            result.Labs["notation/song_beats.txt"] = _rows(_abcBeats.Select(b => new object[] { b.Time, b.BeatId, b.DeclaredNumerator, b.Denominator }));
            var _clipped = new Dictionary<EventField, List<Interval>>();
            foreach (var (field, name) in new[] { (EventField.Chord, "chords"), (EventField.Key, "keys"), (EventField.Structure, "structures") })
            {
                _clipped[field] = _clip(IntervalRows(_events, field, _duration), _abcBeats[0].Time, _abcBeats[^1].Time);
                result.Labs[$"notation/song_{name}.txt"] = _rows(_clipped[field].Select(r => new object[] { r.Start, r.End, r.Label }));
            }
            if (_intervals[EventField.Key].Count == 0)
                throw new AbcRebuildException("No key was decoded; cannot construct a keyed ABC score");

            var (clean, adjustments) = NotationNotes(_notes);
            result.Diagnostics.AddRange(adjustments);
            _score = AbcScore.Build(_abcBeats, _clipped[EventField.Chord], _clipped[EventField.Key], _clipped[EventField.Structure],
                MidiExport.Quantize(clean), melodyOnly);
            result.Diagnostics.AddRange(_score.Diagnostics);
            result.Abc = AbcSerializer.Serialize(_score);
            result.AbcMeasures = _score.Measures.Count;
        }
        catch (Exception exception) when (exception is AbcRebuildException or FormatException)
        {
            result.AbcError = exception.Message;
            result.AbcMeasures = 0;
        }

        _buildPlayback(result, melodyOnly ? new List<Interval>() : _intervals[EventField.Chord], _score, _duration, _notes);
    }

    /// <summary>Interval rows of a field: each event's value until the next one that has it.</summary>
    public static List<Interval> IntervalRows(List<DecodedEvent> events, EventField field, double duration)
    {
        var _rows = events.Where(e => e.Values.Has(field))
            .Select(e => (Time: e.Time!.Value, Label: _label(e.Values, field)))
            .ToList();
        var _intervals = new List<Interval>();
        for (int i = 0; i < _rows.Count; i++)
        {
            double _end = i + 1 < _rows.Count ? _rows[i + 1].Time : duration;
            if (_end > _rows[i].Time) _intervals.Add(new Interval(_rows[i].Time, _end, _rows[i].Label));
        }
        return _intervals;
    }

    /// <summary>
    /// Beat rows from the rhythm events: the meter carries over, the eighth-note position has to
    /// land on that meter's beat grid.
    /// </summary>
    public static List<BeatEvent> BeatRows(List<DecodedEvent> events)
    {
        var _rows = new List<BeatEvent>();
        (int Numerator, int Denominator)? _meter = null;
        foreach (var item in events)
        {
            var _rhythm = item.Values.Rhythm;
            if (_rhythm?.Meter != null) _meter = _rhythm.Meter;
            if (_rhythm?.EighthPosition is not int eighth || _meter is not (int numerator, int denominator)) continue;

            //the model stores an eighth-note position, not a beat index in the meter's denominator
            if (eighth * denominator % 8 != 0)
                throw new AbcRebuildException($"Eighth position {eighth} is off the {numerator}/{denominator} beat grid");
            int _position = eighth * denominator / 8;
            if (_position < 0 || _position >= numerator)
                throw new AbcRebuildException($"Eighth position {eighth} is outside meter ({numerator}, {denominator})");
            _rows.Add(new BeatEvent(item.Time!.Value, _position + 1, numerator, denominator, _rows.Count + 1));
        }
        return _rows;
    }

    /// <summary>
    /// A monophonic view for notation: notes are clipped to the next onset of their own voice,
    /// the raw prediction stays untouched in the LAB/MIDI outputs.
    /// </summary>
    public static (List<TimedNote>, List<string>) NotationNotes(List<TimedNote> notes)
    {
        var _result = new List<TimedNote>();
        var _diagnostics = new List<string>();
        foreach (int track in new[] { 0, 1 })
        {
            var _ordered = notes.Where(n => n.Track == track).OrderBy(n => n.Start).ThenBy(n => n.Pitch).ThenBy(n => n.End).ToList();
            for (int i = 0; i < _ordered.Count; i++)
            {
                var _note = _ordered[i];
                if (i + 1 < _ordered.Count && _note.End > _ordered[i + 1].Start + 1e-6)
                {
                    _note = _note with { End = _ordered[i + 1].Start };
                    _diagnostics.Add(FormattableString.Invariant($"notation only: clipped track {track} note at {_note.Start:F3} to next onset"));
                }
                if (_note.End > _note.Start + 1e-6) _result.Add(_note);
            }
        }
        return (_result.OrderBy(n => n.Start).ThenBy(n => n.End).ThenBy(n => n.Pitch).ThenBy(n => n.Track).ToList(), _diagnostics);
    }

    /// <summary>events.json, the lossless decoded representation.</summary>
    public static string EventsJson(TranscriptionResult result)
    {
        var _buffer = new MemoryStream();
        using (var writer = new Utf8JsonWriter(_buffer, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            writer.WriteString("schema_version", "v1");
            writer.WriteStartArray("prompts");
            foreach (var prompt in result.Prompts) writer.WriteStringValue(prompt);
            writer.WriteEndArray();
            writer.WriteStartArray("events");
            foreach (var item in result.Events)
            {
                writer.WriteStartObject();
                writer.WriteNumber("subbeat", item.Subbeat);
                writer.WriteStartObject("tokens_by_field");
                foreach (var field in item.Fields)
                {
                    writer.WriteStartArray(_fieldName(field));
                    foreach (int token in item[field]!) writer.WriteNumberValue(token);
                    writer.WriteEndArray();
                }
                writer.WriteEndObject();
                writer.WriteStartObject("values");
                _writeValues(writer, item.Values);
                writer.WriteEndObject();
                writer.WriteNumber("time", item.Time!.Value);
                writer.WriteNumber("window_index", item.WindowIndex);
                writer.WriteNumber("window_start", item.WindowStart);
                writer.WriteNumber("source_subbeat", item.SourceSubbeat);
                writer.WriteNumber("global_subbeat", item.GlobalSubbeat);
                writer.WriteEndObject();
            }
            writer.WriteEndArray();
            writer.WriteBoolean("has_eos", true);
            writer.WriteEndObject();
        }
        return Encoding.UTF8.GetString(_buffer.ToArray());
    }

    static void _writeValues(Utf8JsonWriter writer, EventValues values)
    {
        if (values.Timestamp is double stamp) writer.WriteNumber("timestamp", stamp);
        if (values.Rhythm is RhythmValue rhythm)
        {
            writer.WriteStartObject("rhythm");
            if (rhythm.Meter is (int numerator, int denominator))
            {
                writer.WriteStartArray("meter");
                writer.WriteNumberValue(numerator);
                writer.WriteNumberValue(denominator);
                writer.WriteEndArray();
            }
            if (rhythm.EighthPosition is int eighth) writer.WriteNumber("eighth_position", eighth);
            writer.WriteEndObject();
        }
        if (values.Structure != null) writer.WriteString("structure", values.Structure);
        if (values.Key != null) writer.WriteString("key", values.Key);
        if (values.Chord != null) writer.WriteString("chord", values.Chord);
        if (values.Melody is { } melody)
        {
            writer.WriteStartArray("melody");
            foreach (var note in melody)
            {
                writer.WriteStartObject();
                writer.WriteNumber("pitch", note.Pitch);
                writer.WriteNumber("track", note.Track);
                writer.WriteNumber("duration_bin", note.DurationBin);
                writer.WriteNumber("duration_steps", note.DurationSteps);
                if (note.EndTime is double end) writer.WriteNumber("end_time", end);
                writer.WriteEndObject();
            }
            writer.WriteEndArray();
        }
    }

    /// <summary>Melody MIDI with the reference's fixed Vocal/Ins track names.</summary>
    static List<MidiExport.Track> _melodyTracks(List<TimedNote> notes) =>
        new List<MidiExport.Track>
        {
            new MidiExport.Track { Name = "Vocal", Notes = notes.Where(n => n.Track == 0).ToList() },
            new MidiExport.Track { Name = "Ins", Notes = notes.Where(n => n.Track == 1).ToList() }
        };

    static void _buildPlayback(TranscriptionResult result, List<Interval> chords, AbcScore? score, double duration, List<TimedNote> notes)
    {
        var _downbeats = score?.Beats.Where(b => b.BeatId == 1).Select(b => b.Time).ToList() ?? new List<double>();
        var _chordNotes = new List<TimedNote>();
        foreach (var chord in chords)
        {
            double _start = Math.Max(0.0, chord.Start);
            double _end = Math.Min(duration, chord.End);
            List<int> _pitches;
            try
            {
                _pitches = ChordPitches.Of(chord.Label);
            }
            catch (ChordSymbolException exception)
            {
                result.Warnings.Add($"Chord playback skipped {chord.Label}: {exception.Message}");
                continue;
            }
            //long chords get rearticulated on every downbeat they span
            var _cuts = new List<double> { _start };
            _cuts.AddRange(_downbeats.Where(t => _start < t && t < _end));
            _cuts.Add(_end);
            for (int i = 0; i + 1 < _cuts.Count; i++)
                if (_cuts[i + 1] > _cuts[i])
                    foreach (int pitch in _pitches) _chordNotes.Add(new TimedNote(_cuts[i], _cuts[i + 1], pitch, 2));
        }

        var _tracks = _melodyTracks(notes);
        _tracks.Add(new MidiExport.Track { Name = "Chords", Velocity = 48, Notes = _chordNotes });
        result.Midis["transcription"] = MidiExport.Write(_tracks);
        result.Midis["chords"] = MidiExport.Write(new[] { new MidiExport.Track { Name = "Chords", Velocity = 48, Notes = _chordNotes } });
    }

    /// <summary>
    /// Canonical notation needs a closing beat, so the last tempo is continued to the end of the
    /// audio (or of the last note) and the final bar is padded.
    /// </summary>
    static List<BeatEvent> _extendBeats(List<BeatEvent> beats, double duration, List<TimedNote> notes)
    {
        var _extended = new List<BeatEvent>(beats);
        var _recent = beats.Skip(Math.Max(0, beats.Count - 9)).Select(b => b.Time).ToList();
        var _spacing = Enumerable.Range(0, _recent.Count - 1).Select(i => _recent[i + 1] - _recent[i]).Order().ToArray();
        double _period = _spacing.Length % 2 == 1
            ? _spacing[_spacing.Length / 2]
            : (_spacing[_spacing.Length / 2 - 1] + _spacing[_spacing.Length / 2]) / 2;
        if (_period <= 0) throw new AbcRebuildException("Decoded beats must increase in time");

        double _end = Math.Max(duration, notes.Count > 0 ? notes.Max(n => n.End) : 0);
        while (_extended[^1].Time < _end - 1e-6)
        {
            var _previous = _extended[^1];
            _extended.Add(_previous with
            {
                Time = _previous.Time + _period,
                BeatId = _previous.BeatId % _previous.DeclaredNumerator + 1,
                LineNo = _extended.Count + 1
            });
        }
        return _extended;
    }

    static List<Interval> _clip(List<Interval> rows, double first, double last) =>
        rows.Where(r => r.End > first && r.Start < last)
            .Select(r => new Interval(Math.Max(first, r.Start), Math.Min(last, r.End), r.Label))
            .ToList();

    static string _label(EventValues values, EventField field) => field switch
    {
        EventField.Chord => values.Chord!,
        EventField.Key => values.Key!,
        _ => values.Structure!
    };

    static string _fieldName(EventField field) => field switch
    {
        EventField.Timestamp => "timestamp",
        EventField.Rhythm => "rhythm",
        EventField.Structure => "structure",
        EventField.Key => "key",
        EventField.Chord => "chord",
        _ => "melody"
    };

    static string _rhythmJson(RhythmValue? rhythm)
    {
        if (rhythm == null) return "{}";
        var _parts = new List<string>();
        if (rhythm.Meter is (int numerator, int denominator)) _parts.Add($"\"meter\": [{numerator}, {denominator}]");
        if (rhythm.EighthPosition is int eighth) _parts.Add($"\"eighth_position\": {eighth}");
        return "{" + string.Join(", ", _parts) + "}";
    }

    static string _fieldText(DecodedEvent item)
    {
        var _parts = new List<string>();
        foreach (var field in item.Fields)
        {
            var _values = item.Values;
            switch (field)
            {
                case EventField.Timestamp: _parts.Add($"timestamp={Py(_values.Timestamp!.Value)}"); break;
                case EventField.Rhythm:
                    var _rhythm = new List<string>();
                    if (_values.Rhythm!.Meter is (int numerator, int denominator)) _rhythm.Add($"meter:({numerator}, {denominator})");
                    if (_values.Rhythm.EighthPosition is int eighth) _rhythm.Add($"eighth_position:{eighth}");
                    _parts.Add("rhythm=" + string.Join(",", _rhythm));
                    break;
                case EventField.Structure: _parts.Add($"structure={_values.Structure}"); break;
                case EventField.Key: _parts.Add($"key={_values.Key}"); break;
                case EventField.Chord: _parts.Add($"chord={_values.Chord}"); break;
                default:
                    var _notes = _values.Melody!.Select(n => $"pitch={n.Pitch}:track={n.Track}:dur_bin={n.DurationBin}:dur_steps={n.DurationSteps}");
                    _parts.Add("melody=[" + string.Join(",", _notes) + "]");
                    break;
            }
        }
        return string.Join("; ", _parts);
    }

    static string _rows(IEnumerable<object[]> rows)
    {
        var _text = new StringBuilder();
        foreach (var row in rows)
        {
            _text.AppendJoin('\t', row.Select(value => value switch
            {
                double number => Py(number),
                float number => Py(number),
                _ => value.ToString() ?? ""
            }));
            _text.Append('\n');
        }
        return _text.ToString();
    }

    /// <summary>Python's str() for a float: shortest round trip, integers keep a trailing .0</summary>
    internal static string Py(double value)
    {
        string _text = value.ToString("R", CultureInfo.InvariantCulture);
        if (_text.Contains('E'))
        {
            var _parts = _text.Split('E');
            int _exponent = int.Parse(_parts[1], CultureInfo.InvariantCulture);
            return $"{_parts[0]}e{(_exponent < 0 ? "-" : "+")}{Math.Abs(_exponent):00}";
        }
        return _text.Contains('.') ? _text : _text + ".0";
    }
}
