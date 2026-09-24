namespace SheetSage2;

/// <summary>One sliding window of the song and the slice of it we keep.</summary>
public sealed class WindowPlan
{
    public double Start;
    public double End;
    public double AcceptStart;
    public double AcceptEnd;
    public double PrefixEnd;
    public double? GenerationStop;
    public int Index;
}

/// <summary>
/// Window planning, accepting events into song time, and rebuilding the overlap prefix that
/// carries context from the previous window (pipeline_sheetsage2.py / generation_sheetsage2.py).
/// </summary>
internal static class WindowStitcher
{
    const double Eps = 1e-4;

    public static List<WindowPlan> Plan(double duration, double windowSeconds, double overlapSeconds, double lookaheadSeconds)
    {
        if (!(duration > 0) || !(windowSeconds > 0)) throw new ArgumentException("Duration and window length must be positive");
        if (!(0 <= lookaheadSeconds && lookaheadSeconds <= overlapSeconds && overlapSeconds < windowSeconds))
            throw new ArgumentException("Require 0 <= lookahead <= overlap < window length");

        double _hop = windowSeconds - overlapSeconds;
        double _start = 0.0, _accepted = 0.0;
        var _plan = new List<WindowPlan>();
        while (true)
        {
            bool _last = _start + windowSeconds >= duration - 1e-6;
            double _acceptEnd = _last ? duration : _start + windowSeconds - lookaheadSeconds;
            _plan.Add(new WindowPlan
            {
                Index = _plan.Count,
                Start = _start,
                End = Math.Min(duration, _start + windowSeconds),
                AcceptStart = _accepted,
                AcceptEnd = _acceptEnd,
                PrefixEnd = _accepted,
                GenerationStop = _last ? null : windowSeconds - lookaheadSeconds
            });
            if (_last) return _plan;
            _accepted = _acceptEnd;
            _start = Math.Min(_start + _hop, duration - windowSeconds);
        }
    }

    /// <summary>Window samples starting at the given second, zero-padded to a full window.</summary>
    public static float[] Slice(float[] audio, double start, double seconds, int sampleRate)
    {
        int _offset = (int)Math.Round(start * sampleRate);
        int _count = (int)Math.Round(seconds * sampleRate);
        var _slice = new float[Math.Min(_count, Math.Max(0, audio.Length - _offset))];
        Array.Copy(audio, _offset, _slice, 0, _slice.Length);
        return _slice;
    }

    /// <summary>
    /// Moves a window's events into song time and keeps the ones inside its accept range.
    /// </summary>
    public static List<DecodedEvent> Accept(DecodedSequence decoded, Generation.TimeMap lookup, WindowPlan window,
        double songDuration, int globalSubbeatBase)
    {
        var _accepted = new List<DecodedEvent>();
        foreach (var item in decoded.Events)
        {
            double _local = lookup.Lookup(item.Subbeat);
            double _absolute = window.Start + _local;
            if (_absolute < window.AcceptStart - Eps) continue;
            if (_absolute >= window.AcceptEnd - Eps || _absolute >= songDuration - Eps) continue;

            var _output = item.Clone();
            _output.Time = Math.Clamp(_absolute, 0.0, songDuration);
            _output.WindowIndex = window.Index;
            _output.WindowStart = window.Start;
            _output.SourceSubbeat = item.Subbeat;
            _output.GlobalSubbeat = globalSubbeatBase + item.Subbeat;
            if (_output.Values.Timestamp != null) _output.Values.Timestamp = _output.Time;

            foreach (var note in _output.Values.Melody ?? Enumerable.Empty<MelodyNote>())
            {
                double _end = window.Start + lookup.Lookup(item.Subbeat + note.DurationSteps);
                note.EndTime = Math.Min(songDuration, Math.Max(_output.Time.Value + 0.04, _end));
            }
            _accepted.Add(_output);
        }
        return _accepted;
    }

    /// <summary>
    /// Re-encodes the events the previous windows already accepted in this window's prefix range,
    /// so the model continues its own transcription instead of restarting. Null when the window
    /// has no beat or timestamp to anchor on.
    /// </summary>
    public static (List<int>? Tokens, int Base) OverlapPrefix(List<DecodedEvent> stitched, SheetSage2Tokenizer tokenizer,
        IReadOnlyList<string> prompts, double windowStart, double prefixEnd)
    {
        var _source = stitched
            .Where(e => windowStart - Eps <= (e.Time ?? -1.0) && (e.Time ?? -1.0) < prefixEnd - Eps)
            .OrderBy(e => e.GlobalSubbeat).ThenBy(e => e.Time ?? 0.0)
            .ToList();
        int _first = _source.FindIndex(e => e.Values.Timestamp != null || e.Values.Rhythm != null);
        if (_first < 0) return (null, 0);

        _source = _source.GetRange(_first, _source.Count - _first);
        int _base = _source[0].GlobalSubbeat;
        var _context = _activeContext(stitched, tokenizer, _source[0].Time!.Value);

        var _events = new List<DecodedEvent>();
        foreach (var source in _source)
        {
            var _event = source.Clone();
            _event.Subbeat = Math.Max(0, source.GlobalSubbeat - _base);
            if (_event[EventField.Timestamp] != null)
            {
                int _id = (int)Math.Round((source.Time!.Value - windowStart) * tokenizer.TimeHz);
                _id = Math.Clamp(_id, 0, tokenizer.TimeTokens - 1);
                _event.Set(EventField.Timestamp, new List<int> { tokenizer.TimeIdToToken(_id) });
            }
            tokenizer.Refresh(_event);
            _events.Add(_event);
        }
        _applyContext(_events[0], _context, tokenizer);

        var _decoded = new DecodedSequence { Prompts = prompts, Events = _events, HasEos = false };
        return (tokenizer.EncodeSequence(_decoded), _base);
    }

    /// <summary>Structure/key/chord (and the meter) in force at the given time.</summary>
    static Dictionary<EventField, List<int>> _activeContext(List<DecodedEvent> events, SheetSage2Tokenizer tokenizer, double time)
    {
        var _state = new Dictionary<EventField, List<int>>();
        List<int>? _meter = null;
        foreach (var item in events)
        {
            if (item.Time is not double when || when > time + 1e-6) continue;
            foreach (var field in new[] { EventField.Structure, EventField.Key, EventField.Chord })
                if (item[field] is { Count: > 0 } tokens) _state[field] = new List<int>(tokens);
            foreach (var token in item[EventField.Rhythm] ?? Enumerable.Empty<int>())
            {
                if (tokenizer.TypeOf(token) != TokenType.Meter) continue;
                _meter = new List<int> { token };
                break;
            }
        }
        if (_meter != null) _state[EventField.Rhythm] = _meter;
        return _state;
    }

    static void _applyContext(DecodedEvent target, Dictionary<EventField, List<int>> context, SheetSage2Tokenizer tokenizer)
    {
        foreach (var field in new[] { EventField.Structure, EventField.Key, EventField.Chord })
            if (target[field] == null && context.TryGetValue(field, out var tokens)) target.Set(field, new List<int>(tokens));

        var _rhythm = target[EventField.Rhythm];
        if (_rhythm != null && context.TryGetValue(EventField.Rhythm, out var meter))
        {
            bool _hasMeter = _rhythm.Any(t => tokenizer.TypeOf(t) == TokenType.Meter);
            bool _hasEighth = _rhythm.Any(t => tokenizer.TypeOf(t) == TokenType.EighthPosition);
            if (_hasEighth && !_hasMeter) target.Set(EventField.Rhythm, meter.Concat(_rhythm).ToList());
        }
        tokenizer.Refresh(target);
    }
}
