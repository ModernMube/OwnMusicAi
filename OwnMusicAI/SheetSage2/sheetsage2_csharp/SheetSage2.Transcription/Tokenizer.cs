using System.Text;

namespace SheetSage2;

/// <summary>
/// Typed prompt and event vocabulary of tokenization_sheetsage2.py, schema v1. Every block is
/// laid out back to back, so a token's type is just a range check.
/// </summary>
public sealed class SheetSage2Tokenizer
{
    public const int PromptCapacity = 256;
    public const int MaxSubbeatShift = 256;
    public const int EighthPositions = 256;
    public const int PitchTokens = 256;
    public const int KeyTokens = 24;

    public static readonly string[] ChromaticSharps = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };

    public static readonly string[] StructureLabels =
    {
        "silence", "intro", "outro", "verse", "chorus", "bridge", "pre-chorus", "post-chorus", "interlude", "fade-out",
        "loop", "rap", "preshot", "irregular", "instrumental", "intro and verse", "pre-chorus and chorus",
        "verse and pre-chorus", "solo", "theme", "development", "variation", "pre-outro"
    };

    static readonly string[] FullChordQualities =
    {
        "maj", "min", "dim", "aug", "maj7", "min7", "7", "hdim7", "dim7", "minmaj7", "sus2", "sus4", "sus4(b7)", "maj6", "min6"
    };

    static readonly Dictionary<string, string[]> FullChordInversions = new Dictionary<string, string[]>
    {
        { "maj", new[] { "/2", "/3", "/5" } },
        { "min", new[] { "/2", "/b3", "/5" } },
        { "maj7", new[] { "/3", "/5", "/7" } },
        { "min7", new[] { "/b3", "/5", "/b7" } },
        { "7", new[] { "/3", "/5", "/b7" } }
    };

    /// <summary>Prompt tasks of schema v1: name, and the output field it fills (also its sampling group).</summary>
    public static readonly (string Name, EventField Field)[] Tasks =
    {
        ("timestamp", EventField.Timestamp),
        ("downbeat_meter", EventField.Rhythm),
        ("structure", EventField.Structure),
        ("key", EventField.Key),
        ("chord_majmin", EventField.Chord),
        ("chord_full", EventField.Chord),
        ("melody_vocal", EventField.Melody),
        ("melody_full", EventField.Melody)
    };

    public static readonly int[] DurationTemplates =
    {
        1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096
    };

    public const int PadToken = 0;
    public const int SosToken = 1;
    public const int EosToken = 2;
    public const int OutToken = 3;
    public const int PromptStart = 4;

    public int TimeHz { get; }
    public int TimeTokens { get; }
    public string[] FullChordLabels { get; }
    public string[] MajMinChordLabels { get; }
    public (int Numerator, int Denominator)[] MeterPairs { get; }

    public int PromptEnd => PromptStart + PromptCapacity;
    public int SubbeatShiftStart => PromptEnd;
    public int SubbeatShiftEnd => SubbeatShiftStart + MaxSubbeatShift + 1;
    public int TimeStart => SubbeatShiftEnd;
    public int TimeEnd => TimeStart + TimeTokens;
    public int MeterStart => TimeEnd;
    public int MeterEnd => MeterStart + MeterPairs.Length;
    public int EighthStart => MeterEnd;
    public int EighthEnd => EighthStart + EighthPositions;
    public int StructureStart => EighthEnd;
    public int StructureEnd => StructureStart + StructureLabels.Length;
    public int KeyStart => StructureEnd;
    public int KeyEnd => KeyStart + KeyTokens;
    public int MajMinStart => KeyEnd;
    public int MajMinEnd => MajMinStart + MajMinChordLabels.Length;
    public int FullChordStart => MajMinEnd;
    public int FullChordEnd => FullChordStart + FullChordLabels.Length;
    public int PitchStart => FullChordEnd;
    public int PitchEnd => PitchStart + PitchTokens;
    public int DurationStart => PitchEnd;
    public int DurationEnd => DurationStart + DurationTemplates.Length;
    public int TokenCount => DurationEnd;

    public SheetSage2Tokenizer(double audioLengthSeconds = 300.0, int timeHz = 100)
    {
        TimeHz = timeHz;
        TimeTokens = (int)Math.Round(audioLengthSeconds * timeHz);
        if (TimeTokens <= 0) throw new ArgumentException("audio length must produce at least one time token");

        MeterPairs = (from numerator in Enumerable.Range(1, 32)
                      from denominator in new[] { 1, 2, 4, 8, 16, 32 }
                      select (numerator, denominator)).ToArray();

        MajMinChordLabels = new[] { "N" }
            .Concat(ChromaticSharps.Select(r => $"{r}:maj"))
            .Concat(ChromaticSharps.Select(r => $"{r}:min"))
            .ToArray();

        var _full = new List<string> { "N" };
        foreach (var quality in FullChordQualities)
        {
            var _inversions = FullChordInversions.TryGetValue(quality, out var inv) ? inv : Array.Empty<string>();
            foreach (var root in ChromaticSharps)
                foreach (var inversion in _inversions.Append(""))
                    _full.Add($"{root}:{quality}{inversion}");
        }
        FullChordLabels = _full.ToArray();
    }

    public TokenType TypeOf(int token)
    {
        if (token >= SubbeatShiftStart && token < SubbeatShiftEnd) return TokenType.SubbeatShift;
        if (token >= TimeStart && token < TimeEnd) return TokenType.Time;
        if (token >= MeterStart && token < MeterEnd) return TokenType.Meter;
        if (token >= EighthStart && token < EighthEnd) return TokenType.EighthPosition;
        if (token >= StructureStart && token < StructureEnd) return TokenType.Structure;
        if (token >= KeyStart && token < KeyEnd) return TokenType.Key;
        if (token >= MajMinStart && token < MajMinEnd) return TokenType.ChordMajMin;
        if (token >= FullChordStart && token < FullChordEnd) return TokenType.ChordFull;
        if (token >= PitchStart && token < PitchEnd) return TokenType.Pitch;
        if (token >= DurationStart && token < DurationEnd) return TokenType.Duration;
        if (token >= PromptStart && token < PromptStart + Tasks.Length) return TokenType.Prompt;
        if (token == PadToken) return TokenType.Pad;
        if (token == SosToken) return TokenType.Sos;
        if (token == EosToken) return TokenType.Eos;
        if (token == OutToken) return TokenType.Out;
        throw new ArgumentException($"token {token} is outside vocabulary size {TokenCount}");
    }

    public static EventField? OutputField(TokenType type) => type switch
    {
        TokenType.Time => EventField.Timestamp,
        TokenType.Meter or TokenType.EighthPosition => EventField.Rhythm,
        TokenType.Structure => EventField.Structure,
        TokenType.Key => EventField.Key,
        TokenType.ChordMajMin or TokenType.ChordFull => EventField.Chord,
        TokenType.Pitch or TokenType.Duration => EventField.Melody,
        _ => null
    };

    /// <summary>Canonical schema order, rejecting unknown names and two prompts of one field.</summary>
    public IReadOnlyList<string> NormalizePrompts(IEnumerable<string> prompts)
    {
        var _names = new List<string>();
        foreach (var prompt in prompts)
        {
            string _name = prompt.Trim();
            if (_name.StartsWith("<|") && _name.EndsWith("|>")) _name = _name[2..^2];
            int _index = Array.FindIndex(Tasks, t => t.Name == _name);
            if (_index < 0) throw new ArgumentException($"Unknown prompt: '{prompt}'");
            if (!_names.Contains(_name)) _names.Add(_name);
        }
        _names.Sort((a, b) => Array.FindIndex(Tasks, t => t.Name == a).CompareTo(Array.FindIndex(Tasks, t => t.Name == b)));
        var _groups = new HashSet<EventField>();
        foreach (var name in _names)
        {
            var _field = Tasks.First(t => t.Name == name).Field;
            if (!_groups.Add(_field)) throw new ArgumentException($"Prompts of field '{_field}' are mutually exclusive");
        }
        if (_names.Count == 0) throw new ArgumentException("At least one task prompt is required");
        return _names;
    }

    public List<int> PromptPrefix(IEnumerable<string> prompts)
    {
        var _prefix = new List<int> { SosToken };
        foreach (var name in NormalizePrompts(prompts)) _prefix.Add(PromptStart + Array.FindIndex(Tasks, t => t.Name == name));
        _prefix.Add(OutToken);
        return _prefix;
    }

    public string PromptOf(int token)
    {
        int _index = token - PromptStart;
        if (_index < 0 || _index >= Tasks.Length) throw new ArgumentException($"token {token} is not a prompt token");
        return Tasks[_index].Name;
    }

    public IEnumerable<int> SubbeatShiftTokens(int shift)
    {
        if (shift < 0) throw new ArgumentException("subbeat shift must be non-negative");
        while (shift > MaxSubbeatShift)
        {
            yield return SubbeatShiftStart + MaxSubbeatShift;
            shift -= MaxSubbeatShift;
        }
        yield return SubbeatShiftStart + shift;
    }

    public int TimeIdToToken(int timeId)
    {
        if (timeId < 0 || timeId >= TimeTokens) throw new ArgumentException($"time id {timeId} is outside [0, {TimeTokens})");
        return TimeStart + timeId;
    }

    public int TimeIdOf(int token) => token - TimeStart;

    public int DurationBinOf(int token) => token - DurationStart;

    /// <summary>Human readable token, same spelling as tokenizer.describe.</summary>
    public string Describe(int token)
    {
        var _type = TypeOf(token);
        switch (_type)
        {
            case TokenType.Prompt: return $"<|{PromptOf(token)}|>";
            case TokenType.SubbeatShift: return $"<subbeat_shift_{token - SubbeatShiftStart}>";
            case TokenType.Time: return FormattableString.Invariant($"<time_{(token - TimeStart) / (double)TimeHz:0.00}s>");
            case TokenType.Meter:
                var _meter = MeterPairs[token - MeterStart];
                return $"<meter_{_meter.Numerator}/{_meter.Denominator}>";
            case TokenType.EighthPosition: return $"<eighth_pos_{token - EighthStart}>";
            case TokenType.Structure: return $"<structure_{StructureLabels[token - StructureStart]}>";
            case TokenType.Key: return $"<key_{KeyLabel(token)}>";
            case TokenType.ChordMajMin: return $"<chord_majmin_{MajMinChordLabels[token - MajMinStart]}>";
            case TokenType.ChordFull: return $"<chord_full_{FullChordLabels[token - FullChordStart]}>";
            case TokenType.Pitch:
                int _pitch = token - PitchStart;
                return $"<pitch_{_pitch % 128}_track_{(_pitch >= 128 ? 1 : 0)}>";
            case TokenType.Duration: return $"<duration_{token - DurationStart}>";
            case TokenType.Pad: return "<|pad|>";
            case TokenType.Sos: return "<|sos|>";
            case TokenType.Eos: return "<|eos|>";
            default: return "<|out|>";
        }
    }

    public string KeyLabel(int token)
    {
        int _id = token - KeyStart;
        return $"{ChromaticSharps[_id % 12]}:{(_id >= 12 ? "minor" : "major")}";
    }

    /// <summary>Fills one field's value from its tokens (the _decode_field of the reference).</summary>
    public void DecodeField(EventValues values, EventField field, List<int> tokens)
    {
        switch (field)
        {
            case EventField.Timestamp:
                values.Timestamp = TimeIdOf(tokens[0]) / (double)TimeHz;
                break;
            case EventField.Rhythm:
                var _rhythm = new RhythmValue();
                foreach (var token in tokens)
                {
                    if (TypeOf(token) == TokenType.Meter) _rhythm.Meter = MeterPairs[token - MeterStart];
                    else if (TypeOf(token) == TokenType.EighthPosition) _rhythm.EighthPosition = token - EighthStart;
                }
                values.Rhythm = _rhythm;
                break;
            case EventField.Structure:
                values.Structure = StructureLabels[tokens[0] - StructureStart];
                break;
            case EventField.Key:
                values.Key = KeyLabel(tokens[0]);
                break;
            case EventField.Chord:
                values.Chord = TypeOf(tokens[0]) == TokenType.ChordMajMin
                    ? MajMinChordLabels[tokens[0] - MajMinStart]
                    : FullChordLabels[tokens[0] - FullChordStart];
                break;
            default:
                var _notes = new List<MelodyNote>();
                for (int i = 0; i < tokens.Count;)
                {
                    int _pitchId = tokens[i] - PitchStart;
                    int _bin = 0;
                    if (i + 1 < tokens.Count && TypeOf(tokens[i + 1]) == TokenType.Duration)
                    {
                        _bin = DurationBinOf(tokens[i + 1]);
                        i += 2;
                    }
                    else i += 1;
                    _notes.Add(new MelodyNote
                    {
                        Pitch = _pitchId % 128,
                        Track = _pitchId >= 128 ? 1 : 0,
                        DurationBin = _bin,
                        DurationSteps = DurationTemplates[_bin]
                    });
                }
                values.Melody = _notes;
                break;
        }
    }

    /// <summary>Re-decodes every field of an event from its tokens.</summary>
    public void Refresh(DecodedEvent target)
    {
        target.Values = new EventValues();
        foreach (var field in target.Fields) DecodeField(target.Values, field, target[field]!);
    }

    /// <summary>
    /// Parses one prompt-conditioned sequence into timed, typed events. strict mirrors the
    /// reference's validation, including the messages the pipeline treats as recoverable.
    /// </summary>
    public DecodedSequence DecodeSequence(IReadOnlyList<int> tokens, bool strict = true)
    {
        var _tokens = new List<int>(tokens);
        while (_tokens.Count > 0 && _tokens[^1] == PadToken) _tokens.RemoveAt(_tokens.Count - 1);
        if (_tokens.Count == 0 || _tokens[0] != SosToken) throw new FormatException("sequence must begin with <|sos|>");

        int _out = _tokens.IndexOf(OutToken, 1);
        if (_out < 0) throw new FormatException("sequence is missing <|out|>");
        var _prompts = _tokens.GetRange(1, _out - 1).Select(PromptOf).ToList();
        if (strict && !NormalizePrompts(_prompts).SequenceEqual(_prompts))
            throw new FormatException("prompt tokens are not in canonical schema order");
        var _active = _prompts.Select(p => Tasks.First(t => t.Name == p).Field).ToHashSet();

        var _events = new List<DecodedEvent>();
        int _position = _out + 1;
        int _step = 0;
        bool _sawEos = false;
        while (_position < _tokens.Count)
        {
            if (_tokens[_position] == EosToken)
            {
                _sawEos = true;
                _position++;
                break;
            }
            if (TypeOf(_tokens[_position]) != TokenType.SubbeatShift)
                throw new FormatException($"event at token index {_position} has no subbeat shift");
            while (_position < _tokens.Count && TypeOf(_tokens[_position]) == TokenType.SubbeatShift)
            {
                _step += _tokens[_position] - SubbeatShiftStart;
                _position++;
            }

            var _event = new DecodedEvent { Subbeat = _step };
            while (_position < _tokens.Count)
            {
                int _token = _tokens[_position];
                var _type = TypeOf(_token);
                if (_type == TokenType.SubbeatShift || _token == EosToken) break;
                var _field = OutputField(_type) ?? throw new FormatException($"token {_token} ({_type}) has no field in schema v1");
                if (strict && !_active.Contains(_field))
                    throw new FormatException($"token {_token} belongs to inactive output field '{_field}'");
                (_event[_field] ?? _newList(_event, _field)).Add(_token);
                _position++;
            }

            if (!_event.Fields.Any())
            {
                if (strict) throw new FormatException($"empty event at subbeat {_step}");
                continue;
            }
            if (strict) _validateEvent(_event);
            foreach (var field in _event.Fields) DecodeField(_event.Values, field, _event[field]!);
            _events.Add(_event);
        }

        if (strict && !_sawEos) throw new FormatException("sequence is missing <|eos|>");
        if (strict && _position != _tokens.Count) throw new FormatException("non-padding tokens follow <|eos|>");
        return new DecodedSequence { Prompts = _prompts, Events = _events, HasEos = _sawEos };
    }

    /// <summary>Turns decoded events back into a token sequence (used for the overlap prefix).</summary>
    public List<int> EncodeSequence(DecodedSequence decoded)
    {
        var _output = PromptPrefix(decoded.Prompts);
        int _previous = 0;
        foreach (var item in decoded.Events)
        {
            if (item.Subbeat < _previous) throw new InvalidOperationException("events must be sorted by non-decreasing subbeat");
            _output.AddRange(SubbeatShiftTokens(item.Subbeat - _previous));
            _previous = item.Subbeat;
            foreach (var field in Enum.GetValues<EventField>())
                if (item[field] is { } tokens) _output.AddRange(tokens);
        }
        if (decoded.HasEos) _output.Add(EosToken);
        return _output;
    }

    List<int> _newList(DecodedEvent target, EventField field)
    {
        var _list = new List<int>();
        target.Set(field, _list);
        return _list;
    }

    void _validateEvent(DecodedEvent item)
    {
        foreach (var field in item.Fields)
        {
            var _types = item[field]!.Select(TypeOf).ToList();
            if (field == EventField.Timestamp && (_types.Count != 1 || _types[0] != TokenType.Time))
                throw new FormatException("timestamp event must contain exactly one time token");
            if (field == EventField.Rhythm)
            {
                bool _ok = _types.SequenceEqual(new[] { TokenType.EighthPosition })
                    || _types.SequenceEqual(new[] { TokenType.Meter, TokenType.EighthPosition });
                if (!_ok) throw new FormatException($"invalid rhythm payload: {string.Join(", ", _types)}");
            }
            if ((field == EventField.Structure || field == EventField.Key || field == EventField.Chord) && _types.Count != 1)
                throw new FormatException($"field '{field}' must contain exactly one token");
            if (field == EventField.Melody)
            {
                for (int i = 0; i < _types.Count;)
                {
                    if (_types[i] != TokenType.Pitch)
                        throw new FormatException("melody payload must contain pitch tokens with optional duration");
                    i += i + 1 < _types.Count && _types[i + 1] == TokenType.Duration ? 2 : 1;
                }
            }
        }
    }
}
