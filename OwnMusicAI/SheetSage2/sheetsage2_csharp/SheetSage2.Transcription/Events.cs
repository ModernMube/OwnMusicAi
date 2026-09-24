namespace SheetSage2;

/// <summary>Output fields of schema v1, in event_field_order.</summary>
public enum EventField { Timestamp = 0, Rhythm = 1, Structure = 2, Key = 3, Chord = 4, Melody = 5 }

public enum TokenType { Pad, Sos, Eos, Out, Prompt, SubbeatShift, Time, Meter, EighthPosition, Structure, Key, ChordMajMin, ChordFull, Pitch, Duration }

/// <summary>One melody note of an event; EndTime is filled in while stitching windows.</summary>
public sealed class MelodyNote
{
    public int Pitch;
    public int Track;
    public int DurationBin;
    public int DurationSteps;
    public double? EndTime;

    public MelodyNote Clone() => (MelodyNote)MemberwiseClone();
}

public sealed class RhythmValue
{
    public (int Numerator, int Denominator)? Meter;
    public int? EighthPosition;
}

/// <summary>Decoded payload of one event, by field.</summary>
public sealed class EventValues
{
    public double? Timestamp;
    public RhythmValue? Rhythm;
    public string? Structure;
    public string? Key;
    public string? Chord;
    public List<MelodyNote>? Melody;

    public bool Has(EventField field) => field switch
    {
        EventField.Timestamp => Timestamp != null,
        EventField.Rhythm => Rhythm != null,
        EventField.Structure => Structure != null,
        EventField.Key => Key != null,
        EventField.Chord => Chord != null,
        _ => Melody != null
    };

    public EventValues Clone() => new EventValues
    {
        Timestamp = Timestamp,
        Rhythm = Rhythm == null ? null : new RhythmValue { Meter = Rhythm.Meter, EighthPosition = Rhythm.EighthPosition },
        Structure = Structure,
        Key = Key,
        Chord = Chord,
        Melody = Melody?.Select(n => n.Clone()).ToList()
    };
}

/// <summary>
/// One decoded event. Tokens are kept per field so a window can be re-encoded as an overlap
/// prefix; Time and the window fields are set once the event is stitched into the song.
/// </summary>
public sealed class DecodedEvent
{
    public int Subbeat;
    public readonly List<int>?[] Tokens = new List<int>?[6];
    public EventValues Values = new EventValues();
    public double? Time;
    public int WindowIndex;
    public double WindowStart;
    public int SourceSubbeat;
    public int GlobalSubbeat;

    public List<int>? this[EventField field] => Tokens[(int)field];

    public void Set(EventField field, List<int>? tokens) => Tokens[(int)field] = tokens;

    public IEnumerable<EventField> Fields =>
        Enum.GetValues<EventField>().Where(f => Tokens[(int)f] is { Count: > 0 });

    public DecodedEvent Clone()
    {
        var _copy = new DecodedEvent { Subbeat = Subbeat, Values = Values.Clone() };
        foreach (var field in Fields) _copy.Set(field, new List<int>(Tokens[(int)field]!));
        return _copy;
    }
}

/// <summary>A parsed token sequence: which prompts it used and the events it carries.</summary>
public sealed class DecodedSequence
{
    public string SchemaVersion = "v1";
    public IReadOnlyList<string> Prompts = Array.Empty<string>();
    public List<DecodedEvent> Events = new List<DecodedEvent>();
    public bool HasEos = true;
}
