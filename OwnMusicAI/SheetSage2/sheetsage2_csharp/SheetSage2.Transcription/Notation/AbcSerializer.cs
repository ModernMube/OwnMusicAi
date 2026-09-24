using System.Text;
using System.Text.RegularExpressions;

namespace SheetSage2.Notation;

/// <summary>A run of at most four measures sharing meter, key and structure label.</summary>
sealed class MeasureGroup
{
    public List<Measure> Measures = new List<Measure>();
    public List<string> StructureLabels = new List<string>();
    public bool MeterChanged;
    public bool KeyChanged;
}

/// <summary>
/// Turns an <see cref="AbcScore"/> into two-voice ABC text, then checks the result against the
/// score before anyone gets to see it (score_to_abc + validate_serialized_abc).
/// </summary>
public static class AbcSerializer
{
    static readonly int[] SupportedDurations = { 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48 };

    static readonly Regex MusicElement = new Regex(
        "\"(?<quoted>[^\"]*)\"" + @"|\[K:(?<key>[^\]]+)\]" + @"|(?<note>[_=^]*[A-Ga-gz][,']*)(?<duration>[0-9]*)(?<tie>-?)",
        RegexOptions.Compiled);

    public static string Serialize(AbcScore score)
    {
        int _unit = UnitDenominator(score);
        var _first = score.Measures[0];
        var _lines = new List<string>
        {
            "X:1",
            "T:",
            $"M:{_first.AbcNumerator}/{_first.AbcDenominator}",
            $"L:1/{_unit}",
            $"Q:1/4={(int)Math.Round(EstimateTempo(score), MidpointRounding.ToEven)}",
            "V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"",
            "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"",
            $"K:{score.KeyArr[_first.StartT]}"
        };

        foreach (var group in _groups(score))
        {
            _lines.AddRange(group.StructureLabels.Select(label => $"% {label}"));
            var _lead = group.Measures[0];
            foreach (var voice in AbcScore.VoiceIds)
            {
                _lines.Add($"V: {voice}");
                if (group.MeterChanged) _lines.Add($"M:{_lead.AbcNumerator}/{_lead.AbcDenominator}");
                if (group.KeyChanged) _lines.Add($"K:{score.KeyArr[_lead.StartT]}");
                _lines.Add(_renderGroup(score, voice, group.Measures, _unit));
            }
        }

        string _text = string.Join("\n", _lines) + "\n";
        Validate(_text, score);
        return _text;
    }

    /// <summary>Smallest note value the score needs; 1/1024 is where we give up.</summary>
    public static int UnitDenominator(AbcScore score)
    {
        int _unit = 1;
        foreach (var measure in score.Measures)
            foreach (int denominator in new[] { measure.Denominator, measure.AbcDenominator })
                _unit = _lcm(_unit, denominator * AbcScore.SubbeatDivision);
        if (_unit > 1024) throw new AbcRebuildException($"Required ABC unit length 1/{_unit} is unreasonably small");
        return _unit;
    }

    public static double EstimateTempo(AbcScore score)
    {
        double _seconds = score.SubbeatTimes[^1] - score.SubbeatTimes[0];
        double _quarters = score.SubbeatQuarters[^1] - score.SubbeatQuarters[0];
        if (_seconds <= 0 || _quarters <= 0) throw new AbcRebuildException("Cannot estimate tempo from a zero-duration score");
        return _quarters / _seconds * 60.0;
    }

    static int MeasureUnits(Measure measure, int unit) => measure.Numerator * unit / measure.Denominator;

    static int MeasureAbcUnits(Measure measure, int unit) => measure.AbcNumerator * unit / measure.AbcDenominator;

    static int PaddingUnits(Measure measure, int unit) => MeasureAbcUnits(measure, unit) - MeasureUnits(measure, unit);

    static int DurationUnits(AbcScore score, int startT, int endT, int unit)
    {
        int _units = 0;
        for (int t = startT; t < endT; t++)
        {
            int _divisor = score.SubbeatDenominators[t] * AbcScore.SubbeatDivision;
            if (unit % _divisor != 0) throw new AbcRebuildException($"ABC L:1/{unit} cannot express a 1/{_divisor} subbeat exactly");
            _units += unit / _divisor;
        }
        return _units;
    }

    static bool ContinuesPitch(int value, int next) => value > 0 && next == (value / 2 - 1) * 2 + 2;

    static bool SameSegment(int value, int next) => value == 0 ? next == 0 : next == (value / 2 - 1) * 2 + 2;

    /// <summary>Splits a length into values strict ABC parsers accept.</summary>
    static List<int> SplitDuration(int duration)
    {
        if (duration <= 0) throw new AbcRebuildException($"Cannot serialize non-positive duration {duration}");
        var _parts = new List<int>();
        int _remaining = duration;
        while (_remaining > 0)
        {
            if (SupportedDurations.Contains(_remaining))
            {
                _parts.Add(_remaining);
                break;
            }
            var _candidates = SupportedDurations.Where(v => v < _remaining).ToList();
            if (_candidates.Count == 0) throw new AbcRebuildException($"Duration {duration} cannot be split into representable ABC values");
            _parts.Add(_candidates.Max());
            _remaining -= _candidates.Max();
        }
        return _parts;
    }

    static IEnumerable<string> RenderDuration(string prefix, string note, int duration, bool tieOut)
    {
        var _chunks = SplitDuration(duration);
        for (int i = 0; i < _chunks.Count; i++)
        {
            bool _continues = note != "z" && (i + 1 < _chunks.Count || tieOut);
            yield return (i == 0 ? prefix : "") + note + (_chunks[i] == 1 ? "" : _chunks[i].ToString()) + (_continues ? "-" : "");
        }
    }

    static string _renderMeasure(AbcScore score, string voiceId, Measure measure, int unit)
    {
        var _voice = score.VoiceArrs[voiceId];
        bool _showChords = voiceId == "Vocal";
        var _accidentals = new Dictionary<int, int>();
        string _key = score.KeyArr[measure.StartT];
        var _keyAccidentals = MusicSymbols.KeyAccidentals(_key);
        var _parts = new List<string>();

        int _padding = PaddingUnits(measure, unit);
        if (_padding < 0) throw new AbcRebuildException($"Measure {measure.Index}: notated meter is shorter than its decoded span");
        int _leading = measure.PadBefore ? _padding : 0;
        int _trailing = measure.PadBefore ? 0 : _padding;

        int t = measure.StartT;
        while (t < measure.EndT)
        {
            int _next = measure.EndT;
            for (int probe = t + 1; probe < measure.EndT; probe++)
                if (!SameSegment(_voice[t], _voice[probe])) { _next = Math.Min(_next, probe); break; }
            for (int probe = t + 1; probe < measure.EndT; probe++)
                if (score.KeyArr[probe] != score.KeyArr[probe - 1]) { _next = Math.Min(_next, probe); break; }
            if (_showChords)
                for (int probe = t + 1; probe < measure.EndT; probe++)
                    if (score.ChordArr[probe] != score.ChordArr[probe - 1]) { _next = Math.Min(_next, probe); break; }

            string _prefix = "";
            if (t > measure.StartT && score.KeyArr[t] != _key)
            {
                _key = score.KeyArr[t];
                _keyAccidentals = MusicSymbols.KeyAccidentals(_key);
                _accidentals.Clear();
                _prefix += $"[K:{_key}]";
            }
            if (_showChords && (t == measure.StartT || score.ChordArr[t] != score.ChordArr[t - 1]))
            {
                string? _chord = MusicSymbols.ChordSymbolToAbc(score.ChordArr[t]);
                if (_chord != null) _prefix += $"\"{_chord}\"";
            }

            int _value = _voice[t];
            string _note = _value == 0 ? "z" : MusicSymbols.NoteToAbc(_value / 2 - 1, _keyAccidentals, _accidentals);
            int _duration = DurationUnits(score, t, _next, unit);
            if (t == measure.StartT && _leading > 0)
            {
                if (_value == 0 && _prefix.Length == 0) _duration += _leading;
                else _parts.AddRange(RenderDuration("", "z", _leading, false));
                _leading = 0;
            }
            if (_value == 0 && _next == measure.EndT && _trailing > 0)
            {
                _duration += _trailing;
                _trailing = 0;
            }
            if (_duration <= 0) throw new AbcRebuildException($"Non-positive ABC duration at subbeats {t}:{_next}");

            bool _tieOut = _value > 0 && _next < _voice.Length && ContinuesPitch(_value, _voice[_next]);
            _parts.AddRange(RenderDuration(_prefix, _note, _duration, _tieOut));
            t = _next;
        }

        if (_leading > 0) throw new AbcRebuildException($"Measure {measure.Index}: leading rest padding was not serialized");
        if (_trailing > 0) _parts.AddRange(RenderDuration("", "z", _trailing, false));
        return string.Concat(_parts);
    }

    /// <summary>Whether a rendered bar is only rests, so ABC Z can stand in for it.</summary>
    static bool _isFullRest(string measure)
    {
        int _cursor = 0;
        bool _sawNote = false;
        foreach (Match match in MusicElement.Matches(measure))
        {
            if (measure[_cursor..match.Index].Length > 0) return false;
            _cursor = match.Index + match.Length;
            if (match.Groups["quoted"].Success || match.Groups["key"].Success) return false;
            _sawNote = true;
            if (match.Groups["note"].Value != "z" || match.Groups["tie"].Value.Length > 0) return false;
        }
        return _sawNote && _cursor == measure.Length;
    }

    static string _renderGroup(AbcScore score, string voiceId, List<Measure> measures, int unit)
    {
        var _rendered = measures.Select(m => _renderMeasure(score, voiceId, m, unit)).ToList();
        var _parts = new StringBuilder();
        int _index = 0;
        while (_index < _rendered.Count)
        {
            if (!_isFullRest(_rendered[_index]))
            {
                _parts.Append(_rendered[_index]).Append('|');
                _index++;
                continue;
            }
            int _end = _index + 1;
            while (_end < _rendered.Count && _isFullRest(_rendered[_end])) _end++;
            int _count = _end - _index;
            _parts.Append('Z').Append(_count > 1 ? _count.ToString() : "").Append('|');
            _index = _end;
        }
        return _parts.ToString();
    }

    static List<MeasureGroup> _groups(AbcScore score)
    {
        var _first = score.Measures[0];
        var _meter = (_first.AbcNumerator, _first.AbcDenominator);
        string _key = score.KeyArr[_first.StartT];
        string _structure = "";
        var _groups = new List<MeasureGroup>();

        foreach (var measure in score.Measures)
        {
            var _current = (measure.AbcNumerator, measure.AbcDenominator);
            string _measureKey = score.KeyArr[measure.StartT];
            bool _meterChanged = _current != _meter;
            bool _keyChanged = _measureKey != _key;
            var _labels = new List<string>();
            foreach (var (t, label) in score.StructureEvents)
            {
                if (t < measure.StartT || t >= measure.EndT) continue;
                string _clean = string.Join(" ", label.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
                if (_clean.Length > 0 && _clean != _structure)
                {
                    _labels.Add(_clean);
                    _structure = _clean;
                }
            }

            if (_groups.Count == 0 || _groups[^1].Measures.Count >= 4 || _meterChanged || _keyChanged || _labels.Count > 0)
                _groups.Add(new MeasureGroup { Measures = { measure }, StructureLabels = _labels, MeterChanged = _meterChanged, KeyChanged = _keyChanged });
            else
                _groups[^1].Measures.Add(measure);

            _meter = _current;
            _key = score.KeyArr[measure.EndT - 1];
        }
        return _groups;
    }

    /// <summary>Re-reads the serialized text and insists it says exactly what the score says.</summary>
    public static void Validate(string text, AbcScore score)
    {
        var _lines = text.Split('\n');
        if (_lines.Length > 0 && _lines[^1].Length == 0) _lines = _lines[..^1];
        if (_lines.Length == 0 || _lines[0] != "X:1") throw new AbcRebuildException("ABC must start with X:1");
        if (_lines.Length < 2 || _lines[1] != "T:") throw new AbcRebuildException("ABC title must be fixed as empty T:");
        if (_lines.Any(l => l.StartsWith("%abc-") || l.StartsWith("I:abc-creator")))
            throw new AbcRebuildException("ABC must not contain version or creator metadata");

        var _voices = _lines.Select(l => Regex.Match(l, "^V: (Vocal|Ins) ")).Where(m => m.Success).Select(m => m.Groups[1].Value).ToList();
        if (!_voices.SequenceEqual(AbcScore.VoiceIds)) throw new AbcRebuildException("Expected fixed Vocal/Ins voice definitions");

        int _insIndex = Array.FindIndex(_lines, l => l.StartsWith("V: Ins "));
        int _keyIndex = -1;
        for (int i = 1; i < _lines.Length; i++)
            if (_lines[i].StartsWith("K:") && _insIndex >= 0 && _insIndex < i) { _keyIndex = i; break; }
        if (_keyIndex < 0) throw new AbcRebuildException("ABC header K: field is missing");

        var _first = score.Measures[0];
        var _headerMeters = _lines[..(_keyIndex + 1)].Where(l => l.StartsWith("M:")).ToList();
        if (_headerMeters.Count != 1 || _headerMeters[0] != $"M:{_first.AbcNumerator}/{_first.AbcDenominator}")
            throw new AbcRebuildException($"ABC header meters [{string.Join(", ", _headerMeters)}] do not match the score");
        var _headerKeys = _lines[..(_keyIndex + 1)].Where(l => l.StartsWith("K:")).ToList();
        if (_headerKeys.Count != 1 || _headerKeys[0] != $"K:{score.KeyArr[_first.StartT]}")
            throw new AbcRebuildException($"ABC header keys [{string.Join(", ", _headerKeys)}] do not match the score");

        int _unit = UnitDenominator(score);
        var _expectedGroups = _groups(score);
        int _cursor = _keyIndex + 1;
        for (int index = 0; index < _expectedGroups.Count; index++)
        {
            var _group = _expectedGroups[index];
            var _labels = new List<string>();
            while (_cursor < _lines.Length && _lines[_cursor].StartsWith("% "))
            {
                _labels.Add(_lines[_cursor][2..].Trim());
                _cursor++;
            }
            if (!_labels.SequenceEqual(_group.StructureLabels))
                throw new AbcRebuildException($"Group {index}: structure labels do not match the score");

            var (cursorAfterVocal, vocalFields, vocalBars) = _parseVoiceGroup(_lines, _cursor, "Vocal", index);
            var (cursorAfterIns, insFields, insBars) = _parseVoiceGroup(_lines, cursorAfterVocal, "Ins", index);
            _cursor = cursorAfterIns;
            if (!_sameFields(vocalFields, insFields))
                throw new AbcRebuildException($"Group {index}: meter/key changes must be scoped to both voices");

            var _expectedFields = new Dictionary<string, string>();
            if (_group.MeterChanged) _expectedFields["M"] = $"{_group.Measures[0].AbcNumerator}/{_group.Measures[0].AbcDenominator}";
            if (_group.KeyChanged) _expectedFields["K"] = score.KeyArr[_group.Measures[0].StartT];
            if (!_sameFields(vocalFields, _expectedFields)) throw new AbcRebuildException($"Group {index}: fields do not match the required changes");
            if (vocalBars.Count != _group.Measures.Count || insBars.Count != _group.Measures.Count)
                throw new AbcRebuildException($"Group {index}: both voices must contain {_group.Measures.Count} measures");

            for (int bar = 0; bar < _group.Measures.Count; bar++)
            {
                var _measure = _group.Measures[bar];
                int _expectedDuration = _measure.AbcNumerator * _unit / _measure.AbcDenominator;
                foreach (var (voiceId, bars) in new[] { ("Vocal", vocalBars), ("Ins", insBars) })
                {
                    var (quoted, keys) = _parseMusicMeasure(bars[bar], _expectedDuration, $"measure {_measure.Index} {voiceId}");
                    var _expectedKeys = _expectedKeyChanges(score, _measure, _unit);
                    if (!keys.SequenceEqual(_expectedKeys))
                        throw new AbcRebuildException($"Measure {_measure.Index} {voiceId}: inline keys do not match the score");
                    if (voiceId == "Vocal")
                    {
                        var _expectedChords = _expectedChordSymbols(score, _measure, _unit);
                        if (!quoted.SequenceEqual(_expectedChords))
                            throw new AbcRebuildException($"Measure {_measure.Index}: chord symbols do not match the score");
                    }
                    else if (quoted.Count > 0) throw new AbcRebuildException($"Measure {_measure.Index}: chords must only be in Vocal");
                }
            }
        }
        if (_cursor != _lines.Length) throw new AbcRebuildException("Unexpected trailing ABC body lines");
    }

    static bool _sameFields(Dictionary<string, string> a, Dictionary<string, string> b) =>
        a.Count == b.Count && a.All(pair => b.TryGetValue(pair.Key, out var value) && value == pair.Value);

    static (int, Dictionary<string, string>, List<string>) _parseVoiceGroup(string[] lines, int cursor, string voiceId, int groupIndex)
    {
        if (cursor >= lines.Length || lines[cursor] != $"V: {voiceId}")
            throw new AbcRebuildException($"Group {groupIndex}: expected V: {voiceId}");
        cursor++;
        var _fields = new Dictionary<string, string>();
        while (cursor < lines.Length && (lines[cursor].StartsWith("M:") || lines[cursor].StartsWith("K:")))
        {
            string _name = lines[cursor][..1];
            if (!_fields.TryAdd(_name, lines[cursor][2..]))
                throw new AbcRebuildException($"Group {groupIndex} {voiceId}: repeated {_name}: field");
            cursor++;
        }
        if (cursor >= lines.Length) throw new AbcRebuildException($"Group {groupIndex} {voiceId}: missing music line");
        string _music = lines[cursor];
        if (_music.StartsWith("V:") || _music.StartsWith("M:") || _music.StartsWith("K:") || _music.StartsWith("%"))
            throw new AbcRebuildException($"Group {groupIndex} {voiceId}: invalid music line");
        cursor++;

        var _split = _music.Split('|');
        if (_split[^1].Trim().Length > 0) throw new AbcRebuildException($"Group {groupIndex} {voiceId}: music line must end with a barline");
        var _serialized = _split[..^1].Select(b => b.Trim()).ToList();
        if (_serialized.Any(b => b.Length == 0)) throw new AbcRebuildException($"Group {groupIndex} {voiceId}: empty serialized measure");

        var _bars = new List<string>();
        foreach (var bar in _serialized)
        {
            var _match = Regex.Match(bar, "^Z([1-4])?$");
            if (!_match.Success)
            {
                _bars.Add(bar);
                continue;
            }
            if (_match.Groups[1].Value == "1") throw new AbcRebuildException($"Group {groupIndex} {voiceId}: Z1 must be written as Z");
            _bars.AddRange(Enumerable.Repeat("Z", _match.Groups[1].Success ? int.Parse(_match.Groups[1].Value) : 1));
        }
        if (_bars.Count is < 1 or > 4) throw new AbcRebuildException($"Group {groupIndex} {voiceId}: expected 1-4 semantic measures");
        return (cursor, _fields, _bars);
    }

    static (List<(int, string)>, List<(int, string)>) _parseMusicMeasure(string body, int expected, string context)
    {
        var _quoted = new List<(int, string)>();
        var _keys = new List<(int, string)>();
        if (body == "Z") return (_quoted, _keys);

        int _position = 0;
        int _cursor = 0;
        foreach (Match match in MusicElement.Matches(body))
        {
            if (body[_cursor..match.Index].Trim().Length > 0)
                throw new AbcRebuildException($"{context}: unsupported serialized ABC tokens '{body[_cursor..match.Index]}'");
            _cursor = match.Index + match.Length;
            if (match.Groups["quoted"].Success)
            {
                _quoted.Add((_position, match.Groups["quoted"].Value));
                continue;
            }
            if (match.Groups["key"].Success)
            {
                _keys.Add((_position, match.Groups["key"].Value));
                continue;
            }
            string _note = match.Groups["note"].Value;
            if (match.Groups["tie"].Value.Length > 0 && _note == "z") throw new AbcRebuildException($"{context}: a rest cannot be tied");
            int _duration = match.Groups["duration"].Value.Length > 0 ? int.Parse(match.Groups["duration"].Value) : 1;
            if (!SupportedDurations.Contains(_duration)) throw new AbcRebuildException($"{context}: duration {_duration} is not parser-representable");
            _position += _duration;
        }
        if (body[_cursor..].Trim().Length > 0) throw new AbcRebuildException($"{context}: unsupported serialized ABC tokens '{body[_cursor..]}'");
        if (_position != expected) throw new AbcRebuildException($"{context}: duration {_position} does not match meter duration {expected}");
        if (Regex.IsMatch(body, @"(^|[\s|])-[_=^A-Ga-g]")) throw new AbcRebuildException($"{context}: tie is written before its second note");
        return (_quoted, _keys);
    }

    static List<(int, string)> _expectedChordSymbols(AbcScore score, Measure measure, int unit)
    {
        int _leading = measure.PadBefore ? PaddingUnits(measure, unit) : 0;
        var _expected = new List<(int, string)>();
        for (int t = measure.StartT; t < measure.EndT; t++)
        {
            if (t != measure.StartT && score.ChordArr[t] == score.ChordArr[t - 1]) continue;
            string? _text = MusicSymbols.ChordSymbolToAbc(score.ChordArr[t]);
            if (_text != null) _expected.Add((_leading + DurationUnits(score, measure.StartT, t, unit), _text));
        }
        return _expected;
    }

    static List<(int, string)> _expectedKeyChanges(AbcScore score, Measure measure, int unit)
    {
        int _leading = measure.PadBefore ? PaddingUnits(measure, unit) : 0;
        var _expected = new List<(int, string)>();
        for (int t = measure.StartT + 1; t < measure.EndT; t++)
            if (score.KeyArr[t] != score.KeyArr[t - 1])
                _expected.Add((_leading + DurationUnits(score, measure.StartT, t, unit), score.KeyArr[t]));
        return _expected;
    }

    static int _lcm(int a, int b) => a / _gcd(a, b) * b;

    static int _gcd(int a, int b) => b == 0 ? a : _gcd(b, a % b);
}
