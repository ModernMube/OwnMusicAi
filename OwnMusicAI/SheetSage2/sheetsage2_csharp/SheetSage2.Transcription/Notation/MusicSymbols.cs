using System.Text.RegularExpressions;

namespace SheetSage2.Notation;

/// <summary>Deterministic reconstruction failed; the caller turns this into abc_error.</summary>
public class AbcRebuildException : Exception
{
    public AbcRebuildException(string message) : base(message) { }
}

public sealed class BeatGridException : AbcRebuildException
{
    public BeatGridException(string message) : base(message) { }
}

public sealed class ChordSymbolException : AbcRebuildException
{
    public ChordSymbolException(string message) : base(message) { }
}

public sealed class MelodyVoiceException : AbcRebuildException
{
    public MelodyVoiceException(string message) : base(message) { }
}

/// <summary>
/// Chord, key and pitch spelling for ABC output (notation_sheetsage2.py). Spelling stays
/// key-relative, which is why remote keys can produce double accidentals.
/// </summary>
internal static class MusicSymbols
{
    public const string Letters = "CDEFGAB";

    static readonly Dictionary<string, string> QualityToAbc = new Dictionary<string, string>
    {
        { "maj", "" }, { "min", "m" }, { "dim", "dim" }, { "aug", "aug" }, { "7", "7" }, { "maj7", "maj7" },
        { "min7", "m7" }, { "dim7", "dim7" }, { "hdim7", "m7b5" }, { "sus4", "sus4" }, { "sus2", "sus2" },
        { "maj6", "6" }, { "min6", "m6" }, { "sus4(b7)", "7sus4" }, { "minmaj7", "m(maj7)" }
    };

    static readonly int[] NaturalPitchClasses = { 0, 2, 4, 5, 7, 9, 11 };
    static readonly string[] SharpNames = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };
    static readonly string[] FlatNames = { "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B" };

    static readonly Regex RootRe = new Regex(@"^([A-G])(#{0,2}|b{0,2})$", RegexOptions.Compiled);
    static readonly Regex BassDegreeRe = new Regex(@"^(#{0,2}|b{0,2})([1-9]|1[0-3])$", RegexOptions.Compiled);

    public static readonly Dictionary<string, int> KeySignatureAccidentals = new Dictionary<string, int>
    {
        { "C", 0 }, { "G", 1 }, { "D", 2 }, { "A", 3 }, { "E", 4 }, { "B", 5 }, { "F#", 6 }, { "C#", 7 },
        { "F", -1 }, { "Bb", -2 }, { "Eb", -3 }, { "Ab", -4 }, { "Db", -5 }, { "Gb", -6 }, { "Cb", -7 },
        { "Am", 0 }, { "Em", 1 }, { "Bm", 2 }, { "F#m", 3 }, { "C#m", 4 }, { "G#m", 5 }, { "D#m", 6 }, { "A#m", 7 },
        { "Dm", -1 }, { "Gm", -2 }, { "Cm", -3 }, { "Fm", -4 }, { "Bbm", -5 }, { "Ebm", -6 }, { "Abm", -7 }
    };

    //Key-relative chromatic spelling; MIDI carries no note names, so remote keys keep double accidentals
    static readonly Dictionary<int, string[]> KeyRelativeNames = new Dictionary<int, string[]>
    {
        { 7, new[] { "B#", "C#", "C##", "D#", "D##", "E#", "F#", "F##", "G#", "G##", "A#", "B" } },
        { 6, new[] { "B#", "C#", "C##", "D#", "E", "E#", "F#", "F##", "G#", "G##", "A#", "B" } },
        { 5, new[] { "B#", "C#", "C##", "D#", "E", "E#", "F#", "F##", "G#", "A", "A#", "B" } },
        { 4, new[] { "B#", "C#", "D", "D#", "E", "E#", "F#", "F##", "G#", "A", "A#", "B" } },
        { 3, new[] { "B#", "C#", "D", "D#", "E", "E#", "F#", "G", "G#", "A", "A#", "B" } },
        { 2, new[] { "C", "C#", "D", "D#", "E", "E#", "F#", "G", "G#", "A", "A#", "B" } },
        { 1, new[] { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" } },
        { 0, new[] { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B" } },
        { -1, new[] { "C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B" } },
        { -2, new[] { "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B" } },
        { -3, new[] { "C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B" } },
        { -4, new[] { "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B" } },
        { -5, new[] { "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "Cb" } },
        { -6, new[] { "C", "Db", "D", "Eb", "Fb", "F", "Gb", "G", "Ab", "A", "Bb", "Cb" } },
        { -7, new[] { "C", "Db", "D", "Eb", "Fb", "F", "Gb", "G", "Ab", "Bbb", "Bb", "Cb" } }
    };

    public static (int PitchClass, char Letter, string Accidental) Spell(string root)
    {
        var _match = RootRe.Match(root);
        if (!_match.Success) throw new ChordSymbolException($"Invalid pitch spelling '{root}'");
        char _letter = _match.Groups[1].Value[0];
        string _accidental = _match.Groups[2].Value;
        int _offset = _accidental.Count(c => c == '#') - _accidental.Count(c => c == 'b');
        return (((NaturalPitchClasses[Letters.IndexOf(_letter)] + _offset) % 12 + 12) % 12, _letter, _accidental);
    }

    public static string PortablePitchName(string root, bool preserveDouble = false)
    {
        var (pitchClass, _, accidental) = Spell(root);
        if (preserveDouble || accidental.Length <= 1) return root;
        return (accidental.StartsWith('#') ? SharpNames : FlatNames)[pitchClass];
    }

    /// <summary>Turns a chord bass degree ("b3", "5", ...) into a note name relative to the root.</summary>
    public static string BassDegreeToPitch(string root, string degreeText)
    {
        if (RootRe.IsMatch(degreeText)) return PortablePitchName(degreeText, preserveDouble: true);
        var _match = BassDegreeRe.Match(degreeText);
        if (!_match.Success) throw new ChordSymbolException($"Invalid chord bass degree '{degreeText}'");

        var (rootClass, rootLetter, rootAccidental) = Spell(root);
        int _degree = int.Parse(_match.Groups[2].Value);
        string _degreeAccidental = _match.Groups[1].Value;
        int[] _scale = { 0, 2, 4, 5, 7, 9, 11 };
        int _interval = _scale[(_degree - 1) % 7] + 12 * ((_degree - 1) / 7);
        _interval += _degreeAccidental.Count(c => c == '#') - _degreeAccidental.Count(c => c == 'b');
        int _target = ((rootClass + _interval) % 12 + 12) % 12;

        char _letter = Letters[(Letters.IndexOf(rootLetter) + _degree - 1) % 7];
        int _natural = NaturalPitchClasses[Letters.IndexOf(_letter)];
        int _difference = ((_target - _natural + 6) % 12 + 12) % 12 - 6;
        if (_difference >= -2 && _difference <= 2)
        {
            string _accidental = _difference switch { -2 => "bb", -1 => "b", 0 => "", 1 => "#", _ => "##" };
            return _letter + _accidental;
        }
        return ((rootAccidental + _degreeAccidental).Contains('#') ? SharpNames : FlatNames)[_target];
    }

    /// <summary>ABC chord text, or null for no-chord. Unknown qualities are refused, not flattened.</summary>
    public static string? ChordSymbolToAbc(string chord)
    {
        chord = chord.Trim();
        if (chord is "N" or "X" or "?") return null;
        if (!chord.Contains(':')) throw new ChordSymbolException($"Chord '{chord}' is missing the ':' quality separator");

        string _root = chord[..chord.IndexOf(':')];
        string _descriptor = chord[(chord.IndexOf(':') + 1)..];
        string _quality = _descriptor;
        string? _bass = null;
        if (_descriptor.Contains('/'))
        {
            _quality = _descriptor[.._descriptor.IndexOf('/')];
            _bass = _descriptor[(_descriptor.IndexOf('/') + 1)..];
        }
        if (!QualityToAbc.TryGetValue(_quality, out var suffix))
            throw new ChordSymbolException($"Unsupported chord quality '{_quality}' in '{chord}'; refusing to rewrite it as major");

        string _text = PortablePitchName(_root, preserveDouble: true) + suffix;
        if (!string.IsNullOrEmpty(_bass)) _text += "/" + BassDegreeToPitch(_root, _bass);
        return _text;
    }

    public static string KeySymbolToAbc(string key)
    {
        key = key.Trim();
        string _root;
        string _mode;
        if (key.Contains(':'))
        {
            _root = key[..key.IndexOf(':')];
            _mode = key[(key.IndexOf(':') + 1)..];
            if (_mode != "major" && _mode != "minor") throw new AbcRebuildException($"Unsupported key mode '{_mode}' in '{key}'");
        }
        else if (key.EndsWith('m'))
        {
            _root = key[..^1];
            _mode = "minor";
        }
        else
        {
            _root = key;
            _mode = "major";
        }

        var (pitchClass, _, accidental) = Spell(_root);
        string _minor = _mode == "minor" ? "m" : "";
        string _candidate = PortablePitchName(_root) + _minor;
        if (KeySignatureAccidentals.ContainsKey(_candidate)) return _candidate;

        var _names = accidental.Contains('b') ? FlatNames : SharpNames;
        _candidate = _names[pitchClass] + _minor;
        if (!KeySignatureAccidentals.ContainsKey(_candidate))
            _candidate = (_names == FlatNames ? SharpNames : FlatNames)[pitchClass] + _minor;
        if (!KeySignatureAccidentals.ContainsKey(_candidate)) throw new AbcRebuildException($"Cannot encode portable ABC key for '{key}'");
        return _candidate;
    }

    /// <summary>Per letter (C..B) accidental of a key signature: -1 flat, 0 natural, 1 sharp.</summary>
    public static int[] KeyAccidentals(string key)
    {
        if (!KeySignatureAccidentals.TryGetValue(key, out int count))
            throw new AbcRebuildException($"Unsupported ABC key signature '{key}'");
        var _accidentals = new int[7];
        string _order = count > 0 ? "FCGDAEB" : "BEADGCF";
        foreach (char letter in _order[..Math.Abs(count)]) _accidentals[Letters.IndexOf(letter)] = count > 0 ? 1 : -1;
        return _accidentals;
    }

    /// <summary>
    /// ABC note text for a MIDI pitch, writing only accidentals the bar has not seen yet.
    /// measureAccidentals is keyed by letter index and the caller clears it per bar.
    /// </summary>
    public static string NoteToAbc(int note, int[] keyAccidentals, Dictionary<int, int> measureAccidentals)
    {
        int _count = keyAccidentals.Sum();
        if (!KeyRelativeNames.TryGetValue(_count, out var names))
            throw new AbcRebuildException($"Unsupported key signature accidental count {_count}");

        string _name = names[note % 12];
        char _letter = _name[0];
        int _accidental = _name[1..] switch { "" => 0, "#" => 1, "##" => 2, "b" => -1, "bb" => -2, _ => 0 };
        int _octave = (int)Math.Floor((note - 60) / 12.0);
        //Cb and B# cross the MIDI octave boundary even though their letter does not
        if (note % 12 == 11 && _accidental == -1) _octave += 1;
        else if (note % 12 == 0 && _accidental == 1) _octave -= 1;

        int _index = Letters.IndexOf(_letter);
        int _current = measureAccidentals.TryGetValue(_index, out int seen) ? seen : keyAccidentals[_index];
        string _prefix = "";
        if (_current != _accidental)
        {
            measureAccidentals[_index] = _accidental;
            _prefix = _accidental switch { -2 => "__", -1 => "_", 0 => "=", 1 => "^", _ => "^^" };
        }

        string _text = _letter.ToString();
        if (_octave > 0)
        {
            _text = _text.ToLowerInvariant();
            if (_octave > 1) _text += new string('\'', _octave - 1);
        }
        else if (_octave < 0) _text += new string(',', -_octave);
        return _prefix + _text;
    }
}
