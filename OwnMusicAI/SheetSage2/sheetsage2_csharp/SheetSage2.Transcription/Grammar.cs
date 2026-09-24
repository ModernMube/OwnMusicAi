namespace SheetSage2;

/// <summary>
/// The event grammar generation is constrained to (PromptGrammarState in the reference): which
/// token types may follow, given what the sequence has emitted so far.
/// </summary>
internal sealed class PromptGrammar
{
    enum Pending { None, RhythmAfterMeter, MelodyAfterPitch }

    readonly SheetSage2Tokenizer _t;
    readonly List<(int Start, int End)> _allowed = new List<(int, int)>(8);
    bool _inShift = true;
    int _shiftRun;
    int _payloadCount;
    int _lastField = -1;
    Pending _pending = Pending.None;

    public PromptGrammar(SheetSage2Tokenizer tokenizer)
    {
        _t = tokenizer;
    }

    /// <summary>Highest-scoring allowed token; ties go to the lowest id, like torch argmax.</summary>
    public int Pick(ReadOnlySpan<float> logits)
    {
        _fillAllowed();
        int _best = -1;
        foreach (var (start, end) in _allowed)
            for (int token = start; token < end; token++)
                if (_best < 0 || logits[token] > logits[_best]) _best = token;
        return _best < 0 ? SheetSage2Tokenizer.EosToken : _best;
    }

    /// <summary>Feeds a token back in; true once the sequence is finished.</summary>
    public bool Update(int token)
    {
        if (token == SheetSage2Tokenizer.EosToken) return true;
        var _type = _t.TypeOf(token);
        if (_type == TokenType.SubbeatShift)
        {
            if (!_inShift && _payloadCount > 0)
            {
                _payloadCount = 0;
                _lastField = -1;
                _pending = Pending.None;
            }
            _inShift = true;
            _shiftRun++;
            return false;
        }

        _inShift = false;
        _shiftRun = 0;
        _payloadCount++;
        _pending = Pending.None;
        switch (_type)
        {
            case TokenType.Time: _lastField = (int)EventField.Timestamp; break;
            case TokenType.Meter:
                _lastField = (int)EventField.Rhythm;
                _pending = Pending.RhythmAfterMeter;
                break;
            case TokenType.EighthPosition: _lastField = (int)EventField.Rhythm; break;
            case TokenType.Structure: _lastField = (int)EventField.Structure; break;
            case TokenType.Key: _lastField = (int)EventField.Key; break;
            case TokenType.ChordFull: _lastField = (int)EventField.Chord; break;
            case TokenType.Pitch:
                _lastField = (int)EventField.Melody;
                _pending = Pending.MelodyAfterPitch;
                break;
            case TokenType.Duration: _lastField = (int)EventField.Melody; break;
            default: throw new InvalidOperationException($"Unexpected prompt token type {_type}");
        }
        return false;
    }

    //Ranges are collected in ascending token order so the scan above keeps torch's tie-break
    void _fillAllowed()
    {
        _allowed.Clear();
        if (_payloadCount > 0) _allowed.Add((SheetSage2Tokenizer.EosToken, SheetSage2Tokenizer.EosToken + 1));
        if ((_payloadCount > 0 || _inShift) && _shiftRun < 4) _allowed.Add((_t.SubbeatShiftStart, _t.SubbeatShiftEnd));

        if (_pending == Pending.RhythmAfterMeter)
        {
            _allowed.Add((_t.EighthStart, _t.EighthEnd));
            return;
        }
        if (_pending == Pending.MelodyAfterPitch)
        {
            _allowed.Add((_t.PitchStart, _t.PitchEnd));
            _allowed.Add((_t.DurationStart, _t.DurationEnd));
            return;
        }

        if (_lastField < (int)EventField.Timestamp) _allowed.Add((_t.TimeStart, _t.TimeEnd));
        if (_lastField < (int)EventField.Rhythm)
        {
            _allowed.Add((_t.MeterStart, _t.MeterEnd));
            _allowed.Add((_t.EighthStart, _t.EighthEnd));
        }
        if (_lastField < (int)EventField.Structure) _allowed.Add((_t.StructureStart, _t.StructureEnd));
        if (_lastField < (int)EventField.Key) _allowed.Add((_t.KeyStart, _t.KeyEnd));
        if (_lastField < (int)EventField.Chord) _allowed.Add((_t.FullChordStart, _t.FullChordEnd));
        if (_lastField <= (int)EventField.Melody) _allowed.Add((_t.PitchStart, _t.PitchEnd));
    }
}
