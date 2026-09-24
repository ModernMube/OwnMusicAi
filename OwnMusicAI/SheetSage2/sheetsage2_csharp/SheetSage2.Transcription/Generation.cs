namespace SheetSage2;

/// <summary>
/// Grammar-constrained greedy decoding of one window, and the subbeat -> seconds map its
/// timestamp tokens imply.
/// </summary>
internal static class Generation
{
    /// <summary>
    /// Decodes until the grammar emits eos, a timestamp passes stopTimeSeconds, or maxLength is
    /// reached. prefixTokens carries the overlap context of a continued window.
    /// </summary>
    public static List<int> Generate(SheetSage2Engine engine, SheetSage2Tokenizer tokenizer, EncoderMemory memory,
        IReadOnlyList<string> prompts, int maxLength, IReadOnlyList<int>? prefixTokens, double? stopTimeSeconds,
        Action<int>? progress)
    {
        var _prefix = prefixTokens != null ? new List<int>(prefixTokens) : tokenizer.PromptPrefix(prompts);
        if (_prefix.Count == 0 || _prefix[0] != SheetSage2Tokenizer.SosToken)
            throw new ArgumentException("generation prefix must begin with <|sos|>");
        if (_prefix[^1] == SheetSage2Tokenizer.EosToken) _prefix.RemoveAt(_prefix.Count - 1);

        var _grammar = new PromptGrammar(tokenizer);
        for (int i = _prefix.IndexOf(SheetSage2Tokenizer.OutToken) + 1; i < _prefix.Count; i++) _grammar.Update(_prefix[i]);

        var _output = new List<int>(_prefix);
        var _logits = new float[engine.Info.VocabSize];
        int _length = _prefix.Count;
        var _one = new int[1];

        using (var _cache = engine.NewCache(maxLength))
        {
            engine.Decode(memory, _cache, _prefix.ToArray(), _logits);
            while (_length < maxLength)
            {
                int _token = _grammar.Pick(_logits);
                _output.Add(_token);
                bool _finished = _grammar.Update(_token);
                if (!_finished && stopTimeSeconds is double stop && tokenizer.TypeOf(_token) == TokenType.Time
                    && tokenizer.TimeIdOf(_token) / (double)tokenizer.TimeHz >= stop)
                {
                    _output.Add(SheetSage2Tokenizer.EosToken);
                    _finished = true;
                }
                _length++;
                if (_length % 64 == 0) progress?.Invoke(_length);
                if (_finished) break;

                _one[0] = _token;
                engine.Decode(memory, _cache, _one, _logits);
            }
        }
        if (_output[^1] != SheetSage2Tokenizer.EosToken) _output.Add(SheetSage2Tokenizer.EosToken);
        return _output;
    }

    /// <summary>
    /// Subbeat -> seconds inside a window, anchored on the decoded timestamps and extrapolated
    /// with their median spacing (0.125 s when the window has no anchors).
    /// </summary>
    public sealed class TimeMap
    {
        readonly double[] _steps;
        readonly double[] _times;
        readonly double _stepSeconds;
        readonly double _target;

        public TimeMap(DecodedSequence decoded, double targetSeconds)
        {
            _target = targetSeconds;
            var _anchors = new Dictionary<int, double>();
            foreach (var item in decoded.Events)
                if (item.Values.Timestamp is double value) _anchors[item.Subbeat] = value;

            _steps = _anchors.Keys.Select(k => (double)k).Order().ToArray();
            _times = _steps.Select(s => _anchors[(int)s]).ToArray();
            _stepSeconds = 0.125;
            if (_steps.Length >= 2)
            {
                var _spacing = Enumerable.Range(0, _steps.Length - 1)
                    .Select(i => (_times[i + 1] - _times[i]) / Math.Max(_steps[i + 1] - _steps[i], 1))
                    .ToArray();
                double _median = _medianOf(_spacing);
                if (double.IsFinite(_median) && _median > 0) _stepSeconds = _median;
            }
        }

        public double Lookup(double step)
        {
            if (_steps.Length == 0) return Math.Min(_target, Math.Max(0.0, step * 0.125));
            if (step <= _steps[0]) return Math.Clamp(_times[0] + (step - _steps[0]) * _stepSeconds, 0, _target);
            if (step >= _steps[^1]) return Math.Clamp(_times[^1] + (step - _steps[^1]) * _stepSeconds, 0, _target);

            int _upper = Array.BinarySearch(_steps, step);
            if (_upper >= 0) return _times[_upper];
            _upper = ~_upper;
            double _slope = (_times[_upper] - _times[_upper - 1]) / (_steps[_upper] - _steps[_upper - 1]);
            return _slope * (step - _steps[_upper - 1]) + _times[_upper - 1];
        }

        static double _medianOf(double[] values)
        {
            var _sorted = values.Order().ToArray();
            int _mid = _sorted.Length / 2;
            return _sorted.Length % 2 == 1 ? _sorted[_mid] : (_sorted[_mid - 1] + _sorted[_mid]) / 2;
        }
    }
}
