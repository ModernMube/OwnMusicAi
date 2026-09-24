namespace YuE2.Song;

/// <summary>
/// C# take on yue2.sampling.distribution + the multinomial draw.
/// Same math, System.Random instead of torch's generator - seeds won't match Python.
/// </summary>
internal sealed class TokenSampler
{
    readonly Random _rng;
    readonly float[] _keys;
    readonly int[] _slots;
    readonly double[] _probs;
    readonly Dictionary<int, int> _counts = new Dictionary<int, int>();

    public TokenSampler(int seed)
    {
        _rng = new Random(seed);
        int _max = Math.Max(Protocol.Eod, Protocol.CodecSize) + 1;
        _keys = new float[_max];
        _slots = new int[_max];
        _probs = new double[_max];
    }

    /// <summary>
    /// abc = true draws score text ([0, EOD) + ABC_END), otherwise codec ids + MUSIC_END.
    /// </summary>
    public int Next(float[] logits, SamplingParams p, List<int> history, int step, bool abc)
    {
        int _first = abc ? 0 : Protocol.CodecOffset;
        int _span = abc ? Protocol.Eod : Protocol.CodecSize;
        int _end = abc ? Protocol.AbcEnd : Protocol.MusicEnd;
        int n = _span + 1;

        Array.Copy(logits, _first, _keys, 0, _span);
        _keys[_span] = step < p.MinTokens ? float.NegativeInfinity : logits[_end];

        if (p.RepetitionPenalty != 1.0 && history.Count > 0)
        {
            _counts.Clear();
            for (int i = Math.Max(0, history.Count - p.PenaltyWindow); i < history.Count; i++)
                _counts[history[i]] = _counts.GetValueOrDefault(history[i]) + 1;
            foreach (var (id, count) in _counts)
            {
                int slot = id - _first;
                float _alpha = (float)Math.Pow(p.RepetitionPenalty, count);
                _keys[slot] = _keys[slot] < 0 ? _keys[slot] * _alpha : _keys[slot] / _alpha;
            }
        }

        if (p.Temperature == 0)
        {
            int _argmax = 0;
            for (int i = 1; i < n; i++) if (_keys[i] > _keys[_argmax]) _argmax = i;
            return _token(_argmax, _first, _span, _end);
        }

        float _invTemp = (float)(1.0 / p.Temperature);
        for (int i = 0; i < n; i++)
        {
            _slots[i] = i;
            _keys[i] *= _invTemp;
        }
        Array.Sort(_keys, _slots, 0, n);

        //Ascending now. top_k keeps ties at the threshold, like torch's `scores < threshold` mask
        int _lo = n - Math.Min(p.TopK, n);
        float _threshold = _keys[_lo];
        while (_lo > 0 && _keys[_lo - 1] >= _threshold) _lo--;

        double _best = _keys[n - 1], _sum = 0;
        for (int i = n - 1; i >= _lo; i--)
        {
            _probs[i] = Math.Exp(_keys[i] - _best);
            _sum += _probs[i];
        }

        int _cut = _lo;
        if (p.TopP < 1)
        {
            double _before = 0;
            for (int i = n - 1; i >= _lo; i--)
            {
                if (i < n - 1 && _before > p.TopP) { _cut = i + 1; break; }
                _before += _probs[i] / _sum;
            }
        }

        double _kept = 0;
        for (int i = n - 1; i >= _cut; i--) _kept += _probs[i];
        double _r = _rng.NextDouble() * _kept;
        for (int i = n - 1; i > _cut; i--)
        {
            _r -= _probs[i];
            if (_r < 0) return _token(_slots[i], _first, _span, _end);
        }
        return _token(_slots[_cut], _first, _span, _end);
    }

    static int _token(int slot, int first, int span, int end) => slot == span ? end : first + slot;
}
