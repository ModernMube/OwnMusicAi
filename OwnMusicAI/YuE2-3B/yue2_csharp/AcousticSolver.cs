namespace YuE2.Song;

/// <summary>
/// Codec tokens -> VAE latents: chunking, song-wide noise and the midpoint ODE (yue2/nar.py),
/// with the engine only doing the velocity evaluations.
/// </summary>
internal static class AcousticSolver
{
    const int PrefillBlock = 1024;

    /// <summary>
    /// Returns [frames * 64] row-major latents. context splits long songs into chunks
    /// (24576 = reference; smaller means shorter chunks, less memory, slightly different seams).
    /// </summary>
    public static float[] Solve(Yue2Engine engine, List<int> prefix, List<int> codec, int seed, int steps, int context,
        Action<int, int> progress)
    {
        int _dim = engine.Info.LatentDim;
        int _frames = codec.Count;
        var _latents = new float[_frames * _dim];
        _fillNoise(_latents, seed);

        int _size = Math.Min((context - prefix.Count - 3) / 2, Protocol.Context);
        if (_size < 1) throw new InvalidOperationException($"prefix of {prefix.Count} tokens leaves no room in a {context} context");
        int _chunks = (_frames + _size - 1) / _size;
        var _scratch = new float[engine.Info.VocabSize];

        int _chunk = 0;
        for (int a = 0; a < _frames; a += _size)
        {
            int b = Math.Min(_frames, a + _size);
            var _tokens = new List<int>(prefix.Count + b - a + 1);
            _tokens.AddRange(prefix);
            for (int i = a; i < b; i++) _tokens.Add(Protocol.CodecOffset + codec[i]);
            _tokens.Add(Protocol.MusicEnd);

            int _done = _chunk * steps;
            using (var _cache = engine.NewCache(_tokens.Count))
            {
                var _ids = _tokens.ToArray();
                for (int s = 0; s < _ids.Length; s += PrefillBlock)
                    engine.Forward(_cache, _ids.AsSpan(s, Math.Min(PrefillBlock, _ids.Length - s)), _scratch);

                _midpoint(engine, _cache, _latents.AsSpan(a * _dim, (b - a) * _dim), steps, st => progress(_done + st, _chunks * steps));
            }
            _chunk++;
        }
        return _latents;
    }

    static void _midpoint(Yue2Engine engine, KvCache cache, Span<float> x, int steps, Action<int> progress)
    {
        var _mid = new float[x.Length];
        var _v = new float[x.Length];
        double dt = 1.0 / steps;
        float _half = (float)(dt / 2), _full = (float)dt;

        for (int step = 0; step < steps; step++)
        {
            double t = 1.0 - step * dt;
            engine.Velocity(cache, x, _logit(t), _v);
            for (int i = 0; i < x.Length; i++) _mid[i] = x[i] - _v[i] * _half;

            engine.Velocity(cache, _mid, _logit(t - dt / 2), _v);
            for (int i = 0; i < x.Length; i++) x[i] -= _v[i] * _full;
            progress(step + 1);
        }
    }

    static float _logit(double t) => (float)Math.Clamp(Math.Log(t / (1 - t)), -20, 20);

    static void _fillNoise(float[] buffer, int seed)
    {
        var rng = new Random(seed);
        for (int i = 0; i < buffer.Length; i += 2)
        {
            double r = Math.Sqrt(-2.0 * Math.Log(1.0 - rng.NextDouble()));
            double a = 2 * Math.PI * rng.NextDouble();
            buffer[i] = (float)(r * Math.Cos(a));
            if (i + 1 < buffer.Length) buffer[i + 1] = (float)(r * Math.Sin(a));
        }
    }
}
