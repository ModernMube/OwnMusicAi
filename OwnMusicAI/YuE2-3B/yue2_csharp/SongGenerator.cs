using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace YuE2.Song;

/// <summary>
/// What survives a stopped run: written after planning and after semantic sampling.
/// </summary>
internal sealed class SongState
{
    public string Cot { get; set; } = "full";
    public long Seed { get; set; }
    public double Cfg { get; set; }
    public string? Abc { get; set; }
    public List<int> Prefix { get; set; } = new List<int>();
    public List<int>? Semantic { get; set; }
}

/// <summary>
/// The YuE2 song loop driven from C#: plan the ABC score, sample codec tokens (with CFG),
/// solve latents, tile-decode audio. The Rust engine only runs the networks.
/// </summary>
internal sealed class SongGenerator
{
    readonly SongOptions _o;
    readonly string _stem;
    readonly YuE2Tokenizer _tokenizer;

    public SongGenerator(SongOptions options)
    {
        _o = options;
        _stem = options.Resume != null
            ? options.Resume.Replace(".state.json", "")
            : Path.ChangeExtension(options.Output, null);
        _tokenizer = new YuE2Tokenizer(options.ModelDir);
    }

    string _statePath => _stem + ".state.json";
    string _latentPath => _stem + ".latents.f32";

    public void Run()
    {
        if (_o.VaeDir == null) throw new InvalidOperationException("YuE2-Vae not found, run `hf download m-a-p/YuE2-Vae` or pass --vae");
        var _total = Stopwatch.StartNew();
        var _state = _o.Resume != null ? JsonSerializer.Deserialize<SongState>(File.ReadAllText(_o.Resume))! : null;

        if (_o.AbcAudio != null && _o.Abc == null && _state == null) _o.Abc = _transcribe();

        var _clock = Stopwatch.StartNew();
        using (var engine = new Yue2Engine(_o.ModelDir, _o.VaeDir, _o.Backend, _o.DType))
        {
            var _info = engine.Info;
            Console.WriteLine($"Loaded engine in {_clock.Elapsed.TotalSeconds:F1}s ({(EngineBackend)_info.Backend}, dtype {(EngineDType)_info.DType})");

            if (_state == null)
            {
                _state = _plan(engine);
                _save(_state);
            }

            if (_state.Semantic == null)
            {
                _state.Semantic = _semantic(engine, _state);
                _save(_state);
            }

            float[] _latents;
            int _frames = _state.Semantic.Count;
            if (File.Exists(_latentPath) && new FileInfo(_latentPath).Length == (long)_frames * _info.LatentDim * 4)
            {
                _latents = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(_latentPath)).ToArray();
                Console.WriteLine($"Latents loaded from {_latentPath}");
            }
            else
            {
                _clock.Restart();
                _latents = AcousticSolver.Solve(engine, _state.Prefix, _state.Semantic, _rngSeed(_state.Seed), _o.OdeSteps, _o.Context,
                    (done, all) => Console.Write($"\r  Synthesizing audio: step {done}/{all}, {_clock.Elapsed.TotalSeconds:F0}s"));
                Console.WriteLine();
                File.WriteAllBytes(_latentPath, MemoryMarshal.AsBytes(_latents.AsSpan()).ToArray());
            }

            _clock.Restart();
            var _audio = VaeTiler.Decode(engine, _latents, _o.VaeTile, 16, (done, all) => Console.Write($"\r  Decoding audio: tile {done}/{all}"));
            Console.WriteLine($", {_clock.Elapsed.TotalSeconds:F1}s");

            string _wav = _stem + ".wav";
            WavWriter.Write(_wav, _audio, _info.SampleRate, 2);
            Console.WriteLine($"Done: {_wav} ({_audio.Length / 2.0 / _info.SampleRate:F1}s audio) in {_total.Elapsed.TotalSeconds:F0}s");
        }
    }

    /// <summary>
    /// SheetSage2 turns a recording into the ABC score YuE2 will follow. --cot melody asks for a
    /// melody-only score (covers), --cot full keeps the chord symbols.
    /// </summary>
    string _transcribe()
    {
        Console.WriteLine($"Transcribing {Path.GetFileName(_o.AbcAudio)} with SheetSage2 ({_o.Cot} score)");
        var _clock = Stopwatch.StartNew();
        var _mert = SheetSage2.SheetSage2Engine.FindMert(_o.SheetSageDir);
        using (var _transcriber = new SheetSage2.Transcriber(_o.SheetSageDir, _mert, (SheetSage2.ComputeBackend)(int)_o.Backend, SheetSage2.ComputeDType.Auto))
        {
            var _options = new SheetSage2.TranscribeOptions { MelodyOnly = _o.Cot == "melody" };
            var _result = _transcriber.Transcribe(_o.AbcAudio!, _options,
                p => { if (p.Tokens > 0) Console.Write($"\r  window {p.Window}/{p.Windows}, {p.Tokens} tokens    "); });
            Console.WriteLine();
            if (_result.Abc == null)
                throw new InvalidOperationException($"SheetSage2 could not build an ABC score: {_result.AbcError}");
            File.WriteAllText(_stem + ".source.abc", _result.Abc);
            Console.WriteLine($"  {_result.Events.Count} events, {_result.AbcMeasures} measures in {_clock.Elapsed.TotalSeconds:F1}s -> {_stem}.source.abc");
            return _result.Abc;
        }
    }

    SongState _plan(Yue2Engine engine)
    {
        var _state = new SongState { Cot = _o.Cot, Seed = _o.Seed, Cfg = _o.Cfg ?? Protocol.DefaultCfg(_o.Cot) };
        var _base = new List<int> { Protocol.Eod };
        _base.AddRange(_tokenizer.Encode(Protocol.PromptText(_o.Cot, _o.Style, _o.Lyrics)));
        Console.WriteLine($"Prompt: {_base.Count} tokens, cot={_o.Cot}, seed={_o.Seed}");

        _state.Prefix = new List<int>(_base);
        if (_o.Cot == "off")
        {
            _state.Prefix.AddRange(new[] { Protocol.AbcStart, Protocol.AbcEnd, Protocol.MusicStart });
            return _state;
        }

        List<int> _abc;
        if (_o.Abc != null)
        {
            _abc = _tokenizer.Encode(_o.Abc);
            _state.Abc = _o.Abc;
        }
        else
        {
            var _p = Protocol.Abc;
            if (_o.MaxAbcTokens is int cap) _p = _p with { MaxTokens = cap, MinTokens = Math.Min(_p.MinTokens, cap) };
            _abc = _sample(engine, new List<int>(_base) { Protocol.AbcStart }, null, 1.0, _p, _state.Seed, true, "Planning score");
            _state.Abc = _tokenizer.Decode(_abc);
            File.WriteAllText(_stem + ".abc", _state.Abc);
        }
        _state.Prefix.Add(Protocol.AbcStart);
        _state.Prefix.AddRange(_abc);
        _state.Prefix.Add(Protocol.AbcEnd);
        _state.Prefix.Add(Protocol.MusicStart);
        return _state;
    }

    List<int> _semantic(Yue2Engine engine, SongState state)
    {
        List<int>? _negative = null;
        if (state.Cfg != 1.0)
        {
            //CFG negative: instruction only, but the exact same ABC ids
            _negative = new List<int> { Protocol.Eod };
            _negative.AddRange(_tokenizer.Encode(Protocol.Instructions[state.Cot]));
            if (state.Cot == "off") _negative.Add(Protocol.MusicStart);
            else _negative.AddRange(state.Prefix.SkipWhile(t => t != Protocol.AbcStart));
        }

        var _p = Protocol.Semantic;
        if (_o.MaxSeconds is double sec)
        {
            int _cap = Math.Max(1, (int)(sec * 25));
            _p = _p with { MaxTokens = _cap, MinTokens = Math.Min(_p.MinTokens, _cap) };
        }
        var _ids = _sample(engine, state.Prefix, _negative, state.Cfg, _p, state.Seed, false, $"Generating song (cfg {state.Cfg})");
        return _ids.Select(id => id - Protocol.CodecOffset).ToList();
    }

    List<int> _sample(Yue2Engine engine, List<int> prefix, List<int>? negative, double cfgScale, SamplingParams p, long seed, bool abc, string label)
    {
        var _out = new List<int>();
        var _sampler = new TokenSampler(_rngSeed(seed));
        int _vocab = engine.Info.VocabSize;
        var _cond = new float[_vocab];
        var _uncond = new float[_vocab];
        var _mixed = new float[_vocab];
        int _end = abc ? Protocol.AbcEnd : Protocol.MusicEnd;
        int _max = Math.Min(p.MaxTokens, Protocol.Context - prefix.Count);
        bool _ended = false;

        var _clock = Stopwatch.StartNew();
        using (var _pos = engine.NewCache(prefix.Count + _max))
        using (var _neg = engine.NewCache(negative == null ? 1 : negative.Count + _max))
        {
            _prefill(engine, _pos, prefix, _cond);
            if (negative != null) _prefill(engine, _neg, negative, _uncond);
            Console.WriteLine($"  {label}: prefill {prefix.Count} tokens in {_clock.Elapsed.TotalSeconds:F1}s");
            var _decode = Stopwatch.StartNew();

            var _one = new int[1];
            for (int step = 0; step < _max; step++)
            {
                var _logits = _cond;
                if (negative != null)
                {
                    float _scale = (float)cfgScale;
                    for (int i = 0; i < _vocab; i++) _mixed[i] = _uncond[i] + _scale * (_cond[i] - _uncond[i]);
                    _logits = _mixed;
                }

                int _token = _sampler.Next(_logits, p, _out, step, abc);
                if (_token == _end) { _ended = true; break; }
                _out.Add(_token);
                if (_out.Count % 25 == 0)
                    Console.Write($"\r  {label}: {_out.Count}/{_max} tokens, {_out.Count / _decode.Elapsed.TotalSeconds:F1} tok/s");
                if (step + 1 == _max) break;

                _one[0] = _token;
                engine.Forward(_pos, _one, _cond);
                if (negative != null) engine.Forward(_neg, _one, _uncond);
            }
        }
        Console.WriteLine($"\r  {label}: {_out.Count} tokens in {_clock.Elapsed.TotalSeconds:F1}s{(_ended ? "" : " (hit the token cap)")}          ");
        return _out;
    }

    static void _prefill(Yue2Engine engine, KvCache cache, List<int> tokens, float[] logits)
    {
        var _ids = tokens.ToArray();
        for (int s = 0; s < _ids.Length; s += 1024)
            engine.Forward(cache, _ids.AsSpan(s, Math.Min(1024, _ids.Length - s)), logits);
    }

    void _save(SongState state)
    {
        File.WriteAllText(_statePath, JsonSerializer.Serialize(state));
        Console.WriteLine($"  saved {_statePath}");
    }

    static int _rngSeed(long seed) => unchecked((int)(seed ^ (seed >> 32)));
}
