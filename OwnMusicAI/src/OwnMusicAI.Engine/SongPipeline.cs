using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using SheetSage2;
using YuE2.Song;

namespace OwnMusicAI.Engine;

/// <summary>
/// Same shape as the YuE2 CLI's SongState, so a folder we left behind also resumes with
/// `YuE2.Song --resume song.state.json`.
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
/// The YuE2 song loop (plan score, sample codec tokens, solve latents, decode audio) with progress
/// and cancel, plus the SheetSage2 reference analysis. Both share the GPU, so every call queues on
/// one gate. Every phase saves into the work folder, a stopped song picks up where it was.
/// </summary>
public sealed class SongPipeline : IDisposable
{
    readonly SemaphoreSlim _gate = new SemaphoreSlim(1, 1);
    Yue2Engine? _engine;
    YuE2Tokenizer? _tokenizer;
    string? _engineKey;

    public SongPipeline(ModelPaths paths)
    {
        Paths = paths;
    }

    /// <summary>A change here reloads the engine on the next song.</summary>
    public ModelPaths Paths { get; set; }

    /// <summary>Keeps the 7 GB of YuE2 weights between songs, saves the ~16 s load.</summary>
    public bool KeepLoaded { get; set; } = true;

    /// <summary>
    /// Makes (or finishes) the song in workDir, returns the wav path.
    /// </summary>
    public async Task<string> GenerateAsync(SongRequest request, string workDir, IProgress<GenerationProgress>? progress, CancellationToken ct)
    {
        progress?.Report(new GenerationProgress(GenerationStage.Waiting, 0, "Waiting for the engine"));
        await _gate.WaitAsync(ct);
        try
        {
            return await Task.Run(() => _generate(request, workDir, progress, ct), ct);
        }
        finally
        {
            if (!KeepLoaded) _unload();
            _gate.Release();
        }
    }

    /// <summary>
    /// SheetSage2 over a recording. melodyOnly drops the chord symbols from the score (cover style),
    /// maxSeconds only listens to the start.
    /// </summary>
    public async Task<ReferenceAnalysis> AnalyzeAsync(string audioPath, bool melodyOnly, double? maxSeconds,
        IProgress<GenerationProgress>? progress, CancellationToken ct)
    {
        progress?.Report(new GenerationProgress(GenerationStage.Waiting, 0, "Waiting for the engine"));
        await _gate.WaitAsync(ct);
        try
        {
            return await Task.Run(() => _analyze(audioPath, melodyOnly, maxSeconds, progress, ct), ct);
        }
        finally
        {
            _gate.Release();
        }
    }

    ReferenceAnalysis _analyze(string audioPath, bool melodyOnly, double? maxSeconds, IProgress<GenerationProgress>? progress, CancellationToken ct)
    {
        //SheetSage2 + MERT is another 2.7 GB, never keep it next to YuE2
        _unload();
        progress?.Report(new GenerationProgress(GenerationStage.Transcribing, 0, "Loading SheetSage2"));
        string _mert = SheetSage2Engine.FindMert(Paths.SheetSageDir)
            ?? throw new InvalidOperationException("MERT-v2-FullSong not found, run: `hf download m-a-p/MERT-v2-FullSong`");

        using (var _transcriber = new Transcriber(Paths.SheetSageDir, _mert, Paths.Backend, ComputeDType.Auto))
        {
            var _options = new TranscribeOptions { MelodyOnly = melodyOnly, MaxSeconds = maxSeconds };
            var _result = _transcriber.Transcribe(audioPath, _options, p =>
            {
                ct.ThrowIfCancellationRequested();
                double _done = p.Windows == 0 ? 0 : (p.Window - (p.Stage == "window_complete" ? 0 : 1)) / (double)p.Windows;
                string _what = p.Stage == "decoding" ? $"{p.Tokens} tokens" : p.Stage;
                progress?.Report(new GenerationProgress(GenerationStage.Transcribing, _done, $"window {p.Window}/{p.Windows}, {_what}"));
            });
            return ReferenceAnalysis.From(_result);
        }
    }

    string _generate(SongRequest r, string workDir, IProgress<GenerationProgress>? progress, CancellationToken ct)
    {
        Directory.CreateDirectory(workDir);
        string _stem = Path.Combine(workDir, "song");
        string _statePath = _stem + ".state.json";
        string _latentPath = _stem + ".latents.f32";
        var _state = File.Exists(_statePath) ? JsonSerializer.Deserialize<SongState>(File.ReadAllText(_statePath)) : null;

        var engine = _load(progress);
        if (_state == null)
        {
            _state = _plan(engine, r, _stem, progress, ct);
            File.WriteAllText(_statePath, JsonSerializer.Serialize(_state));
        }
        if (_state.Semantic == null)
        {
            _state.Semantic = _semantic(engine, r, _state, progress, ct);
            File.WriteAllText(_statePath, JsonSerializer.Serialize(_state));
        }

        float[] _latents;
        if (File.Exists(_latentPath) && new FileInfo(_latentPath).Length == (long)_state.Semantic.Count * engine.Info.LatentDim * 4)
        {
            _latents = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(_latentPath)).ToArray();
        }
        else
        {
            var _clock = Stopwatch.StartNew();
            _latents = AcousticSolver.Solve(engine, _state.Prefix, _state.Semantic, _rngSeed(_state.Seed), r.OdeSteps, r.Context, (done, all) =>
            {
                ct.ThrowIfCancellationRequested();
                progress?.Report(new GenerationProgress(GenerationStage.SynthesizingAudio, done / (double)all,
                    $"step {done}/{all}, {_clock.Elapsed.TotalSeconds:F0} s"));
            });
            File.WriteAllBytes(_latentPath, MemoryMarshal.AsBytes(_latents.AsSpan()).ToArray());
        }

        var _audio = VaeTiler.Decode(engine, _latents, r.VaeTile, 16, (done, all) =>
        {
            ct.ThrowIfCancellationRequested();
            progress?.Report(new GenerationProgress(GenerationStage.DecodingAudio, done / (double)all, $"tile {done}/{all}"));
        });

        string _wav = _stem + ".wav";
        WavWriter.Write(_wav, _audio, engine.Info.SampleRate, 2);
        progress?.Report(new GenerationProgress(GenerationStage.Finished, 1, $"{_audio.Length / 2.0 / engine.Info.SampleRate:F1} s of audio"));
        return _wav;
    }

    Yue2Engine _load(IProgress<GenerationProgress>? progress)
    {
        string _vae = Paths.VaeDir ?? ModelPaths.FindVae()
            ?? throw new InvalidOperationException("YuE2-Vae not found, run: `hf download m-a-p/YuE2-Vae`, or set it in Settings");
        string _key = $"{Paths.YuE2Dir}|{_vae}|{Paths.Backend}|{Paths.DType}";
        if (_engine != null && _engineKey == _key) return _engine;

        _unload();
        progress?.Report(new GenerationProgress(GenerationStage.LoadingModel, 0, "Loading YuE2-3B"));
        _engine = new Yue2Engine(Paths.YuE2Dir, _vae, (EngineBackend)(int)Paths.Backend, (EngineDType)(int)Paths.DType);
        _tokenizer = new YuE2Tokenizer(Paths.YuE2Dir);
        _engineKey = _key;
        return _engine;
    }

    SongState _plan(Yue2Engine engine, SongRequest r, string stem, IProgress<GenerationProgress>? progress, CancellationToken ct)
    {
        string _cot = r.Cot;
        var _state = new SongState { Cot = _cot, Seed = r.Seed, Cfg = r.Cfg ?? Protocol.DefaultCfg(_cot) };
        var _base = new List<int> { Protocol.Eod };
        _base.AddRange(_tokenizer!.Encode(Protocol.PromptText(_cot, r.Style, r.Lyrics)));

        _state.Prefix = new List<int>(_base);
        if (_cot == "off")
        {
            _state.Prefix.AddRange(new[] { Protocol.AbcStart, Protocol.AbcEnd, Protocol.MusicStart });
            return _state;
        }

        List<int> _abc;
        if (!string.IsNullOrWhiteSpace(r.Abc))
        {
            _abc = _tokenizer.Encode(r.Abc);
            _state.Abc = r.Abc;
            File.WriteAllText(stem + ".source.abc", r.Abc);
        }
        else
        {
            var _p = r.ScoreSampling?.Apply(Protocol.Abc) ?? Protocol.Abc;
            if (r.MaxAbcTokens is int cap) _p = _p with { MaxTokens = cap, MinTokens = Math.Min(_p.MinTokens, cap) };
            _abc = _sample(engine, new List<int>(_base) { Protocol.AbcStart }, null, 1.0, _p, _state.Seed, true, progress, ct);
            _state.Abc = _tokenizer.Decode(_abc);
            File.WriteAllText(stem + ".abc", _state.Abc);
        }
        _state.Prefix.Add(Protocol.AbcStart);
        _state.Prefix.AddRange(_abc);
        _state.Prefix.Add(Protocol.AbcEnd);
        _state.Prefix.Add(Protocol.MusicStart);
        return _state;
    }

    List<int> _semantic(Yue2Engine engine, SongRequest r, SongState state, IProgress<GenerationProgress>? progress, CancellationToken ct)
    {
        List<int>? _negative = null;
        if (state.Cfg != 1.0)
        {
            //CFG negative: instruction only, same ABC ids
            _negative = new List<int> { Protocol.Eod };
            _negative.AddRange(_tokenizer!.Encode(Protocol.Instructions[state.Cot]));
            if (state.Cot == "off") _negative.Add(Protocol.MusicStart);
            else _negative.AddRange(state.Prefix.SkipWhile(t => t != Protocol.AbcStart));
        }

        var _p = r.MusicSampling?.Apply(Protocol.Semantic) ?? Protocol.Semantic;
        if (r.MaxSeconds is double sec)
        {
            int _cap = Math.Max(1, (int)(sec * 25));
            _p = _p with { MaxTokens = _cap, MinTokens = Math.Min(_p.MinTokens, _cap) };
        }
        var _ids = _sample(engine, state.Prefix, _negative, state.Cfg, _p, state.Seed, false, progress, ct);
        return _ids.Select(id => id - Protocol.CodecOffset).ToList();
    }

    /// <summary>
    /// The AR loop of SongGenerator: abc = score text, otherwise codec tokens; negative != null runs CFG.
    /// </summary>
    List<int> _sample(Yue2Engine engine, List<int> prefix, List<int>? negative, double cfgScale, SamplingParams p, long seed, bool abc,
        IProgress<GenerationProgress>? progress, CancellationToken ct)
    {
        var _stage = abc ? GenerationStage.PlanningScore : GenerationStage.GeneratingMusic;
        var _out = new List<int>();
        var _sampler = new TokenSampler(_rngSeed(seed));
        int _vocab = engine.Info.VocabSize;
        var _cond = new float[_vocab];
        var _uncond = new float[_vocab];
        var _mixed = new float[_vocab];
        int _end = abc ? Protocol.AbcEnd : Protocol.MusicEnd;
        int _max = Math.Min(p.MaxTokens, Protocol.Context - prefix.Count);

        progress?.Report(new GenerationProgress(_stage, 0, $"prefill: {prefix.Count} tokens"));
        using (var _pos = engine.NewCache(prefix.Count + _max))
        using (var _neg = engine.NewCache(negative == null ? 1 : negative.Count + _max))
        {
            _prefill(engine, _pos, prefix, _cond);
            if (negative != null) _prefill(engine, _neg, negative, _uncond);
            var _clock = Stopwatch.StartNew();

            var _one = new int[1];
            for (int step = 0; step < _max; step++)
            {
                ct.ThrowIfCancellationRequested();
                var _logits = _cond;
                if (negative != null)
                {
                    float _scale = (float)cfgScale;
                    for (int i = 0; i < _vocab; i++) _mixed[i] = _uncond[i] + _scale * (_cond[i] - _uncond[i]);
                    _logits = _mixed;
                }

                int _token = _sampler.Next(_logits, p, _out, step, abc);
                if (_token == _end) break;
                _out.Add(_token);
                if (_out.Count % 25 == 0)
                {
                    string _unit = abc ? $"{_out.Count} score tokens" : $"{_out.Count / 25.0:F0} s of music";
                    progress?.Report(new GenerationProgress(_stage, _out.Count / (double)_max,
                        $"{_unit}, {_out.Count / _clock.Elapsed.TotalSeconds:F1} tok/s"));
                }
                if (step + 1 == _max) break;

                _one[0] = _token;
                engine.Forward(_pos, _one, _cond);
                if (negative != null) engine.Forward(_neg, _one, _uncond);
            }
        }
        return _out;
    }

    static void _prefill(Yue2Engine engine, KvCache cache, List<int> tokens, float[] logits)
    {
        var _ids = tokens.ToArray();
        for (int s = 0; s < _ids.Length; s += 1024)
            engine.Forward(cache, _ids.AsSpan(s, Math.Min(1024, _ids.Length - s)), logits);
    }

    static int _rngSeed(long seed) => unchecked((int)(seed ^ (seed >> 32)));

    void _unload()
    {
        _engine?.Dispose();
        _engine = null;
        _engineKey = null;
        _tokenizer = null;
    }

    /// <summary>
    /// Waits a bit for a running step to let go of the engine - a cancelled song stops at the next step.
    /// </summary>
    public void Dispose()
    {
        if (_gate.Wait(TimeSpan.FromSeconds(15))) _unload();
    }
}
