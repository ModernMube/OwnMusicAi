using System.Diagnostics;

namespace SheetSage2;

public sealed class TranscribeOptions
{
    /// <summary>Default prompt set: everything the full transcription needs.</summary>
    public static readonly string[] FullTasks = { "timestamp", "downbeat_meter", "structure", "key", "chord_full", "melody_full" };

    public IReadOnlyList<string> Prompts = FullTasks;
    public double? MaxSeconds;
    public double OverlapSeconds = 200.0;
    public double LookaheadSeconds = 100.0;
    /// <summary>Keep both melodies but leave chords out of the ABC and the playback MIDI.</summary>
    public bool MelodyOnly;
}

public readonly record struct TranscribeProgress(string Stage, int Window, int Windows, int Tokens);

/// <summary>
/// Whole-song transcription: sliding windows over the audio, each decoded with the overlap
/// prefix of the previous ones, then stitched and exported (pipeline_sheetsage2.py).
/// </summary>
public sealed class Transcriber : IDisposable
{
    readonly SheetSage2Engine _engine;
    readonly bool _owned;
    readonly SheetSage2Tokenizer _tokenizer;

    public Transcriber(SheetSage2Engine engine, bool ownsEngine = false)
    {
        _engine = engine;
        _owned = ownsEngine;
        _tokenizer = new SheetSage2Tokenizer(engine.Info.WindowSamples / (double)engine.Info.SampleRate, engine.Info.TimeHz);
        if (_tokenizer.TokenCount != engine.Info.VocabSize)
            throw new InvalidOperationException($"Tokenizer vocabulary ({_tokenizer.TokenCount}) does not match the model ({engine.Info.VocabSize})");
    }

    public Transcriber(string modelDir, string? mertDir, ComputeBackend backend, ComputeDType dtype)
        : this(new SheetSage2Engine(modelDir, mertDir, backend, dtype), ownsEngine: true) { }

    public SheetSage2Tokenizer Tokenizer => _tokenizer;

    public TranscriptionResult Transcribe(string audioPath, TranscribeOptions? options = null, Action<TranscribeProgress>? progress = null)
    {
        options ??= new TranscribeOptions();
        var _audio = AudioLoader.Load(audioPath, options.MaxSeconds);
        var _result = Transcribe(_audio, options, progress);
        _result.Audio = Path.GetFileName(audioPath);
        return _result;
    }

    /// <summary>
    /// Runs stitching and export over token sequences someone else decoded (one per window).
    /// Only useful for comparing this pipeline against the reference implementation.
    /// </summary>
    public TranscriptionResult Replay(float[] audio, IReadOnlyList<List<int>> windowTokens, TranscribeOptions? options = null) =>
        Transcribe(audio, options, null, windowTokens);

    /// <summary>Mono samples at the model rate (24 kHz), already trimmed to what you want.</summary>
    public TranscriptionResult Transcribe(float[] audio, TranscribeOptions? options = null, Action<TranscribeProgress>? progress = null,
        IReadOnlyList<List<int>>? replayTokens = null)
    {
        options ??= new TranscribeOptions();
        var _prompts = _tokenizer.NormalizePrompts(options.Prompts);
        if (!_prompts.Contains("timestamp")) throw new ArgumentException("timestamp is required to export timed annotations");
        AudioLoader.Validate(audio);

        var _clock = Stopwatch.StartNew();
        double _duration = audio.Length / (double)_engine.Info.SampleRate;
        double _window = _engine.Info.WindowSamples / (double)_engine.Info.SampleRate;
        var _plan = WindowStitcher.Plan(_duration, _window, options.OverlapSeconds, options.LookaheadSeconds);

        var _result = new TranscriptionResult
        {
            Audio = "audio",
            DurationSeconds = _duration,
            Prompts = _prompts,
            MelodyOnly = options.MelodyOnly
        };
        var _stitched = new List<DecodedEvent>();

        foreach (var window in _plan)
        {
            progress?.Invoke(new TranscribeProgress("encoding", window.Index + 1, _plan.Count, 0));
            List<int>? _prefix = null;
            int _base = 0;
            if (window.Index > 0)
            {
                (_prefix, _base) = WindowStitcher.OverlapPrefix(_stitched, _tokenizer, _prompts, window.Start, window.PrefixEnd);
                if (_prefix != null && _prefix.Count >= _engine.Info.MaxTokens - 128)
                    throw new InvalidOperationException("Overlap prefix fills the context; reduce overlap_seconds");
            }

            double _stop = window.GenerationStop ?? Math.Min(_duration - window.Start, _window);
            List<int> _tokens;
            if (replayTokens != null)
            {
                _tokens = replayTokens[window.Index];
            }
            else
            {
                using (var memory = _engine.Encode(WindowStitcher.Slice(audio, window.Start, _window, _engine.Info.SampleRate)))
                {
                    _tokens = Generation.Generate(_engine, _tokenizer, memory, _prompts, _engine.Info.MaxTokens, _prefix, _stop,
                        count => progress?.Invoke(new TranscribeProgress("decoding", window.Index + 1, _plan.Count, count)));
                }
            }
            if (_tokens.Count > _engine.Info.MaxTokens)
                _result.Warnings.Add($"Window {window.Index + 1} reached the token limit; inspect its token coverage");

            var _decoded = _decodeTokens(_tokens, _result.Warnings);
            var _lookup = new Generation.TimeMap(_decoded, _window);
            _stitched.AddRange(WindowStitcher.Accept(_decoded, _lookup, window, _duration, _base));
            _result.WindowTokens.Add(_tokens);
            _result.WindowPrefixLengths.Add(_prefix?.Count ?? 0);
            progress?.Invoke(new TranscribeProgress("window_complete", window.Index + 1, _plan.Count, _tokens.Count));
        }

        _result.Events = _stitched.OrderBy(e => e.Time!.Value).ThenBy(e => e.GlobalSubbeat).ToList();
        progress?.Invoke(new TranscribeProgress("notation", _plan.Count, _plan.Count, 0));
        Exports.Build(_result, _tokenizer, options.MelodyOnly);
        _result.ElapsedSeconds = _clock.Elapsed.TotalSeconds;
        return _result;
    }

    /// <summary>Writes the reference's output set: score.abc, events, LAB annotations and MIDI.</summary>
    public void Save(TranscriptionResult result, string outputDir)
    {
        Directory.CreateDirectory(outputDir);
        File.WriteAllText(Path.Combine(outputDir, "events.json"), Exports.EventsJson(result));
        foreach (var (name, text) in result.Labs)
        {
            string _path = Path.Combine(outputDir, name);
            Directory.CreateDirectory(Path.GetDirectoryName(_path)!);
            File.WriteAllText(_path, text);
        }
        foreach (var (name, bytes) in result.Midis) File.WriteAllBytes(Path.Combine(outputDir, $"{name}.mid"), bytes);

        string _abcPath = Path.Combine(outputDir, "score.abc");
        if (result.Abc != null) File.WriteAllText(_abcPath, result.Abc);
        else File.Delete(_abcPath);

        using (var _tokens = new StreamWriter(Path.Combine(outputDir, "tokens.txt")))
        {
            for (int i = 0; i < result.WindowTokens.Count; i++)
            {
                _tokens.WriteLine($"# window_index={i} tokens={result.WindowTokens[i].Count}");
                for (int t = 0; t < result.WindowTokens[i].Count; t++)
                    _tokens.WriteLine($"{t}\t{result.WindowTokens[i][t]}\t{_tokenizer.Describe(result.WindowTokens[i][t])}");
                _tokens.WriteLine();
            }
        }
        File.WriteAllText(Path.Combine(outputDir, "result.json"), _summary(result));
    }

    public void Dispose()
    {
        if (_owned) _engine.Dispose();
    }

    /// <summary>Strict decode, falling back to the lenient one for the two recoverable errors.</summary>
    DecodedSequence _decodeTokens(List<int> tokens, List<string> warnings)
    {
        try
        {
            return _tokenizer.DecodeSequence(tokens, strict: true);
        }
        catch (FormatException exception) when (exception.Message.Contains("empty event at subbeat")
            || exception.Message.Contains("belongs to inactive output field"))
        {
            warnings.Add(exception.Message);
            return _tokenizer.DecodeSequence(tokens, strict: false);
        }
    }

    static string _summary(TranscriptionResult result)
    {
        var _notes = result.Events.SelectMany(e => e.Values.Melody ?? Enumerable.Empty<MelodyNote>()).ToList();
        string _escape(string value) => System.Text.Json.JsonSerializer.Serialize(value);
        var _lines = new List<string>
        {
            $"  \"audio\": {_escape(result.Audio)}",
            $"  \"duration_seconds\": {Exports.Py(result.DurationSeconds)}",
            $"  \"prompts\": [{string.Join(", ", result.Prompts.Select(_escape))}]",
            $"  \"melody_only\": {(result.MelodyOnly ? "true" : "false")}",
            $"  \"events\": {result.Events.Count}",
            $"  \"melody_notes\": {_notes.Count}",
            $"  \"vocal_notes\": {_notes.Count(n => n.Track == 0)}",
            $"  \"instrumental_notes\": {_notes.Count(n => n.Track == 1)}",
            $"  \"abc_measures\": {result.AbcMeasures}",
            $"  \"abc_error\": {(result.AbcError == null ? "null" : _escape(result.AbcError))}",
            $"  \"windows\": [{string.Join(", ", result.WindowTokens.Select(t => t.Count))}]",
            $"  \"window_prefix_tokens\": [{string.Join(", ", result.WindowPrefixLengths)}]",
            $"  \"elapsed_seconds\": {Exports.Py(Math.Round(result.ElapsedSeconds, 3))}",
            $"  \"warnings\": [{string.Join(", ", result.Warnings.Select(_escape))}]",
            $"  \"diagnostics\": [{string.Join(", ", result.Diagnostics.Select(_escape))}]"
        };
        return "{\n" + string.Join(",\n", _lines) + "\n}\n";
    }
}
