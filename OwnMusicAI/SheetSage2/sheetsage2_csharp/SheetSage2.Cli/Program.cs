using System.Diagnostics;
using System.Globalization;
using SheetSage2;

namespace SheetSage2.Cli;

/// <summary>
/// SheetSage2 transcription from the command line: audio in, ABC/MIDI/annotations out.
/// </summary>
internal static class Program
{
    static int Main(string[] args)
    {
        var _options = CliOptions.Parse(args);
        if (_options == null)
        {
            CliOptions.PrintUsage();
            return 1;
        }

        var _clock = Stopwatch.StartNew();
        using (var transcriber = new Transcriber(_options.ModelDir, _options.MertDir, _options.Backend, _options.DType))
        {
            Console.WriteLine($"Loaded engine in {_clock.Elapsed.TotalSeconds:F1}s ({_options.Backend})");
            _clock.Restart();
            var _result = _options.Replay == null
                ? transcriber.Transcribe(_options.Audio, _options.Transcribe, _report)
                : transcriber.Replay(AudioLoader.Load(_options.Audio, _options.Transcribe.MaxSeconds), _readTokens(_options.Replay), _options.Transcribe);
            Console.WriteLine();
            transcriber.Save(_result, _options.Output);

            Console.WriteLine($"{_result.Events.Count} events, {_result.AbcMeasures} measures in {_clock.Elapsed.TotalSeconds:F1}s -> {Path.GetFullPath(_options.Output)}");
            foreach (var warning in _result.Warnings) Console.WriteLine($"  warning: {warning}");
            if (_result.AbcError != null)
            {
                Console.WriteLine($"ABC unavailable: {_result.AbcError}. MIDI and annotations were saved.");
                return _options.Transcribe.MelodyOnly ? 1 : 0;
            }
        }
        return 0;
    }

    /// <summary>tokens.json of a reference run: [{ "tokens": [...] }, ...] per window.</summary>
    static List<List<int>> _readTokens(string path)
    {
        using (var _doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(path)))
        {
            return _doc.RootElement.EnumerateArray()
                .Select(window => window.GetProperty("tokens").EnumerateArray().Select(t => t.GetInt32()).ToList())
                .ToList();
        }
    }

    static void _report(TranscribeProgress progress)
    {
        if (progress.Stage == "decoding")
            Console.Write($"\r  Window {progress.Window}/{progress.Windows}: {progress.Tokens} tokens   ");
        else if (progress.Stage == "encoding")
            Console.Write($"\r  Window {progress.Window}/{progress.Windows}: encoding audio   ");
        else if (progress.Stage == "window_complete")
            Console.Write($"\r  Window {progress.Window}/{progress.Windows}: {progress.Tokens} tokens done");
    }
}

/// <summary>Command line of the transcriber.</summary>
internal sealed class CliOptions
{
    public string Audio = "";
    public string ModelDir = "";
    public string? MertDir;
    public ComputeBackend Backend = OperatingSystem.IsMacOS() ? ComputeBackend.Metal : ComputeBackend.Cpu;
    public ComputeDType DType = ComputeDType.Auto;
    public string Output = "output";
    public string? Replay;
    public TranscribeOptions Transcribe = new TranscribeOptions();

    public static CliOptions? Parse(string[] args)
    {
        var o = new CliOptions();
        for (int i = 0; i < args.Length; i++)
        {
            string _key = args[i];
            if (!_key.StartsWith("--"))
            {
                if (o.Audio.Length > 0) return null;
                o.Audio = _key;
                continue;
            }
            if (_key == "--melody-only")
            {
                o.Transcribe.MelodyOnly = true;
                continue;
            }
            if (i + 1 >= args.Length) return null;
            string _value = args[++i];
            switch (_key)
            {
                case "--model": o.ModelDir = _value; break;
                case "--mert": o.MertDir = _value; break;
                case "--out": o.Output = _value; break;
                case "--replay": o.Replay = _value; break;
                case "--device":
                    o.Backend = _value.ToLowerInvariant() switch
                    {
                        "cpu" => ComputeBackend.Cpu,
                        "cuda" => ComputeBackend.Cuda,
                        _ => ComputeBackend.Metal
                    };
                    break;
                case "--dtype": o.DType = Enum.Parse<ComputeDType>(_value, true); break;
                case "--max-seconds": o.Transcribe.MaxSeconds = double.Parse(_value, CultureInfo.InvariantCulture); break;
                case "--overlap": o.Transcribe.OverlapSeconds = double.Parse(_value, CultureInfo.InvariantCulture); break;
                case "--lookahead": o.Transcribe.LookaheadSeconds = double.Parse(_value, CultureInfo.InvariantCulture); break;
                case "--prompts": o.Transcribe.Prompts = _value.Split(',', StringSplitOptions.RemoveEmptyEntries); break;
                default: return null;
            }
        }
        if (o.Audio.Length == 0) return null;
        if (o.ModelDir.Length == 0) o.ModelDir = FindModel() ?? "";
        if (o.ModelDir.Length == 0)
        {
            Console.Error.WriteLine("SheetSage2: cannot find the model folder, pass --model");
            return null;
        }
        o.MertDir ??= SheetSage2Engine.FindMert(o.ModelDir);
        return o;
    }

    /// <summary>Walks up from the binary looking for the SheetSage2 checkpoint folder.</summary>
    public static string? FindModel()
    {
        var _directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (_directory != null)
        {
            string _config = Path.Combine(_directory.FullName, "config.json");
            if (File.Exists(_config) && File.ReadAllText(_config).Contains("sheetsage2")) return _directory.FullName;
            _directory = _directory.Parent;
        }
        return null;
    }

    public static void PrintUsage()
    {
        Console.WriteLine("""
            SheetSage2 transcription (Rust/Candle engine, C# pipeline)

              SheetSage2 song.mp3 [options]

              --out <dir>             output folder (default output)
              --model <dir>           SheetSage2 folder (default: found next to the binary)
              --mert <dir>            MERT-v2-FullSong folder (default: Hugging Face cache)
              --device metal|cuda|cpu (default metal on macOS, cuda elsewhere)
              --dtype auto|f32|f16|bf16
              --melody-only           no chords in the ABC score and the playback MIDI
              --max-seconds <s>       transcribe only the first seconds
              --overlap <s>           window overlap (default 200)
              --lookahead <s>         discarded tail of each window (default 100)
              --prompts a,b,c         task prompts (default: the full transcription set)
              --replay <tokens.json>  skip decoding and export a reference run's tokens (parity tool)
            """);
    }
}
