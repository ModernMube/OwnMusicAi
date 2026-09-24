using System.Globalization;
using System.Text.Json;

namespace YuE2.Song;

/// <summary>
/// Command line. Example json = examples/*.json of the model repo (style, lyrics, seed).
/// </summary>
internal sealed class SongOptions
{
    public string ModelDir = "..";
    public string? VaeDir;
    public EngineBackend Backend = OperatingSystem.IsMacOS() ? EngineBackend.Metal : EngineBackend.Cpu;
    public EngineDType DType = EngineDType.Auto;
    public string Style = "";
    public string Lyrics = "";
    public string Cot = "full";
    public string? Abc;
    public string? AbcAudio;
    public string SheetSageDir = "";
    public long Seed = 831001;
    public double? Cfg;
    public int? MaxAbcTokens;
    public double? MaxSeconds;
    public int OdeSteps = Protocol.OdeSteps;
    public int Context = Protocol.Context;
    public int VaeTile = 256;
    public string? Resume;
    public string Output = "song.wav";

    public static SongOptions? Parse(string[] args)
    {
        var o = new SongOptions();
        for (int i = 0; i < args.Length; i++)
        {
            string _key = args[i];
            if (i + 1 >= args.Length) return null;
            string _val = args[++i];
            switch (_key)
            {
                case "--model": o.ModelDir = _val; break;
                case "--vae": o.VaeDir = _val; break;
                case "--device":
                    o.Backend = _val.ToLowerInvariant() switch
                    {
                        "cpu" => EngineBackend.Cpu,
                        "cuda" => EngineBackend.Cuda,
                        _ => EngineBackend.Metal
                    };
                    break;
                case "--dtype": o.DType = Enum.Parse<EngineDType>(_val, true); break;
                case "--example":
                    using (var _doc = JsonDocument.Parse(File.ReadAllText(_val)))
                    {
                        o.Style = _doc.RootElement.GetProperty("style").GetString()!;
                        o.Lyrics = _doc.RootElement.GetProperty("lyrics").GetString()!;
                        if (_doc.RootElement.TryGetProperty("seed", out var _seed)) o.Seed = _seed.GetInt64();
                    }
                    break;
                case "--style": o.Style = _val; break;
                case "--lyrics": o.Lyrics = File.Exists(_val) ? File.ReadAllText(_val) : _val.Replace("\\n", "\n"); break;
                case "--cot": o.Cot = _val; break;
                case "--abc": o.Abc = File.ReadAllText(_val); break;
                case "--abc-audio": o.AbcAudio = _val; break;
                case "--sheetsage": o.SheetSageDir = _val; break;
                case "--seed": o.Seed = long.Parse(_val); break;
                case "--cfg": o.Cfg = double.Parse(_val, CultureInfo.InvariantCulture); break;
                case "--max-abc-tokens": o.MaxAbcTokens = int.Parse(_val); break;
                case "--max-seconds": o.MaxSeconds = double.Parse(_val, CultureInfo.InvariantCulture); break;
                case "--ode-steps": o.OdeSteps = int.Parse(_val); break;
                case "--context": o.Context = int.Parse(_val); break;
                case "--vae-tile": o.VaeTile = int.Parse(_val); break;
                case "--resume": o.Resume = _val; break;
                case "--out": o.Output = _val; break;
                default: return null;
            }
        }
        if (o.Resume == null && (o.Style.Length == 0 || o.Lyrics.Length == 0)) return null;
        if (o.Cot is not ("off" or "melody" or "full")) return null;
        if (o.AbcAudio != null)
        {
            if (o.Cot == "off") return null;
            if (o.SheetSageDir.Length == 0) o.SheetSageDir = Path.GetFullPath(Path.Combine(o.ModelDir, "..", "SheetSage2"));
            if (!File.Exists(Path.Combine(o.SheetSageDir, "config.json"))) return null;
        }
        o.VaeDir ??= _findVae();
        return o;
    }

    /// <summary>Python keeps the seed as int64, System.Random only takes an int.</summary>
    public int RngSeed => unchecked((int)(Seed ^ (Seed >> 32)));

    //We look for m-a-p/YuE2-Vae in the Hugging Face cache
    static string? _findVae()
    {
        string _home = Environment.GetEnvironmentVariable("HF_HOME")
            ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface");
        string _snapshots = Path.Combine(_home, "hub", "models--m-a-p--YuE2-Vae", "snapshots");
        if (!Directory.Exists(_snapshots)) return null;
        return Directory.GetDirectories(_snapshots).FirstOrDefault(d => File.Exists(Path.Combine(d, "model.safetensors")));
    }

    public static void PrintUsage()
    {
        Console.WriteLine("""
            YuE2 song generator (Rust/Candle engine, C# pipeline)

              dotnet run -c Release -- --example ../examples/tonight-awake.json [options]
              dotnet run -c Release -- --style "funk, female vocal" --lyrics lyrics.txt [options]
              dotnet run -c Release -- --resume song.state.json

              --model <dir>           YuE2-3B folder (default ..)
              --vae <dir>             YuE2-Vae folder (default: Hugging Face cache)
              --device metal|cuda|cpu (default metal on macOS, cuda elsewhere)
              --dtype auto|f32|f16|bf16
              --cot off|melody|full   score planning mode (default full)
              --abc <file>            use this ABC score instead of planning one
              --abc-audio <file>      transcribe this recording with SheetSage2 and use its score
              --sheetsage <dir>       SheetSage2 folder (default ../SheetSage2 next to the model)
              --seed <n>              default 831001 or the example's seed
              --cfg <scale>           text guidance (default 1.01 for off, 1.0 otherwise)
              --max-abc-tokens <n>    cap the score length
              --max-seconds <s>       cap the song length (25 codec tokens = 1 s)
              --ode-steps <n>         flow-matching steps (default 32)
              --context <n>           acoustic chunk context (default 24576)
              --vae-tile <frames>     VAE tile core (default 256)
              --resume <state.json>   continue a stopped run from its saved state
              --out <file.wav>        default song.wav
            """);
    }
}
