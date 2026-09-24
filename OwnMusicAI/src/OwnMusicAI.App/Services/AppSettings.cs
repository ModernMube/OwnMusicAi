using System.Text.Json;
using System.Text.Json.Serialization;
using SheetSage2;
using OwnMusicAI.Engine;

namespace OwnMusicAI.App.Services;

/// <summary>
/// Model folders, device and output folder. Lives in the user's app data as settings.json.
/// </summary>
public sealed class AppSettings
{
    static readonly JsonSerializerOptions _json = new JsonSerializerOptions
    {
        WriteIndented = true,
        Converters = { new JsonStringEnumConverter() }
    };

    public string YuE2Dir { get; set; } = "";
    public string? VaeDir { get; set; }
    public string SheetSageDir { get; set; } = "";
    public ComputeBackend Backend { get; set; } = OperatingSystem.IsMacOS() ? ComputeBackend.Metal : ComputeBackend.Cpu;
    public ComputeDType DType { get; set; } = ComputeDType.Auto;
    public string OutputDir { get; set; } = "";
    public bool KeepModelLoaded { get; set; } = true;
    public double Volume { get; set; } = 90;

    /// <summary>The create sliders, 0..100.</summary>
    public double Weirdness { get; set; } = 50;
    public double StyleInfluence { get; set; }
    public double Variety { get; set; } = 50;

    static string _file => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "OwnMusicAI", "settings.json");

    public static JsonSerializerOptions Json => _json;

    /// <summary>
    /// Saved file if there is one, holes filled from what Discover() finds.
    /// </summary>
    public static AppSettings Load()
    {
        AppSettings? _settings = null;
        try
        {
            if (File.Exists(_file)) _settings = JsonSerializer.Deserialize<AppSettings>(File.ReadAllText(_file), _json);
        }
        catch (JsonException) { }
        _settings ??= new AppSettings();

        var _found = ModelPaths.Discover();
        if (_settings.YuE2Dir.Length == 0) _settings.YuE2Dir = _found.YuE2Dir;
        if (_settings.SheetSageDir.Length == 0) _settings.SheetSageDir = _found.SheetSageDir;
        if(_settings.OutputDir.Length == 0)
            _settings.OutputDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyMusic), "OwnMusicLocal");
        return _settings;
    }

    public void Save()
    {
        Directory.CreateDirectory(Path.GetDirectoryName(_file)!);
        File.WriteAllText(_file, JsonSerializer.Serialize(this, _json));
    }

    public ModelPaths ToModelPaths() => new ModelPaths
    {
        YuE2Dir = YuE2Dir,
        VaeDir = string.IsNullOrWhiteSpace(VaeDir) ? null : VaeDir,
        SheetSageDir = SheetSageDir,
        Backend = Backend,
        DType = DType
    };
}
