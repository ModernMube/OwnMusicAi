using System.Collections.ObjectModel;
using System.Diagnostics;
using CommunityToolkit.Mvvm.ComponentModel;
using OwnMusicAI.App.Services;
using OwnMusicAI.Engine;

namespace OwnMusicAI.App.ViewModels;

/// <summary>
/// What the view needs from the OS: file and folder pickers. MainWindow implements it.
/// </summary>
public interface IFilePicker
{
    /// <summary>startFolder = where the dialog opens, null lets the OS pick.</summary>
    Task<string?> OpenFileAsync(string title, string kind, string? startFolder, params string[] patterns);
    Task<string?> OpenFolderAsync(string title);
}

/// <summary>
/// The whole studio: create form (Simple / Advanced), settings, the song queue + library and the
/// player bar. Split by task into the .Create, .Queue and .Settings partials.
/// </summary>
public partial class MainWindowViewModel : ObservableObject, IDisposable
{
    readonly AppSettings _settings;
    readonly SongPipeline _pipeline;

    [ObservableProperty] private bool _isSimplePage = true;
    [ObservableProperty] private bool _isAdvancedPage;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsCreatePage))]
    private bool _isSettingsPage;

    [ObservableProperty] private string? _statusMessage;
    [ObservableProperty] private bool _statusIsError;
    [ObservableProperty] private string? _modelProblem;

    public MainWindowViewModel()
    {
        _settings = AppSettings.Load();
        _pipeline = new SongPipeline(_settings.ToModelPaths()) { KeepLoaded = _settings.KeepModelLoaded };
        Player.Volume = _settings.Volume;

        _loadSettingsFields();
        _checkModels();
        if (MissingModels != null) IsSettingsPage = true;
        _loadLibrary();
        _resetParameters();
        _watchCost();
        _ = _startAudioAsync();
    }

    public PlayerViewModel Player { get; } = new PlayerViewModel();

    public ObservableCollection<SongItemViewModel> Songs { get; } = new ObservableCollection<SongItemViewModel>();

    public IFilePicker? Picker { get; set; }

    public bool IsCreatePage => !IsSettingsPage;

    partial void OnIsSimplePageChanged(bool value) { if (value) { IsAdvancedPage = false; IsSettingsPage = false; } }
    partial void OnIsAdvancedPageChanged(bool value) { if (value) { IsSimplePage = false; IsSettingsPage = false; } }
    partial void OnIsSettingsPageChanged(bool value) { if (value) { IsSimplePage = false; IsAdvancedPage = false; } }

    async Task _startAudioAsync()
    {
        try
        {
            await AudioEngineService.Instance.InitializeAsync();
            Player.Volume = _settings.Volume;
        }
        catch (Exception ex)
        {
            _status($"The audio engine did not start: {ex.Message}", true);
        }
    }

    void _checkModels()
    {
        var _paths = _settings.ToModelPaths();
        ModelProblem = _paths.SongProblem();
        AnalysisProblem = _paths.AnalysisProblem();
        var _missing = ModelDownloader.Missing(_paths);
        MissingModels = _missing.Count == 0 ? null : string.Join(", ", _missing);
    }

    void _status(string text, bool error = false)
    {
        StatusMessage = text;
        StatusIsError = error;
    }

    /// <summary>Finder / Explorer on a folder.</summary>
    public static void Reveal(string folder)
    {
        if (Directory.Exists(folder)) Process.Start(new ProcessStartInfo(folder) { UseShellExecute = true });
    }

    public void Dispose()
    {
        foreach (var s in Songs) s.Cts?.Cancel();
        _downloadCts?.Cancel();
        SaveLyricsDraft();
        _settings.Volume = Player.Volume;
        _settings.Weirdness = Weirdness;
        _settings.StyleInfluence = StyleInfluence;
        _settings.Variety = Variety;
        try { _settings.Save(); } catch (IOException) { }
        Player.Dispose();
        _pipeline.Dispose();
        AudioEngineService.Instance.Dispose();
    }
}
