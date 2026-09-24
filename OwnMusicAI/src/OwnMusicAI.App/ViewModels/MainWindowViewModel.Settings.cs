using System.Diagnostics;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SheetSage2;
using OwnMusicAI.App.Services;
using OwnMusicAI.Engine;

namespace OwnMusicAI.App.ViewModels;

/// <summary>
/// Settings page: model folders, device, where the songs go.
/// </summary>
public partial class MainWindowViewModel
{
    [ObservableProperty] private string _yuE2Dir = "";
    [ObservableProperty] private string _vaeDir = "";
    [ObservableProperty] private string _sheetSageDir = "";
    [ObservableProperty] private string _outputDir = "";
    [ObservableProperty] private ComputeBackend _backend;
    [ObservableProperty] private ComputeDType _dType;
    [ObservableProperty] private bool _keepModelLoaded;

    public ComputeBackend[] Backends { get; } = Enum.GetValues<ComputeBackend>();
    public ComputeDType[] DTypes { get; } = Enum.GetValues<ComputeDType>();

    /// <summary>Shown as the VAE box's watermark when the HF cache already has it.</summary>
    public string VaeHint => ModelPaths.FindVae() is string found ? $"auto: {found}" : "not in the Hugging Face cache";

    void _loadSettingsFields()
    {
        YuE2Dir = _settings.YuE2Dir;
        VaeDir = _settings.VaeDir ?? "";
        SheetSageDir = _settings.SheetSageDir;
        OutputDir = _settings.OutputDir;
        Backend = _settings.Backend;
        DType = _settings.DType;
        KeepModelLoaded = _settings.KeepModelLoaded;
    }

    [RelayCommand]
    private void SaveSettings()
    {
        bool _newLibrary = OutputDir.Trim() != _settings.OutputDir;
        if (_newLibrary && Songs.Any(s => s.IsBusy)) { _status("Wait for the queue to empty before moving the library", true); return; }

        _settings.YuE2Dir = YuE2Dir.Trim();
        _settings.VaeDir = string.IsNullOrWhiteSpace(VaeDir) ? null : VaeDir.Trim();
        _settings.SheetSageDir = SheetSageDir.Trim();
        _settings.OutputDir = OutputDir.Trim();
        _settings.Backend = Backend;
        _settings.DType = DType;
        _settings.KeepModelLoaded = KeepModelLoaded;
        _settings.Volume = Player.Volume;
        try
        {
            _settings.Save();
        }
        catch (IOException ex)
        {
            _status($"Could not save the settings: {ex.Message}", true);
            return;
        }

        _pipeline.Paths = _settings.ToModelPaths();
        _pipeline.KeepLoaded = KeepModelLoaded;
        _checkModels();
        _refreshCost();
        if (_newLibrary) _loadLibrary();
        _status(ModelProblem ?? "Settings saved", ModelProblem != null);
    }

    [RelayCommand]
    private void RevertSettings() => _loadSettingsFields();

    [RelayCommand]
    private async Task BrowseFolder(string which)
    {
        string? _dir = Picker == null ? null : await Picker.OpenFolderAsync(which == "output" ? "Choose the song folder" : $"Choose the {which} folder");
        if (_dir == null) return;
        switch (which)
        {
            case "YuE2-3B": YuE2Dir = _dir; break;
            case "YuE2-Vae": VaeDir = _dir; break;
            case "SheetSage2": SheetSageDir = _dir; break;
            default: OutputDir = _dir; break;
        }
    }

    [RelayCommand]
    private void OpenLibraryFolder()
    {
        Directory.CreateDirectory(_settings.OutputDir);
        Reveal(_settings.OutputDir);
    }

    [ObservableProperty] private string? _missingModels;
    [ObservableProperty] private bool _isDownloading;
    [ObservableProperty] private double _downloadPercent;
    [ObservableProperty] private string _downloadText = "";

    CancellationTokenSource? _downloadCts;

    /// <summary>
    /// Pulls whatever ModelDownloader.Missing() lists. Empty model folders get a default under
    /// app data, a stopped run goes on from where it was.
    /// </summary>
    [RelayCommand]
    private async Task DownloadModels()
    {
        string _home = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "OwnMusicAI", "models");
        if (_settings.YuE2Dir.Length == 0) _settings.YuE2Dir = YuE2Dir = Path.Combine(_home, "YuE2-3B");
        if (_settings.SheetSageDir.Length == 0) _settings.SheetSageDir = SheetSageDir = Path.Combine(_home, "SheetSage2");

        _downloadCts = new CancellationTokenSource();
        IsDownloading = true;
        DownloadPercent = 0;
        DownloadText = "Fetching the file list…";
        var _clock = Stopwatch.StartNew();
        long _start = -1;
        var _progress = new Progress<DownloadProgress>(p =>
        {
            if (!IsDownloading) return;
            if (_start < 0) _start = p.Done;
            double _mbps = (p.Done - _start) / 1e6 / Math.Max(_clock.Elapsed.TotalSeconds, 0.1);
            DownloadPercent = p.Total > 0 ? 100.0 * p.Done / p.Total : 0;
            DownloadText = $"{p.File} · {p.Done / 1e9:0.00} / {p.Total / 1e9:0.00} GB · {_mbps:0} MB/s";
        });
        try
        {
            await ModelDownloader.DownloadAsync(_settings.ToModelPaths(), _progress, _downloadCts.Token);
            DownloadText = "";
            _status("Models downloaded");
        }
        catch (OperationCanceledException)
        {
            DownloadText = "Canceled, press the button again to resume from here";
        }
        catch (Exception ex)
        {
            DownloadText = "";
            _status($"Download failed: {ex.Message}", true);
        }
        finally
        {
            IsDownloading = false;
            _downloadCts.Dispose();
            _downloadCts = null;
            _pipeline.Paths = _settings.ToModelPaths();
            _checkModels();
            OnPropertyChanged(nameof(VaeHint));
        }
    }

    [RelayCommand]
    private void CancelDownload() => _downloadCts?.Cancel();
}
