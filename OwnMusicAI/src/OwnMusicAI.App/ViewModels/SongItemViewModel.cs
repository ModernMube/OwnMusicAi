using System.Globalization;
using Avalonia;
using Avalonia.Media;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using OwnMusicAI.App.Services;
using OwnMusicAI.Engine;

namespace OwnMusicAI.App.ViewModels;

/// <summary>
/// One card in the library: a song folder, its job and the live progress while it's cooking.
/// The actions just hand themselves back to the main view model.
/// </summary>
public partial class SongItemViewModel : ObservableObject
{
    readonly MainWindowViewModel _owner;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsDone), nameof(IsBusy), nameof(CanResume), nameof(StatusText), nameof(HasError))]
    private SongStatus _status;

    [ObservableProperty] private string _stageText = "";
    [ObservableProperty] private double _progress;
    [ObservableProperty] private bool _isIndeterminate;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasError))]
    private string? _error;

    [ObservableProperty] private string _durationText = "";
    [ObservableProperty] private bool _pendingDelete;

    public SongItemViewModel(MainWindowViewModel owner, string folder, SongJob job)
    {
        _owner = owner;
        Folder = folder;
        Job = job;
        Cover = _cover(folder);
        Refresh();
    }

    public string Folder { get; }
    public SongJob Job { get; }
    public IBrush Cover { get; }

    /// <summary>Set while the queue runs this song.</summary>
    public CancellationTokenSource? Cts { get; set; }

    public string WavPath => Path.Combine(Folder, SongLibrary.WavFile);

    public string Title
    {
        get
        {
            if (Job.Request.Title.Length > 0) return Job.Request.Title;
            var _words = Job.Request.Style.Split(new[] { ',', ' ' }, StringSplitOptions.RemoveEmptyEntries);
            return _words.Length == 0 ? "Untitled" : string.Join(' ', _words.Take(3));
        }
    }

    public string Style => Job.Request.Style;

    public string Meta
    {
        get
        {
            var r = Job.Request;
            string _mode = r.Mode switch { ScoreMode.Off => "no score", ScoreMode.Melody => "melody score", _ => "full score" };
            string _from = r.ReferenceAudio != null ? $" · cover of: {Path.GetFileNameWithoutExtension(r.ReferenceAudio)}" : r.Abc != null ? " · custom score" : "";
            return $"{Job.Created.ToString("MMM d, HH:mm", CultureInfo.GetCultureInfo("en-US"))} · seed {r.Seed} · {_mode}{_from}";
        }
    }

    public bool IsDone => Status == SongStatus.Done;
    public bool IsBusy => Status is SongStatus.Queued or SongStatus.Running;
    public bool CanResume => Status is SongStatus.Failed or SongStatus.Canceled;
    public bool HasError => !string.IsNullOrEmpty(Error) && CanResume;

    public string StatusText => Status switch
    {
        SongStatus.Queued => "Queued",
        SongStatus.Running => "Running",
        SongStatus.Failed => "Failed",
        SongStatus.Canceled => "Stopped",
        _ => ""
    };

    /// <summary>
    /// Pulls status/error/length back out of the job after the queue touched it.
    /// </summary>
    public void Refresh()
    {
        Status = Job.Status;
        Error = Job.Error;
        DurationText = Job.DurationSeconds > 0 ? PlayerViewModel.Format(Job.DurationSeconds) : "";
        if (Status == SongStatus.Queued) { StageText = "Waiting in the queue"; Progress = 0; IsIndeterminate = false; }
        if (Status == SongStatus.Canceled) StageText = "Stopped - resume picks up from the last finished phase";
    }

    public void Report(GenerationProgress p)
    {
        if (Status != SongStatus.Running) return;
        string _stage = p.Stage switch
        {
            GenerationStage.Waiting => "Waiting for the engine",
            GenerationStage.Transcribing => "Reading the reference song",
            GenerationStage.LoadingModel => "Loading YuE2",
            GenerationStage.PlanningScore => "Writing the score",
            GenerationStage.GeneratingMusic => "Composing",
            GenerationStage.SynthesizingAudio => "Synthesizing audio",
            GenerationStage.DecodingAudio => "Decoding",
            _ => "Finishing"
        };
        StageText = p.Detail.Length > 0 ? $"{_stage} · {p.Detail}" : _stage;
        IsIndeterminate = p.Stage is GenerationStage.Waiting or GenerationStage.LoadingModel;
        Progress = p.Fraction * 100;
    }

    [RelayCommand] private Task Play() => _owner.PlaySongAsync(this);
    [RelayCommand]
    private void Cancel()
    {
        if (Cts != null) Cts.Cancel();
        else _owner.Unqueue(this);
    }
    [RelayCommand] private void Resume() => _owner.ResumeSong(this);
    [RelayCommand] private void Reuse() => _owner.ReuseSong(this);
    [RelayCommand] private void UseAsReference() => _owner.UseAsReference(this);
    [RelayCommand] private void OpenFolder() => MainWindowViewModel.Reveal(Folder);

    /// <summary>First click arms it, second one deletes the folder.</summary>
    [RelayCommand]
    private async Task Delete()
    {
        if (!PendingDelete) { PendingDelete = true; return; }
        await _owner.DeleteSongAsync(this);
    }

    //Card cover: a gradient seeded from the folder name
    static IBrush _cover(string folder)
    {
        int _hash = 17;
        foreach (char c in Path.GetFileName(folder)) _hash = unchecked(_hash * 31 + c);
        double _hue = (uint)_hash % 360;
        var _a = HsvColor.ToRgb(_hue, 0.7, 0.95);
        var _b = HsvColor.ToRgb((_hue + 70) % 360, 0.8, 0.55);
        return new LinearGradientBrush
        {
            StartPoint = new RelativePoint(0, 0, RelativeUnit.Relative),
            EndPoint = new RelativePoint(1, 1, RelativeUnit.Relative),
            GradientStops = { new GradientStop(_a, 0), new GradientStop(_b, 1) }
        };
    }
}
