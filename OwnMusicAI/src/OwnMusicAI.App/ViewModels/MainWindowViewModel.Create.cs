using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using OwnaudioNET.Sources;
using OwnMusicAI.App.Services;
using OwnMusicAI.Engine;

namespace OwnMusicAI.App.ViewModels;

public sealed record LengthOption(string Label, double? Seconds)
{
    public override string ToString() => Label;
}

public sealed record ModeOption(string Label, ScoreMode Mode)
{
    public override string ToString() => Label;
}

/// <summary>
/// The create side. Style, lyrics and title are shared by both pages; Simple adds a length pick,
/// Advanced the reference song, the score and every knob of the pipeline.
/// </summary>
public partial class MainWindowViewModel
{
    const int ReferencePeaks = 20_000;

    #region Prompt (both pages)

    [ObservableProperty] private string _songTitle = "";
    [ObservableProperty] private string _style = "";
    [ObservableProperty] private string _lyrics = "";

    public string[] StyleTags { get; } =
    {
        "pop", "rock", "city pop", "lofi", "hip hop", "edm", "jazz", "folk", "r&b", "ballad", "metal", "cinematic",
        "female vocal", "male vocal", "acoustic guitar", "piano", "synth", "groovy bass",
        "upbeat", "melancholic", "energetic", "dreamy"
    };

    public string[] SectionTags { get; } = { "[Intro]", "[Verse]", "[Pre-Chorus]", "[Chorus]", "[Bridge]", "[Interlude]", "[Outro]" };

    [RelayCommand]
    private void AddStyleTag(string tag)
    {
        string _s = Style.TrimEnd().TrimEnd(',');
        Style = _s.Length == 0 ? tag : $"{_s}, {tag}";
    }

    [RelayCommand]
    private void AddSection(string tag)
    {
        string _l = Lyrics.TrimEnd();
        Lyrics = _l.Length == 0 ? $"{tag}\n" : $"{_l}\n\n{tag}\n";
    }

    [RelayCommand]
    private async Task LoadLyrics()
    {
        string? _file = Picker == null ? null : await Picker.OpenFileAsync("Load lyrics", "Text", _lyricsDir, "*.txt", "*.lrc", "*.json");
        if (_file == null) return;
        SaveLyricsDraft();
        _lyricsDraft = null;

        //An examples/*.json of the model repo carries style + lyrics + seed
        if (_file.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
        {
            using (var _doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(_file)))
            {
                var _root = _doc.RootElement;
                if (_root.TryGetProperty("style", out var s)) Style = s.GetString() ?? Style;
                if (_root.TryGetProperty("lyrics", out var l)) Lyrics = l.GetString() ?? Lyrics;
                if (_root.TryGetProperty("title", out var t)) SongTitle = t.GetString() ?? SongTitle;
                if (_root.TryGetProperty("seed", out var sd)) { Seed = sd.GetInt64(); RandomSeed = false; }
            }
            return;
        }
        Lyrics = File.ReadAllText(_file);

        //One of our own drafts: keep editing that file, its name is the title
        if (Path.GetDirectoryName(_file) == _lyricsDir)
        {
            _lyricsDraft = _file;
            _lyricsSaved = Lyrics;
            if (SongTitle.Length == 0 && !Path.GetFileNameWithoutExtension(_file).StartsWith("lyrics-")) SongTitle = Path.GetFileNameWithoutExtension(_file);
        }
    }

    string _lyricsDir => Path.Combine(_settings.OutputDir, SongLibrary.LyricsFolder);

    string? _lyricsDraft;
    string _lyricsSaved = "";

    /// <summary>
    /// Lyrics box lost focus: the text goes to Library/Lyrics/&lt;title&gt;.txt (lyrics-&lt;date&gt;.txt
    /// untitled). Same draft = same file, a new title renames it.
    /// </summary>
    public void SaveLyricsDraft()
    {
        if (string.IsNullOrWhiteSpace(Lyrics) || Lyrics == _lyricsSaved) return;

        string _name = SongLibrary.FileName(SongTitle);
        if (_name.Length == 0) _name = _lyricsDraft != null ? Path.GetFileNameWithoutExtension(_lyricsDraft) : $"lyrics-{DateTime.Now:yyyyMMdd-HHmmss}";
        string _file = Path.Combine(_lyricsDir, _name + ".txt");
        try
        {
            Directory.CreateDirectory(_lyricsDir);
            if (_lyricsDraft != null && _lyricsDraft != _file && File.Exists(_lyricsDraft) && !File.Exists(_file)) File.Move(_lyricsDraft, _file);
            File.WriteAllText(_file, Lyrics);
        }
        catch (IOException ex)
        {
            _status($"Could not save the lyrics: {ex.Message}", true);
            return;
        }
        _lyricsDraft = _file;
        _lyricsSaved = Lyrics;
        _status($"Lyrics saved: Lyrics/{Path.GetFileName(_file)}");
    }

    #endregion

    #region Simple

    public LengthOption[] Lengths { get; } =
    {
        new LengthOption("Full song", null),
        new LengthOption("~30 seconds", 30),
        new LengthOption("~1 minute", 60),
        new LengthOption("~2 minutes", 120),
        new LengthOption("~3 minutes", 180)
    };

    [ObservableProperty] private LengthOption? _selectedLength;

    #endregion

    #region Advanced: reference song

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasReference), nameof(ReferenceName))]
    private string? _referencePath;

    [ObservableProperty] private float[]? _referenceWaveform;
    [ObservableProperty] private string _referenceInfo = "";
    [ObservableProperty] private decimal _analyzeSeconds;
    [ObservableProperty] private bool _isAnalyzing;
    [ObservableProperty] private string _analysisStage = "";
    [ObservableProperty] private double _analysisProgress;
    [ObservableProperty] private bool _hasAnalysis;
    [ObservableProperty] private string _keyText = "–";
    [ObservableProperty] private string _tempoText = "–";
    [ObservableProperty] private string _meterText = "–";
    [ObservableProperty] private string _measuresText = "–";
    [ObservableProperty] private string _chordsText = "";
    [ObservableProperty] private string _sectionsText = "";
    [ObservableProperty] private string? _analysisProblem;

    CancellationTokenSource? _analyzeCts;

    public bool HasReference => ReferencePath != null;
    public string ReferenceName => ReferencePath == null ? "No reference song" : Path.GetFileName(ReferencePath);

    [RelayCommand]
    private async Task PickReference()
    {
        string? _file = Picker == null ? null
            : await Picker.OpenFileAsync("Reference song", "Audio", null, "*.mp3", "*.wav", "*.flac", "*.m4a", "*.aac", "*.ogg", "*.aiff");
        if (_file != null) await SetReferenceAsync(_file);
    }

    public async Task SetReferenceAsync(string file)
    {
        ReferencePath = file;
        ReferenceWaveform = null;
        ReferenceInfo = "reading…";
        IsScoreFromReference = true;
        _clearAnalysis();
        try
        {
            var (_peaks, _seconds) = await Task.Run(() =>
            {
                using (var s = new FileSource(file, 8192, 48000, 2))
                    return (s.GetPeaks(ReferencePeaks), s.Duration);
            });
            if (ReferencePath != file) return;
            ReferenceWaveform = _peaks;
            ReferenceInfo = PlayerViewModel.Format(_seconds);
        }
        catch (Exception ex)
        {
            ReferenceInfo = $"can't decode: {ex.Message}";
        }
    }

    [RelayCommand]
    private void ClearReference()
    {
        _analyzeCts?.Cancel();
        ReferencePath = null;
        ReferenceWaveform = null;
        ReferenceInfo = "";
        _clearAnalysis();
        if (IsScoreFromReference) IsScorePlanned = true;
    }

    [RelayCommand]
    private async Task PreviewReference()
    {
        if (ReferencePath != null) await Player.LoadAsync(ReferencePath, ReferenceName, "reference song");
    }

    /// <summary>
    /// SheetSage2 over the reference: fills the score box and the key / tempo / chord readout.
    /// </summary>
    [RelayCommand]
    private async Task Analyze()
    {
        if (ReferencePath == null) return;
        if (AnalysisProblem != null) { _status(AnalysisProblem, true); return; }

        _analyzeCts = new CancellationTokenSource();
        IsAnalyzing = true;
        AnalysisProgress = 0;
        bool _melody = SelectedMode?.Mode == ScoreMode.Melody;
        var _progress = new Progress<GenerationProgress>(p =>
        {
            if (!IsAnalyzing) return;
            AnalysisStage = p.Stage == GenerationStage.Waiting ? "Waiting for the running song to finish…" : p.Detail;
            AnalysisProgress = p.Fraction * 100;
        });
        try
        {
            double? _max = AnalyzeSeconds > 0 ? (double)AnalyzeSeconds : null;
            var a = await _pipeline.AnalyzeAsync(ReferencePath, _melody, _max, _progress, _analyzeCts.Token);
            _showAnalysis(a, _melody);
        }
        catch (OperationCanceledException)
        {
            AnalysisStage = "Canceled";
        }
        catch (Exception ex)
        {
            AnalysisStage = "";
            _status($"Analysis failed: {ex.Message}", true);
        }
        finally
        {
            IsAnalyzing = false;
            _analyzeCts.Dispose();
            _analyzeCts = null;
        }
    }

    [RelayCommand]
    private void CancelAnalysis() => _analyzeCts?.Cancel();

    void _showAnalysis(ReferenceAnalysis a, bool melody)
    {
        HasAnalysis = true;
        KeyText = a.Key?.Replace(":", " ") ?? "–";
        TempoText = a.Bpm is double bpm ? $"{bpm:F0} BPM" : "–";
        MeterText = a.Meter ?? "–";
        MeasuresText = a.Measures > 0 ? a.Measures.ToString() : "–";

        //Consecutive repeats squashed, a progression reads better than a timeline
        var _chords = new List<string>();
        foreach (var c in a.Chords)
            if (c.Label != "N" && (_chords.Count == 0 || _chords[^1] != c.Label)) _chords.Add(c.Label);
        ChordsText = _chords.Count == 0 ? "–" : string.Join("  ", _chords.Take(48)) + (_chords.Count > 48 ? "  …" : "");

        var _sections = new List<string>();
        foreach (var s in a.Sections)
            if (_sections.Count == 0 || !_sections[^1].StartsWith(s.Label)) _sections.Add($"{s.Label} {PlayerViewModel.Format(s.Start)}");
        SectionsText = _sections.Count == 0 ? "–" : string.Join(" → ", _sections);

        if (a.Abc != null)
        {
            AbcText = a.Abc;
            AnalysisStage = $"Done, {a.ElapsedSeconds:F0} s · {(melody ? "melody-only" : "chord")} score, {a.Measures} bars";
        }
        else
        {
            AnalysisStage = $"Could not build a score: {a.AbcError}";
        }
    }

    void _clearAnalysis()
    {
        HasAnalysis = false;
        AnalysisStage = "";
        AnalysisProgress = 0;
        KeyText = TempoText = MeterText = MeasuresText = "–";
        ChordsText = SectionsText = "";
        if (IsScoreFromReference) AbcText = "";
    }

    #endregion

    #region Advanced: score

    public ModeOption[] Modes { get; } =
    {
        new ModeOption("Full score (melody + chords)", ScoreMode.Full),
        new ModeOption("Melody only (for covers)", ScoreMode.Melody),
        new ModeOption("No score (fastest)", ScoreMode.Off)
    };

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(UsesScore))]
    private ModeOption? _selectedMode;

    [ObservableProperty] private bool _isScorePlanned = true;
    [ObservableProperty] private bool _isScoreFromReference;
    [ObservableProperty] private bool _isScoreCustom;
    [ObservableProperty] private string _abcText = "";

    public bool UsesScore => SelectedMode?.Mode != ScoreMode.Off;

    partial void OnIsScorePlannedChanged(bool value) { if (value) { IsScoreFromReference = false; IsScoreCustom = false; } }
    partial void OnIsScoreFromReferenceChanged(bool value) { if (value) { IsScorePlanned = false; IsScoreCustom = false; } }
    partial void OnIsScoreCustomChanged(bool value) { if (value) { IsScorePlanned = false; IsScoreFromReference = false; } }

    [RelayCommand]
    private async Task LoadAbc()
    {
        string? _file = Picker == null ? null : await Picker.OpenFileAsync("Load ABC score", "ABC", null, "*.abc", "*.txt");
        if (_file == null) return;
        AbcText = File.ReadAllText(_file);
        IsScoreCustom = true;
    }

    #endregion

    #region Advanced: parameters

    [ObservableProperty] private decimal _seed;
    [ObservableProperty] private bool _randomSeed;
    [ObservableProperty] private bool _cfgAuto;
    [ObservableProperty] private decimal _cfg;
    [ObservableProperty] private decimal _maxSeconds;
    [ObservableProperty] private decimal _maxAbcTokens;
    [ObservableProperty] private decimal _odeSteps;
    [ObservableProperty] private decimal _contextTokens;
    [ObservableProperty] private decimal _vaeTile;
    [ObservableProperty] private decimal _musicTemperature;
    [ObservableProperty] private decimal _musicTopP;
    [ObservableProperty] private decimal _musicTopK;
    [ObservableProperty] private decimal _musicRepetition;
    [ObservableProperty] private decimal _scoreTemperature;
    [ObservableProperty] private decimal _scoreTopP;
    [ObservableProperty] private decimal _scoreTopK;
    [ObservableProperty] private decimal _scoreRepetition;

    [RelayCommand]
    private void ResetParameters() => _resetParameters();

    [RelayCommand]
    private void RollSeed() => Seed = _newSeed();

    void _resetParameters()
    {
        var d = new SongRequest();
        SelectedLength ??= Lengths[0];
        SelectedMode = Modes[0];
        Seed = d.Seed;
        RandomSeed = true;
        CfgAuto = true;
        Cfg = 1.5m;
        MaxSeconds = 0;
        MaxAbcTokens = 0;
        OdeSteps = d.OdeSteps;
        ContextTokens = d.Context;
        VaeTile = d.VaeTile;
        _setSampling(SamplingSettings.MusicDefaults, SamplingSettings.ScoreDefaults);
        _setMix(CreativeMix.Default, true);
    }

    void _setSampling(SamplingSettings music, SamplingSettings score)
    {
        MusicTemperature = (decimal)music.Temperature;
        MusicTopP = (decimal)music.TopP;
        MusicTopK = music.TopK;
        MusicRepetition = (decimal)music.RepetitionPenalty;
        ScoreTemperature = (decimal)score.Temperature;
        ScoreTopP = (decimal)score.TopP;
        ScoreTopK = score.TopK;
        ScoreRepetition = (decimal)score.RepetitionPenalty;
    }

    static long _newSeed() => Random.Shared.NextInt64(1, 1_000_000_000);

    #endregion

    #region Create

    [RelayCommand]
    private void Create()
    {
        if (ModelProblem != null) { _status(ModelProblem, true); return; }
        if (string.IsNullOrWhiteSpace(Style)) { _status("Describe the style first (genre, instruments, mood, voice)", true); return; }
        if (string.IsNullOrWhiteSpace(Lyrics)) { _status("YuE2 needs lyrics - even just section tags, e.g. [Verse] / [Chorus]", true); return; }

        var r = IsAdvancedPage ? _advancedRequest() : _simpleRequest();
        if (r == null) return;
        Enqueue(r);
        SaveLyricsDraft();
        _lyricsDraft = null;

        string _queued = $"'{(r.Title.Length > 0 ? r.Title : "New song")}' queued (seed {r.Seed})";
        var _cost = SongCost.Estimate(r, _settings.Backend, _settings.DType);
        if (_cost.Level == CostLevel.Light) _status(_queued);
        else _status($"{_queued}. Heads up: {_cost.Warnings[0]}", _cost.Level == CostLevel.TooMuch);
    }

    SongRequest _simpleRequest()
    {
        var r = new SongRequest
        {
            Title = SongTitle.Trim(),
            Style = Style.Trim(),
            Lyrics = Lyrics.Trim(),
            Seed = _newSeed(),
            Mode = ScoreMode.Full,
            MaxSeconds = SelectedLength?.Seconds
        };
        Mix.ApplyTo(r);
        return r;
    }

    SongRequest? _advancedRequest()
    {
        if (RandomSeed) Seed = _newSeed();
        var r = _knobRequest();
        if (r.Mode == ScoreMode.Off) return r;

        if (IsScoreFromReference)
        {
            if (ReferencePath == null) { _status("Pick a reference song, or let YuE2 write the score", true); return null; }
            if (AnalysisProblem != null) { _status(AnalysisProblem, true); return null; }
            r.ReferenceAudio = ReferencePath;
            //Empty box = the queue transcribes it right before the song
            r.Abc = string.IsNullOrWhiteSpace(AbcText) ? null : AbcText;
        }
        else if (IsScoreCustom)
        {
            if (string.IsNullOrWhiteSpace(AbcText)) { _status("The score box is empty", true); return null; }
            r.Abc = AbcText;
        }
        return r;
    }

    /// <summary>The Advanced form as is, no checks - the cost meter reads it too.</summary>
    SongRequest _knobRequest()
    {
        return new SongRequest
        {
            Title = SongTitle.Trim(),
            Style = Style.Trim(),
            Lyrics = Lyrics.Trim(),
            Seed = (long)Seed,
            Mode = SelectedMode?.Mode ?? ScoreMode.Full,
            Cfg = CfgAuto ? null : (double)Cfg,
            MaxSeconds = MaxSeconds > 0 ? (double)MaxSeconds : null,
            MaxAbcTokens = MaxAbcTokens > 0 ? (int)MaxAbcTokens : null,
            OdeSteps = (int)OdeSteps,
            Context = (int)ContextTokens,
            VaeTile = (int)VaeTile,
            MusicSampling = new SamplingSettings((double)MusicTemperature, (double)MusicTopP, (int)MusicTopK, (double)MusicRepetition),
            ScoreSampling = new SamplingSettings((double)ScoreTemperature, (double)ScoreTopP, (int)ScoreTopK, (double)ScoreRepetition)
        };
    }

    /// <summary>
    /// A library song's recipe back into the Advanced form, ready to tweak and run again.
    /// </summary>
    public void ReuseSong(SongItemViewModel item)
    {
        var r = item.Job.Request;
        SaveLyricsDraft();
        _lyricsDraft = null;
        _lyricsSaved = r.Lyrics;
        SongTitle = r.Title;
        Style = r.Style;
        Lyrics = r.Lyrics;
        Seed = r.Seed;
        RandomSeed = false;
        _setMix(CreativeMix.From(r), true);
        SelectedMode = Modes.First(m => m.Mode == r.Mode);
        CfgAuto = r.Cfg == null;
        if (r.Cfg is double cfg) Cfg = (decimal)cfg;
        MaxSeconds = (decimal)(r.MaxSeconds ?? 0);
        MaxAbcTokens = r.MaxAbcTokens ?? 0;
        OdeSteps = r.OdeSteps;
        ContextTokens = r.Context;
        VaeTile = r.VaeTile;
        _setSampling(r.MusicSampling ?? SamplingSettings.MusicDefaults, r.ScoreSampling ?? SamplingSettings.ScoreDefaults);

        //The score it actually used: a planned one sits in song.abc
        string _planned = Path.Combine(item.Folder, "song.abc");
        if (r.ReferenceAudio != null && File.Exists(r.ReferenceAudio))
        {
            _ = SetReferenceAsync(r.ReferenceAudio);
            AbcText = r.Abc ?? "";
            IsScoreFromReference = true;
        }
        else if (r.Abc != null) { AbcText = r.Abc; IsScoreCustom = true; }
        else if (File.Exists(_planned)) { AbcText = File.ReadAllText(_planned); IsScorePlanned = true; }
        else { AbcText = ""; IsScorePlanned = true; }

        IsAdvancedPage = true;
        _status($"Recipe of '{item.Title}' loaded. For the exact same score pick 'Custom score'.");
    }

    /// <summary>A finished song becomes the reference of a cover.</summary>
    public void UseAsReference(SongItemViewModel item)
    {
        if (!File.Exists(item.WavPath)) return;
        _ = SetReferenceAsync(item.WavPath);
        IsAdvancedPage = true;
        _status($"'{item.Title}' is now the reference - analyze it, then give it a new style or lyrics");
    }

    #endregion
}
