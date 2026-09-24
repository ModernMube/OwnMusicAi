using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using OwnMusicAI.Engine;

namespace OwnMusicAI.App.ViewModels;

/// <summary>
/// Weirdness / Style influence / Variety sliders. Moving one rewrites the Advanced knobs behind it,
/// and every change re-checks what the song would cost in memory and time.
/// </summary>
public partial class MainWindowViewModel
{
    static readonly HashSet<string> _costInputs = new HashSet<string>
    {
        nameof(Style), nameof(Lyrics), nameof(SelectedLength), nameof(IsAdvancedPage), nameof(IsSimplePage), nameof(SelectedMode),
        nameof(IsScorePlanned), nameof(AbcText), nameof(CfgAuto), nameof(Cfg), nameof(MaxSeconds), nameof(MaxAbcTokens),
        nameof(OdeSteps), nameof(ContextTokens), nameof(VaeTile)
    };

    bool _mixSilent;

    [ObservableProperty] private double _weirdness = 50;
    [ObservableProperty] private double _styleInfluence;
    [ObservableProperty] private double _variety = 50;

    [ObservableProperty] private string _weirdnessHint = "";
    [ObservableProperty] private string _styleHint = "";
    [ObservableProperty] private string _varietyHint = "";

    [ObservableProperty] private string _costSummary = "";
    [ObservableProperty] private string[] _costWarnings = Array.Empty<string>();
    [ObservableProperty] private bool _costIsLight = true;
    [ObservableProperty] private bool _costIsHeavy;
    [ObservableProperty] private bool _costIsTooMuch;

    public CreativeMix Mix => new CreativeMix(Weirdness, StyleInfluence, Variety);

    partial void OnWeirdnessChanged(double value) => _applyMix();
    partial void OnStyleInfluenceChanged(double value) => _applyMix();
    partial void OnVarietyChanged(double value) => _applyMix();

    [RelayCommand]
    private void ResetMix() => _setMix(CreativeMix.Default, false);

    void _watchCost()
    {
        PropertyChanged += (s, e) => { if (e.PropertyName != null && _costInputs.Contains(e.PropertyName)) _refreshCost(); };
        _setMix(new CreativeMix(_settings.Weirdness, _settings.StyleInfluence, _settings.Variety), false);
    }

    /// <summary>silent = only the sliders move, the knobs keep what they have (library reuse).</summary>
    void _setMix(CreativeMix m, bool silent)
    {
        _mixSilent = true;
        Weirdness = m.Weirdness;
        StyleInfluence = m.StyleInfluence;
        Variety = m.Variety;
        _mixSilent = silent;
        _applyMix();
        _mixSilent = false;
    }

    void _applyMix()
    {
        var m = Mix;
        var _music = m.Music;
        double? _cfg = m.Cfg(SelectedMode?.Mode ?? ScoreMode.Full);

        WeirdnessHint = $"temperature {_music.Temperature:0.00} · top-p {_music.TopP:0.00} · top-k {_music.TopK}";
        StyleHint = _cfg is double c ? $"CFG {c:0.00} · a second pass per music token" : "auto CFG · follows the prompt loosely, fastest";
        VarietyHint = $"repetition penalty {_music.RepetitionPenalty:0.000} (score: {m.Score.RepetitionPenalty:0.000})";

        if (!_mixSilent)
        {
            _setSampling(_music, m.Score);
            CfgAuto = _cfg == null;
            if (_cfg is double cfg) Cfg = (decimal)cfg;
        }
        _refreshCost();
    }

    void _refreshCost()
    {
        //ctor order: the knobs are not there yet on the first slider set
        if (SelectedMode == null) return;

        var r = IsAdvancedPage ? _knobRequest() : _simpleRequest();
        if (IsAdvancedPage && !IsScorePlanned && !string.IsNullOrWhiteSpace(AbcText)) r.Abc = AbcText;
        var _cost = SongCost.Estimate(r, _settings.Backend, _settings.DType);
        CostSummary = $"Estimate: {_cost.Summary}";
        CostWarnings = _cost.Warnings.ToArray();
        CostIsTooMuch = _cost.Level == CostLevel.TooMuch;
        CostIsHeavy = _cost.Level == CostLevel.Heavy;
        CostIsLight = _cost.Level == CostLevel.Light;
    }
}
