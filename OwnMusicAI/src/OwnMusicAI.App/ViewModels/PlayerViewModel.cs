using System.Diagnostics;
using System.Globalization;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Material.Icons;
using OwnaudioNET;
using OwnaudioNET.Sources;
using OwnMusicAI.App.Services;

namespace OwnMusicAI.App.ViewModels;

/// <summary>
/// The bottom player bar: one FileSource on the OwnAudio mixer, the OwnBackingPlayer transport
/// (prepared add, clock seek, interpolated position) cut down to a single track.
/// </summary>
public partial class PlayerViewModel : ObservableObject, IDisposable
{
    const int WaveformPoints = 100_000;

    readonly AudioEngineService _audio = AudioEngineService.Instance;
    readonly SemaphoreSlim _transportLock = new SemaphoreSlim(1, 1);
    readonly Stopwatch _interpWatch = Stopwatch.StartNew();

    //~30 fps and only while playing
    readonly DispatcherTimer _positionTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(33) };

    FileSource? _source;
    double _lastEnginePos, _lastEnginePosAt;
    int _lastShownSecond = -1;

    [ObservableProperty] private string _title = "Nothing playing";
    [ObservableProperty] private string? _subtitle;
    [ObservableProperty] private float[]? _waveformData;
    [ObservableProperty] private double _duration;
    [ObservableProperty] private double _position;
    [ObservableProperty] private double _normalizedPosition;
    [ObservableProperty] private string _positionText = "0:00 / 0:00";
    [ObservableProperty] private bool _hasTrack;
    [ObservableProperty] private double _volume = 90;
    [ObservableProperty] private string? _error;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(PlayIcon))]
    private bool _isPlaying;

    [ObservableProperty] private bool _isPaused;

    public PlayerViewModel()
    {
        _positionTimer.Tick += (_, _) => _tick();
        _audio.PlaybackEnded += (_, _) => Dispatcher.UIThread.Post(() => { if (IsPlaying) StopCommand.Execute(null); });
    }

    public string? CurrentPath { get; private set; }

    public MaterialIconKind PlayIcon => IsPlaying ? MaterialIconKind.Pause : MaterialIconKind.Play;

    /// <summary>
    /// Swaps the loaded file, peaks decoded off the UI thread, and rolls when autoPlay.
    /// </summary>
    public async Task LoadAsync(string path, string title, string? subtitle, bool autoPlay = true)
    {
        try { await _audio.InitializeAsync(); }
        catch (Exception ex) { Error = $"The audio engine is not running: {ex.Message}"; return; }
        await _stop();
        _unload();

        int _rate = OwnaudioNet.Engine?.Config.SampleRate ?? 48000;
        int _channels = OwnaudioNet.Engine?.Config.Channels ?? 2;
        float[] _peaks = Array.Empty<float>();
        try
        {
            _source = await Task.Run(() =>
            {
                var s = new FileSource(path, 8192, _rate, _channels) { DecodeChannels = _channels };
                _peaks = s.GetPeaks(WaveformPoints);
                return s;
            });
        }
        catch (Exception ex)
        {
            Error = $"Can't open {Path.GetFileName(path)}: {ex.Message}";
            return;
        }

        CurrentPath = path;
        Title = title;
        Subtitle = subtitle;
        Duration = _source.Duration;
        WaveformData = _peaks;
        HasTrack = true;
        Error = null;
        _setPosition(0);
        if (autoPlay) await _play();
    }

    [RelayCommand]
    private async Task PlayPause()
    {
        if (IsPlaying) await _pause();
        else await _play();
    }

    [RelayCommand]
    private Task Stop() => _stop();

    /// <summary>
    /// From the waveform behavior: seconds as an invariant string.
    /// </summary>
    [RelayCommand]
    private void Seek(object? raw)
    {
        if (!double.TryParse(raw?.ToString(), NumberStyles.Any, CultureInfo.InvariantCulture, out double _sec)) return;
        _sec = Math.Clamp(_sec, 0, Duration);
        if (IsPlaying || IsPaused) _audio.Mixer?.Seek(_sec);
        _setPosition(_sec);
    }

    partial void OnVolumeChanged(double value)
    {
        if (_audio.Mixer != null) _audio.Mixer.MasterVolume = (float)(value / 100.0);
    }

    async Task _play()
    {
        var mixer = _audio.Mixer;
        if (mixer == null || _source == null) return;

        double _start = Position >= Duration - 0.2 ? 0 : Position;
        bool _wasPaused = IsPaused;
        IsPlaying = true;
        IsPaused = false;

        await _transportLock.WaitAsync();
        try
        {
            var src = _source;
            await Task.Run(() =>
            {
                mixer.MasterVolume = (float)(Volume / 100.0);
                if (_wasPaused)
                {
                    src.Play();
                    mixer.Start();
                    return;
                }
                mixer.Seek(_start);
                src.AttachToClock(mixer.MasterClock);
                mixer.AddSourcePrepared(src);

                //Pause after the add: the first attach opens the stream, the command queue only drains while it renders
                mixer.Pause();
                mixer.StartPreparedSources(_start);
                mixer.Start();
            });
        }
        catch (Exception ex)
        {
            IsPlaying = false;
            Error = ex.Message;
            return;
        }
        finally
        {
            _transportLock.Release();
        }

        _setPosition(_start);
        _positionTimer.Start();
    }

    async Task _pause()
    {
        _positionTimer.Stop();
        IsPlaying = false;
        IsPaused = true;

        await _transportLock.WaitAsync();
        try
        {
            var mixer = _audio.Mixer;
            var src = _source;
            await Task.Run(() =>
            {
                mixer?.Pause();
                src?.Pause();
            });
            if (mixer != null) _setPosition(mixer.MasterClock.CurrentTimestamp);
        }
        finally
        {
            _transportLock.Release();
        }
    }

    async Task _stop()
    {
        _positionTimer.Stop();
        bool _wasOn = IsPlaying || IsPaused;
        IsPlaying = false;
        IsPaused = false;
        if (!_wasOn || _source == null) { _setPosition(0); return; }

        await _transportLock.WaitAsync();
        try
        {
            var mixer = _audio.Mixer;
            var src = _source;
            await Task.Run(() =>
            {
                try { src.Stop(); } catch { }
                try { src.DetachFromClock(); } catch { }
                if (mixer != null)
                {
                    try { mixer.Seek(0); } catch { }
                    try { mixer.RemoveSource(src.Id); } catch { }
                }
            });
        }
        finally
        {
            _transportLock.Release();
        }
        _setPosition(0);
    }

    void _tick()
    {
        var mixer = _audio.Mixer;
        if (!IsPlaying || mixer == null) return;

        //The clock only moves once per block, we interpolate in between
        double _enginePos = mixer.MasterClock.CurrentTimestamp;
        double _now = _interpWatch.Elapsed.TotalSeconds;
        if (_enginePos != _lastEnginePos)
        {
            _lastEnginePos = _enginePos;
            _lastEnginePosAt = _now;
        }
        double _pos = _lastEnginePos + (_now - _lastEnginePosAt);

        if (Duration > 0 && _pos >= Duration + 0.3)
        {
            StopCommand.Execute(null);
            return;
        }
        _show(_pos);
    }

    void _setPosition(double sec)
    {
        _lastEnginePos = sec;
        _lastEnginePosAt = _interpWatch.Elapsed.TotalSeconds;
        _lastShownSecond = -1;
        _show(sec);
    }

    void _show(double sec)
    {
        Position = sec;
        NormalizedPosition = Duration > 0 ? Math.Clamp(sec / Duration, 0, 1) : 0;
        if ((int)sec == _lastShownSecond) return;
        _lastShownSecond = (int)sec;
        PositionText = $"{Format(sec)} / {Format(Duration)}";
    }

    public static string Format(double sec)
    {
        int s = (int)Math.Round(Math.Max(0, sec));
        return $"{s / 60}:{s % 60:00}";
    }

    /// <summary>
    /// Lets go of the file, e.g. before its folder gets deleted.
    /// </summary>
    public async Task ReleaseAsync(string path)
    {
        if (CurrentPath == null || !CurrentPath.StartsWith(path, StringComparison.Ordinal)) return;
        await _stop();
        _unload();
        CurrentPath = null;
        HasTrack = false;
        WaveformData = null;
        Title = "Nothing playing";
        Subtitle = null;
        Duration = 0;
        _setPosition(0);
    }

    void _unload()
    {
        _source?.Dispose();
        _source = null;
    }

    public void Dispose()
    {
        _positionTimer.Stop();
        _unload();
    }
}
