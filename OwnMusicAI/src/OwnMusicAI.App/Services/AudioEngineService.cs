using OwnaudioNET;
using OwnaudioNET.Mixing;

namespace OwnMusicAI.App.Services;

/// <summary>
/// Owns the OwnAudioNet engine and the AudioMixer, the trimmed down OwnBackingPlayer one:
/// default output device, no capture.
/// </summary>
public sealed class AudioEngineService : IDisposable
{
    static AudioEngineService? _instance;
    static readonly object _lock = new object();

    AudioMixer? _mixer;
    bool _disposed;

    AudioEngineService() { }

    public static AudioEngineService Instance
    {
        get
        {
            if (_instance == null)
            {
                lock (_lock) { _instance ??= new AudioEngineService(); }
            }
            return _instance;
        }
    }

    /// <summary>The live mixer, null until init ran.</summary>
    public AudioMixer? Mixer => _mixer;

    public bool IsInitialized => _mixer != null && OwnaudioNet.IsInitialized;

    /// <summary>
    /// Every attached source ran out. Comes off the native EOS latch, not the UI thread.
    /// </summary>
    public event EventHandler? PlaybackEnded;

    Task? _init;

    /// <summary>
    /// Starts once; everyone who needs the mixer just awaits the same task.
    /// </summary>
    public Task InitializeAsync() => _init ??= _startAsync();

    async Task _startAsync()
    {
        var config = OwnaudioNet.CreateDefaultConfig();
        config.EnableInput = false;
        config.BufferSize = 1024;

        await OwnaudioNet.InitializeAsync(config);
        OwnaudioNet.Start();

        _mixer = new AudioMixer(OwnaudioNet.Engine!.UnderlyingEngine, bufferSizeInFrames: config.BufferSize);
        _mixer.PlaybackEnded += _onPlaybackEnded;
        _mixer.Start();
    }

    void _onPlaybackEnded(object? sender, EventArgs e) => PlaybackEnded?.Invoke(this, e);

    /// <summary>
    /// Every step on its own - a dead device throws in the mixer dispose, and that must not
    /// skip the engine shutdown.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_mixer != null)
        {
            _mixer.PlaybackEnded -= _onPlaybackEnded;
            _step(_mixer.Stop);
            _step(_mixer.Dispose);
            _mixer = null;
        }
        if (OwnaudioNet.IsInitialized)
        {
            _step(OwnaudioNet.Stop);
            _step(OwnaudioNet.Shutdown);
        }
    }

    static void _step(Action action)
    {
        try { action(); } catch { }
    }
}
