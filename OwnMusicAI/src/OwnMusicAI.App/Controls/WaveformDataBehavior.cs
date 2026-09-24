using System.Globalization;
using System.Windows.Input;
using Avalonia;
using Avalonia.Xaml.Interactivity;
using OwnaudioNET.Visualization;

namespace OwnMusicAI.App.Controls;

/// <summary>
/// Same glue as in OwnBackingPlayer: new peaks go in through SetAudioData(), a dragged playhead
/// comes back out as a seek command in seconds.
/// </summary>
public class WaveformDataBehavior : Behavior<WaveAvaloniaDisplay>
{
    public static readonly StyledProperty<float[]?> AudioDataProperty =
        AvaloniaProperty.Register<WaveformDataBehavior, float[]?>(nameof(AudioData));

    public static readonly StyledProperty<double> TotalDurationSecondsProperty =
        AvaloniaProperty.Register<WaveformDataBehavior, double>(nameof(TotalDurationSeconds));

    public static readonly StyledProperty<ICommand?> SeekCommandProperty =
        AvaloniaProperty.Register<WaveformDataBehavior, ICommand?>(nameof(SeekCommand));

    public float[]? AudioData
    {
        get => GetValue(AudioDataProperty);
        set => SetValue(AudioDataProperty, value);
    }

    /// <summary>
    /// What a 0–1 position gets multiplied by.
    /// </summary>
    public double TotalDurationSeconds
    {
        get => GetValue(TotalDurationSecondsProperty);
        set => SetValue(TotalDurationSecondsProperty, value);
    }

    /// <summary>
    /// Gets the seek position in seconds, as an invariant string.
    /// </summary>
    public ICommand? SeekCommand
    {
        get => GetValue(SeekCommandProperty);
        set => SetValue(SeekCommandProperty, value);
    }

    protected override void OnAttached()
    {
        base.OnAttached();
        if (AssociatedObject != null)
        {
            AssociatedObject.PlaybackPositionChanged += _onPlaybackPositionChanged;
            if (AudioData is { Length: > 0 } data) AssociatedObject.SetAudioData(data);
        }
    }

    protected override void OnDetaching()
    {
        if (AssociatedObject != null)
            AssociatedObject.PlaybackPositionChanged -= _onPlaybackPositionChanged;
        base.OnDetaching();
    }

    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
    {
        base.OnPropertyChanged(change);

        if (change.Property == AudioDataProperty && AssociatedObject != null)
            AssociatedObject.SetAudioData(change.NewValue as float[] ?? Array.Empty<float>());
    }

    private void _onPlaybackPositionChanged(object? sender, double normalizedPos)
    {
        if (TotalDurationSeconds <= 0 || SeekCommand == null) return;
        string posStr = (normalizedPos * TotalDurationSeconds).ToString("G", CultureInfo.InvariantCulture);
        if (SeekCommand.CanExecute(posStr))
            SeekCommand.Execute(posStr);
    }
}
