using Avalonia.Controls;
using Avalonia.Interactivity;
using OwnMusicAI.App.ViewModels;

namespace OwnMusicAI.App.Views;

/// <summary>
/// Style + lyrics + title, the same box on the Simple and the Advanced page.
/// </summary>
public partial class PromptEditor : UserControl
{
    public PromptEditor()
    {
        InitializeComponent();
    }

    void _lyricsLostFocus(object? sender, RoutedEventArgs e) => (DataContext as MainWindowViewModel)?.SaveLyricsDraft();
}
