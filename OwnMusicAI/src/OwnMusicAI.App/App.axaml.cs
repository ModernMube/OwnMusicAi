using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using OwnMusicAI.App.ViewModels;
using OwnMusicAI.App.Views;

namespace OwnMusicAI.App;

/// <summary>
/// Root application, builds the main window with its view model.
/// </summary>
public class App : Application
{
    public override void Initialize()
    {
        AvaloniaXamlLoader.Load(this);
    }

    public override void OnFrameworkInitializationCompleted()
    {
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            var _vm = new MainWindowViewModel();
            desktop.MainWindow = new MainWindow { DataContext = _vm };
            desktop.ShutdownRequested += (_, _) => _vm.Dispose();
        }

        base.OnFrameworkInitializationCompleted();
    }
}
