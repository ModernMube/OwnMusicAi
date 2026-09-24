using Avalonia;

namespace OwnMusicAI.App;

/// <summary>
/// Entry point, plain Avalonia desktop start.
/// </summary>
internal sealed class Program
{
    [STAThread]
    public static void Main(string[] args) => BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);

    // the designer uses this too
    public static AppBuilder BuildAvaloniaApp()
        => AppBuilder.Configure<App>()
            .UsePlatformDetect()
            .WithInterFont()
            .LogToTrace();
}
