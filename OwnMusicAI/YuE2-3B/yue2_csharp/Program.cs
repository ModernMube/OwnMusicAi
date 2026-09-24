namespace YuE2.Song;

/// <summary>
/// YuE2 song generation: C# pipeline on top of the Rust/Candle engine.
/// </summary>
internal static class Program
{
    static int Main(string[] args)
    {
        var _options = SongOptions.Parse(args);
        if (_options == null)
        {
            SongOptions.PrintUsage();
            return 1;
        }
        new SongGenerator(_options).Run();
        return 0;
    }
}
