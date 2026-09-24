using System.Text;
using System.Text.Json;
using OwnMusicAI.Engine;

namespace OwnMusicAI.App.Services;

public enum SongStatus { Queued, Running, Done, Failed, Canceled }

/// <summary>
/// What sits in a song folder as job.json: the request plus how it went.
/// </summary>
public sealed class SongJob
{
    public DateTime Created { get; set; } = DateTime.Now;
    public SongRequest Request { get; set; } = new SongRequest();
    public SongStatus Status { get; set; } = SongStatus.Queued;
    public string? Error { get; set; }
    public double DurationSeconds { get; set; }
}

/// <summary>
/// One folder per song under the output dir. The folder is the song's identity: its job.json,
/// the resume state and the wav all live there.
/// </summary>
public static class SongLibrary
{
    public const string JobFile = "job.json";
    public const string WavFile = "song.wav";
    public const string LyricsFile = "lyrics.txt";

    /// <summary>Lyrics drafts, next to the song folders (no job.json, so Scan skips it).</summary>
    public const string LyricsFolder = "Lyrics";

    public static IEnumerable<(string Folder, SongJob Job)> Scan(string root)
    {
        if (!Directory.Exists(root)) yield break;
        foreach (var dir in Directory.GetDirectories(root).OrderByDescending(d => d))
        {
            string _file = Path.Combine(dir, JobFile);
            if (!File.Exists(_file)) continue;
            SongJob? _job;
            try { _job = JsonSerializer.Deserialize<SongJob>(File.ReadAllText(_file), AppSettings.Json); }
            catch (JsonException) { continue; }
            if (_job == null) continue;

            //A run the app didn't live to finish can still be resumed
            if (_job.Status is SongStatus.Running or SongStatus.Queued) _job.Status = SongStatus.Canceled;
            yield return (dir, _job);
        }
    }

    public static string NewFolder(string root, SongJob job)
    {
        string _name = $"{job.Created:yyyyMMdd-HHmmss}-{_slug(job.Request.Title)}";
        string _dir = Path.Combine(root, _name);
        for (int i = 2; Directory.Exists(_dir); i++) _dir = Path.Combine(root, $"{_name}-{i}");
        Directory.CreateDirectory(_dir);
        return _dir;
    }

    public static void Save(string folder, SongJob job) =>
        File.WriteAllText(Path.Combine(folder, JobFile), JsonSerializer.Serialize(job, AppSettings.Json));

    /// <summary>Length straight from the 16-bit wav header math, no decoder needed.</summary>
    public static double WavSeconds(string wav)
    {
        var _info = new FileInfo(wav);
        return _info.Exists ? Math.Max(0, _info.Length - 44) / (48000.0 * 2 * 2) : 0;
    }

    /// <summary>Title with the characters no file system likes dropped, "" if nothing is left.</summary>
    public static string FileName(string title)
    {
        var _bad = Path.GetInvalidFileNameChars();
        return new string(title.Trim().Where(c => !_bad.Contains(c) && c != ':').ToArray()).Trim(' ', '.');
    }

    static string _slug(string title)
    {
        var sb = new StringBuilder();
        foreach (char c in title.ToLowerInvariant())
        {
            if (char.IsLetterOrDigit(c) && c < 128) sb.Append(c);
            else if (sb.Length > 0 && sb[^1] != '-') sb.Append('-');
            if (sb.Length >= 32) break;
        }
        string _s = sb.ToString().Trim('-');
        return _s.Length == 0 ? "song" : _s;
    }
}
