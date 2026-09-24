using System.Net;
using System.Net.Http.Headers;
using System.Text.Json;

namespace OwnMusicAI.Engine;

/// <summary>Bytes so far / all bytes of the whole download, File = the one on the wire.</summary>
public readonly record struct DownloadProgress(long Done, long Total, string File);

/// <summary>
/// Pulls the missing weights from huggingface.co/m-a-p. YuE2-3B and SheetSage2 land in their
/// own folders, YuE2-Vae and MERT-v2-FullSong in the HF cache where FindVae / FindMert look.
/// A stopped download goes on from its .part file.
/// </summary>
public static class ModelDownloader
{
    static readonly HttpClient _http = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };

    static readonly string[] _yueFiles = { "config.json", "qwen.tiktoken", "weights_manifest.json", "LICENSE", "THIRD_PARTY_NOTICES.md", "examples/tonight-awake.json", "model.safetensors" };
    static readonly string[] _sheetSageFiles = { "config.json", "LICENSE", "THIRD_PARTY_NOTICES.md", "model.safetensors" };
    static readonly string[] _cacheFiles = { "config.json", "LICENSE", "model.safetensors" };

    sealed class _Item
    {
        public string Repo = "", Revision = "", Name = "", Target = "";
        public long Size;
    }

    /// <summary>Which of the four repos has something missing, empty list = all there.</summary>
    public static List<string> Missing(ModelPaths paths)
    {
        var _missing = new List<string>();
        if (_incomplete(paths.YuE2Dir, "config.json", "qwen.tiktoken", "model.safetensors")) _missing.Add("YuE2-3B");
        if (paths.VaeDir == null && ModelPaths.FindVae() == null) _missing.Add("YuE2-Vae");
        if (_incomplete(paths.SheetSageDir, "config.json", "model.safetensors")) _missing.Add("SheetSage2");
        if (!_cached("MERT-v2-FullSong")) _missing.Add("MERT-v2-FullSong");
        return _missing;
    }

    /// <summary>
    /// Fetches everything Missing() lists. Only the model files go, the repos' demo audio and
    /// python bits stay on the hub.
    /// </summary>
    public static async Task DownloadAsync(ModelPaths paths, IProgress<DownloadProgress>? progress, CancellationToken ct)
    {
        var _items = new List<_Item>();
        foreach (string _repo in Missing(paths))
        {
            switch (_repo)
            {
                case "YuE2-3B": _items.AddRange(await _plan(_repo, null, paths.YuE2Dir, _yueFiles, ct)); break;
                case "SheetSage2": _items.AddRange(await _plan(_repo, null, paths.SheetSageDir, _sheetSageFiles, ct)); break;
                case "MERT-v2-FullSong": _items.AddRange(await _plan(_repo, _mertRevision(paths.SheetSageDir), null, _cacheFiles, ct)); break;
                default: _items.AddRange(await _plan(_repo, null, null, _cacheFiles, ct)); break;
            }
        }

        long _total = _items.Sum(i => i.Size), _done = 0;
        foreach (var it in _items)
        {
            await _fetch(it, n =>
            {
                _done += n;
                progress?.Report(new DownloadProgress(_done, _total, $"{it.Repo}/{it.Name}"));
            }, ct);
        }
    }

    /// <summary>
    /// What's left of one repo. revision null = main; target null = the HF cache snapshot for
    /// that revision (refs/main gets written too, so the hf CLI knows it).
    /// </summary>
    static async Task<List<_Item>> _plan(string repo, string? revision, string? target, string[] files, CancellationToken ct)
    {
        string _url = $"https://huggingface.co/api/models/m-a-p/{repo}/revision/{revision ?? "main"}?blobs=true";
        using (var _doc = JsonDocument.Parse(await _http.GetStringAsync(_url, ct)))
        {
            string _sha = _doc.RootElement.GetProperty("sha").GetString()!;
            if (target == null)
            {
                string _repoDir = Path.GetDirectoryName(_snapshots(repo))!;
                target = Path.Combine(_repoDir, "snapshots", _sha);
                Directory.CreateDirectory(Path.Combine(_repoDir, "refs"));
                if (revision == null) File.WriteAllText(Path.Combine(_repoDir, "refs", "main"), _sha);
            }

            var _sizes = new Dictionary<string, long>();
            foreach (var s in _doc.RootElement.GetProperty("siblings").EnumerateArray())
                _sizes[s.GetProperty("rfilename").GetString()!] = s.GetProperty("size").GetInt64();

            var _items = new List<_Item>();
            foreach (string f in files)
            {
                if (File.Exists(Path.Combine(target, f)) || !_sizes.ContainsKey(f)) continue;
                _items.Add(new _Item { Repo = repo, Revision = _sha, Name = f, Target = Path.Combine(target, f), Size = _sizes[f] });
            }
            return _items;
        }
    }

    static async Task _fetch(_Item item, Action<long> advance, CancellationToken ct)
    {
        string _part = item.Target + ".part";
        Directory.CreateDirectory(Path.GetDirectoryName(item.Target)!);
        long _have = File.Exists(_part) ? new FileInfo(_part).Length : 0;
        if (_have > item.Size) { File.Delete(_part); _have = 0; }

        if (_have < item.Size)
        {
            var _req = new HttpRequestMessage(HttpMethod.Get, $"https://huggingface.co/m-a-p/{item.Repo}/resolve/{item.Revision}/{item.Name}");
            if (_have > 0) _req.Headers.Range = new RangeHeaderValue(_have, null);
            using (var _resp = await _http.SendAsync(_req, HttpCompletionOption.ResponseHeadersRead, ct))
            {
                _resp.EnsureSuccessStatusCode();
                if (_resp.StatusCode != HttpStatusCode.PartialContent) _have = 0;
                advance(_have);

                using (var _src = await _resp.Content.ReadAsStreamAsync(ct))
                using (var _dst = new FileStream(_part, _have > 0 ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.None, 1 << 16))
                {
                    var _buf = new byte[1 << 20];
                    int n;
                    while ((n = await _src.ReadAsync(_buf, ct)) > 0)
                    {
                        await _dst.WriteAsync(_buf.AsMemory(0, n), ct);
                        advance(n);
                    }
                }
            }
        }
        else advance(_have);

        if (new FileInfo(_part).Length != item.Size)
        {
            File.Delete(_part);
            throw new IOException($"{item.Repo}/{item.Name}: incomplete download, try again");
        }
        File.Move(_part, item.Target, true);
    }

    static bool _incomplete(string dir, params string[] files) =>
        dir.Length == 0 || files.Any(f => !File.Exists(Path.Combine(dir, f)));

    static bool _cached(string repo) =>
        Directory.Exists(_snapshots(repo)) && Directory.GetDirectories(_snapshots(repo)).Any(d => File.Exists(Path.Combine(d, "model.safetensors")));

    /// <summary>The MERT commit SheetSage2 was trained against, if its config is already here.</summary>
    static string? _mertRevision(string sheetSageDir)
    {
        string _config = Path.Combine(sheetSageDir, "config.json");
        if (!File.Exists(_config)) return null;
        using (var _doc = JsonDocument.Parse(File.ReadAllText(_config)))
            return _doc.RootElement.TryGetProperty("base_model_revision", out var r) ? r.GetString() : null;
    }

    static string _snapshots(string repo) => Path.Combine(ModelPaths.HfHome(), "hub", $"models--m-a-p--{repo}", "snapshots");
}
