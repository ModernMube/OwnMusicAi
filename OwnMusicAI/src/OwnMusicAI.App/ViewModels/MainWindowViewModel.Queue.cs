using OwnMusicAI.App.Services;
using OwnMusicAI.Engine;

namespace OwnMusicAI.App.ViewModels;

/// <summary>
/// The song queue: one song at a time on the GPU, oldest first. Every card is a folder under the
/// output dir, so the library is just those folders read back.
/// </summary>
public partial class MainWindowViewModel
{
    bool _pumping;

    public void Enqueue(SongRequest request)
    {
        var _job = new SongJob { Request = request };
        string _folder = SongLibrary.NewFolder(_settings.OutputDir, _job);
        SongLibrary.Save(_folder, _job);
        File.WriteAllText(Path.Combine(_folder, SongLibrary.LyricsFile), request.Lyrics);
        Songs.Insert(0, new SongItemViewModel(this, _folder, _job));
        _ = _pumpAsync();
    }

    void _loadLibrary()
    {
        Songs.Clear();
        foreach (var (folder, job) in SongLibrary.Scan(_settings.OutputDir))
            Songs.Add(new SongItemViewModel(this, folder, job));
    }

    async Task _pumpAsync()
    {
        if (_pumping) return;
        _pumping = true;
        try
        {
            while (Songs.LastOrDefault(s => s.Status == SongStatus.Queued) is SongItemViewModel next)
                await _runAsync(next);
        }
        finally
        {
            _pumping = false;
        }
    }

    async Task _runAsync(SongItemViewModel item)
    {
        var _job = item.Job;
        var r = _job.Request;
        item.Cts = new CancellationTokenSource();
        var ct = item.Cts.Token;
        _job.Status = SongStatus.Running;
        _job.Error = null;
        item.Refresh();
        SongLibrary.Save(item.Folder, _job);

        var _progress = new Progress<GenerationProgress>(item.Report);
        try
        {
            //Cover without a pre-read score: SheetSage2 first, same as the CLI's --abc-audio
            if (r.Mode != ScoreMode.Off && r.Abc == null && r.ReferenceAudio != null)
            {
                var a = await _pipeline.AnalyzeAsync(r.ReferenceAudio, r.Mode == ScoreMode.Melody, null, _progress, ct);
                r.Abc = a.Abc ?? throw new InvalidOperationException($"SheetSage2 could not build a score: {a.AbcError}");
                SongLibrary.Save(item.Folder, _job);
            }

            string _wav = await _pipeline.GenerateAsync(r, item.Folder, _progress, ct);
            _job.Status = SongStatus.Done;
            _job.DurationSeconds = SongLibrary.WavSeconds(_wav);
        }
        catch (OperationCanceledException)
        {
            _job.Status = SongStatus.Canceled;
        }
        catch (Exception ex)
        {
            _job.Status = SongStatus.Failed;
            _job.Error = ex.Message;
        }
        finally
        {
            item.Cts.Dispose();
            item.Cts = null;
        }

        SongLibrary.Save(item.Folder, _job);
        item.Refresh();
        if (_job.Status == SongStatus.Done && !Player.HasTrack)
            await Player.LoadAsync(item.WavPath, item.Title, item.Style, autoPlay: false);
    }

    public async Task PlaySongAsync(SongItemViewModel item)
    {
        if (!File.Exists(item.WavPath)) return;
        if (Player.CurrentPath == item.WavPath) { Player.PlayPauseCommand.Execute(null); return; }
        await Player.LoadAsync(item.WavPath, item.Title, item.Style);
    }

    /// <summary>Back into the queue - the pipeline continues from the saved state.</summary>
    public void ResumeSong(SongItemViewModel item)
    {
        item.Job.Status = SongStatus.Queued;
        item.Job.Error = null;
        item.Refresh();
        SongLibrary.Save(item.Folder, item.Job);
        _ = _pumpAsync();
    }

    public void Unqueue(SongItemViewModel item)
    {
        if (item.Status != SongStatus.Queued) return;
        item.Job.Status = SongStatus.Canceled;
        item.Refresh();
        SongLibrary.Save(item.Folder, item.Job);
    }

    public async Task DeleteSongAsync(SongItemViewModel item)
    {
        if (item.IsBusy)
        {
            item.PendingDelete = false;
            _status("Stop the song before deleting it", true);
            return;
        }
        await Player.ReleaseAsync(item.Folder);
        try
        {
            Directory.Delete(item.Folder, true);
            Songs.Remove(item);
        }
        catch (IOException ex)
        {
            _status($"Could not delete: {ex.Message}", true);
        }
    }
}
