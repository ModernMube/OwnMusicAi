using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Platform.Storage;
using OwnMusicAI.App.ViewModels;

namespace OwnMusicAI.App.Views;

/// <summary>
/// Main window. Hands the storage pickers to the view model and takes dropped audio files as
/// the reference song.
/// </summary>
public partial class MainWindow : Window, IFilePicker
{
    static readonly string[] _audioExt = { ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".aiff", ".aif" };

    public MainWindow()
    {
        InitializeComponent();
        AddHandler(DragDrop.DragOverEvent, _onDragOver);
        AddHandler(DragDrop.DropEvent, _onDrop);
    }

    protected override void OnDataContextChanged(EventArgs e)
    {
        base.OnDataContextChanged(e);
        if (DataContext is MainWindowViewModel vm) vm.Picker = this;
    }

    public async Task<string?> OpenFileAsync(string title, string kind, string? startFolder, params string[] patterns)
    {
        var _start = startFolder != null && Directory.Exists(startFolder) ? await StorageProvider.TryGetFolderFromPathAsync(startFolder) : null;
        var _files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = title,
            AllowMultiple = false,
            SuggestedStartLocation = _start,
            FileTypeFilter = new[] { new FilePickerFileType(kind) { Patterns = patterns }, FilePickerFileTypes.All }
        });
        return _files.Count == 0 ? null : _files[0].TryGetLocalPath();
    }

    public async Task<string?> OpenFolderAsync(string title)
    {
        var _dirs = await StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions { Title = title, AllowMultiple = false });
        return _dirs.Count == 0 ? null : _dirs[0].TryGetLocalPath();
    }

    //Without this the drag is refused before it ever reaches the drop
    void _onDragOver(object? sender, DragEventArgs e)
    {
        e.DragEffects = _droppedAudio(e) != null ? DragDropEffects.Copy : DragDropEffects.None;
        e.Handled = true;
    }

    async void _onDrop(object? sender, DragEventArgs e)
    {
        if (DataContext is not MainWindowViewModel vm || _droppedAudio(e) is not string file) return;
        e.Handled = true;
        vm.IsAdvancedPage = true;
        await vm.SetReferenceAsync(file);
    }

    static string? _droppedAudio(DragEventArgs e) => e.DataTransfer.TryGetFiles()?
        .Select(f => f.TryGetLocalPath())
        .FirstOrDefault(p => p != null && _audioExt.Contains(Path.GetExtension(p).ToLowerInvariant()));
}
