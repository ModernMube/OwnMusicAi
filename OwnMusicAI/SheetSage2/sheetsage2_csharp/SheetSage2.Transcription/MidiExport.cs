using OwnAudio.Midi.File;
using SheetSage2.Notation;

namespace SheetSage2;

/// <summary>
/// SMF writing through OwnAudio.Midi, at the resolution and fixed 120 bpm the reference
/// pretty_midi files use, so note times land on the same ticks.
/// </summary>
internal static class MidiExport
{
    public const int TicksPerBeat = 960;
    public const double SecondsPerTick = 60.0 / (120.0 * TicksPerBeat);

    public static int ToTick(double seconds) => (int)Math.Round(seconds / SecondsPerTick, MidpointRounding.AwayFromZero);

    public static double ToSeconds(int tick) => tick * SecondsPerTick;

    /// <summary>
    /// What survives a pretty_midi write/read round trip: times snapped to ticks, and notes
    /// that collapse to zero length dropped.
    /// </summary>
    public static List<TimedNote> Quantize(IEnumerable<TimedNote> notes) =>
        notes.Select(n => new TimedNote(ToSeconds(ToTick(n.Start)), ToSeconds(ToTick(n.End)), n.Pitch, n.Track))
             .Where(n => n.End > n.Start)
             .ToList();

    public sealed class Track
    {
        public string Name = "";
        public int Program;
        public int Velocity = 100;
        public List<TimedNote> Notes = new List<TimedNote>();
    }

    public static byte[] Write(IEnumerable<Track> tracks)
    {
        var _tracks = new List<MidiTrack> { new MidiTrack(new List<MidiEvent> { new MidiEvent(0, 0x51, new byte[] { 0x07, 0xA1, 0x20 }) }) };
        int _channel = 0;
        foreach (var track in tracks)
        {
            if (_channel == 9) _channel++;
            var _events = new List<MidiEvent>
            {
                new MidiEvent(0, 0x03, System.Text.Encoding.UTF8.GetBytes(track.Name)),
                new MidiEvent(0, (byte)(0xC0 | (_channel & 0x0F)), (byte)track.Program, 0)
            };

            //note offs go first at a shared tick, otherwise a repeated pitch reads back as one note
            var _messages = new List<(int Tick, int Order, byte Status, byte Pitch, byte Velocity)>();
            foreach (var note in track.Notes)
            {
                _messages.Add((ToTick(note.Start), 1, (byte)(0x90 | (_channel & 0x0F)), (byte)note.Pitch, (byte)track.Velocity));
                _messages.Add((ToTick(note.End), 0, (byte)(0x80 | (_channel & 0x0F)), (byte)note.Pitch, 0));
            }
            int _previous = 0;
            foreach (var message in _messages.OrderBy(m => m.Tick).ThenBy(m => m.Order))
            {
                _events.Add(new MidiEvent(message.Tick - _previous, message.Status, message.Pitch, message.Velocity));
                _previous = message.Tick;
            }
            _tracks.Add(new MidiTrack(_events));
            _channel++;
        }

        using (var _stream = new MemoryStream())
        {
            MidiFileWriter.Write(new MidiFile(1, TicksPerBeat, _tracks.ToArray()), _stream);
            return _stream.ToArray();
        }
    }
}
