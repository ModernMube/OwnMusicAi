namespace YuE2.Song;

/// <summary>
/// Bare 16-bit PCM wav, enough for a test run.
/// </summary>
internal static class WavWriter
{
    public static void Write(string path, float[] interleaved, int sampleRate, int channels)
    {
        using (var _stream = File.Create(path))
        using (var w = new BinaryWriter(_stream))
        {
            int _bytes = interleaved.Length * 2;
            w.Write("RIFF"u8);
            w.Write(36 + _bytes);
            w.Write("WAVE"u8);
            w.Write("fmt "u8);
            w.Write(16);
            w.Write((short)1);
            w.Write((short)channels);
            w.Write(sampleRate);
            w.Write(sampleRate * channels * 2);
            w.Write((short)(channels * 2));
            w.Write((short)16);
            w.Write("data"u8);
            w.Write(_bytes);
            foreach (var s in interleaved) w.Write((short)MathF.Round(s * 32767f));
        }
    }
}
