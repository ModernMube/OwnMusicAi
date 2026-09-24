using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Ownaudio.Decoders;

namespace SheetSage2;

/// <summary>
/// File -> mono 24 kHz float samples through OwnAudio's native decoder (Symphonia + rubato),
/// the replacement for the ffmpeg pipe of audio_sheetsage2.py. Amplitude is never normalized,
/// the model was trained on absolute levels.
/// </summary>
public static class AudioLoader
{
    public const int SampleRate = 24000;

    /// <summary>
    /// Decodes and resamples to 24 kHz mono, optionally keeping only the first maxSeconds.
    /// </summary>
    public static float[] Load(string path, double? maxSeconds = null)
    {
        //OwnaudioNET registers the native decoder in its module initializer, so the assembly must be loaded
        RuntimeHelpers.RunModuleConstructor(typeof(OwnaudioNET.OwnaudioNet).Module.ModuleHandle);

        List<float> _interleaved;
        int _channels;
        using (var decoder = AudioDecoderFactory.Create(path, SampleRate, 0))
        {
            _channels = Math.Max(1, decoder.StreamInfo.Channels);
            _interleaved = new List<float>(1 << 20);
            var _buffer = new byte[8192 * _channels * sizeof(float)];
            while (true)
            {
                var _result = decoder.ReadFrames(_buffer);
                if (_result.FramesRead > 0)
                    _interleaved.AddRange(MemoryMarshal.Cast<byte, float>(_buffer.AsSpan(0, _result.FramesRead * _channels * sizeof(float))));
                if (_result.IsEOF) break;
                if (!_result.IsSucceeded) throw new InvalidDataException($"Cannot decode audio: {_result.ErrorMessage}");
            }
        }

        var _samples = _toMono(CollectionsMarshal.AsSpan(_interleaved), _channels);
        if (maxSeconds is double sec)
        {
            if (!double.IsFinite(sec) || sec <= 0) throw new ArgumentException("max_seconds must be finite and positive");
            int _keep = (int)Math.Min(_samples.Length, Math.Round(sec * SampleRate));
            _samples = _samples[.._keep];
        }
        return Validate(_samples);
    }

    /// <summary>Same checks as load_audio, for a waveform that is already mono 24 kHz.</summary>
    public static float[] Validate(float[] samples)
    {
        if (samples.Length < 1025 || !samples.All(float.IsFinite))
            throw new InvalidDataException("Audio must contain at least 1025 finite samples at 24 kHz");
        return samples;
    }

    /// <summary>
    /// Stereo is folded the way ffmpeg's -ac 1 does it, (L+R)/sqrt(2) rather than the plain
    /// average: that is what the reference pipeline feeds the model, and 3 dB of level does
    /// move the predictions. More exotic layouts just get averaged.
    /// </summary>
    static float[] _toMono(ReadOnlySpan<float> interleaved, int channels)
    {
        int _frames = interleaved.Length / channels;
        if (channels == 1) return interleaved[..(_frames * channels)].ToArray();

        var _mono = new float[_frames];
        float _scale = channels == 2 ? (float)(1.0 / Math.Sqrt(2.0)) : 1f / channels;
        for (int f = 0; f < _frames; f++)
        {
            float _sum = 0;
            for (int c = 0; c < channels; c++) _sum += interleaved[f * channels + c];
            _mono[f] = _sum * _scale;
        }
        return _mono;
    }
}
