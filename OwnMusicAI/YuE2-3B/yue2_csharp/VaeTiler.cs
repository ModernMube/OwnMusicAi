namespace YuE2.Song;

/// <summary>
/// YuE2VAE.decode_tiled on the host: core frames + halo on both sides, only the core is kept.
/// </summary>
internal static class VaeTiler
{
    /// <summary>
    /// Interleaved stereo clipped to [-1, 1]. Smaller cores keep Candle's im2col buffers small,
    /// the halo (16 frames) already covers the decoder's receptive field.
    /// </summary>
    public static float[] Decode(Yue2Engine engine, float[] latents, int core, int halo, Action<int, int> progress)
    {
        int _dim = engine.Info.LatentDim;
        int _ratio = engine.Info.DownsamplingRatio;
        int _frames = latents.Length / _dim;
        long _total = engine.VaeOutputLength(_frames);
        int _tiles = (_frames + core - 1) / core;
        var _audio = new float[_total * 2];

        for (int start = 0; start < _frames; start += core)
        {
            int end = Math.Min(_frames, start + core);
            int left = Math.Max(0, start - halo);
            int right = Math.Min(_frames, end + halo);

            long _tileLen = engine.VaeOutputLength(right - left);
            var _tile = new float[_tileLen * 2];
            engine.DecodeVae(latents.AsSpan(left * _dim, (right - left) * _dim), _tile);

            long _outStart = (long)start * _ratio;
            long _count = Math.Min((long)end * _ratio, _total) - _outStart;
            long _crop = (long)(start - left) * _ratio;
            for (long i = 0; i < _count; i++)
            {
                _audio[(_outStart + i) * 2] = Math.Clamp(_tile[_crop + i], -1f, 1f);
                _audio[(_outStart + i) * 2 + 1] = Math.Clamp(_tile[_tileLen + _crop + i], -1f, 1f);
            }
            progress(start / core + 1, _tiles);
        }
        return _audio;
    }
}
