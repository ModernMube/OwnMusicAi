using System.Text;
using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;

namespace YuE2.Song;

/// <summary>
/// qwen.tiktoken + the checkpoint's split regex. Plain text only, protocol ids get added by hand.
/// </summary>
internal sealed class YuE2Tokenizer
{
    readonly TiktokenTokenizer _bpe;

    public YuE2Tokenizer(string modelDir)
    {
        using (var _stream = File.OpenRead(Path.Combine(modelDir, "qwen.tiktoken")))
        {
            _bpe = TiktokenTokenizer.Create(_stream, new RegexPreTokenizer(new Regex(Protocol.SplitPattern, RegexOptions.Compiled), null), null);
        }
    }

    /// <summary>NFC first, same as encode_ordinary in tokenization_yue2.py.</summary>
    public List<int> Encode(string text) => new List<int>(_bpe.EncodeToIds(text.Normalize(NormalizationForm.FormC)));

    public string Decode(IEnumerable<int> ids) => _bpe.Decode(ids);
}
