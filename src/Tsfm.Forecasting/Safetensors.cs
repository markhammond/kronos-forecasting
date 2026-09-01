using System.Buffers.Binary;
using System.Text.Json;
using static TorchSharp.torch;

namespace Tsfm.Forecasting;

/// <summary>
/// Reader for the safetensors container: 8-byte little-endian header length, that many
/// bytes of JSON declaring each tensor's dtype, shape and byte range, then the data
/// section those ranges index into.
/// </summary>
public sealed class Safetensors : IDisposable
{
    private readonly byte[] _blob;
    private readonly long _dataStart;

    public IReadOnlyDictionary<string, Entry> Tensors { get; }

    public readonly record struct Entry(string DType, long[] Shape, long Begin, long End);

    private Safetensors(byte[] blob, long dataStart, Dictionary<string, Entry> tensors)
        => (_blob, _dataStart, Tensors) = (blob, dataStart, tensors);

    public static Safetensors Load(string path)
    {
        var blob = File.ReadAllBytes(path);
        return Load(blob, blob.Length, path);
    }

    /// <summary>Read from a stream. The header declares byte ranges into the data section,
    /// so the whole payload must be held either way and parsing the JSON from the stream
    /// would not avoid that. Read a seekable stream into one exact-size buffer: growing a
    /// <see cref="MemoryStream"/> costs about 3.7x the payload in allocation, doubling its
    /// capacity and then copying out.</summary>
    public static Safetensors Load(Stream stream, string label = "<stream>")
    {
        if (stream.CanSeek)
        {
            var exact = new byte[stream.Length - stream.Position];
            stream.ReadExactly(exact);
            return Load(exact, exact.Length, label);
        }

        using var buffer = new MemoryStream();
        stream.CopyTo(buffer);
        return Load(buffer.GetBuffer(), (int)buffer.Length, label);   // GetBuffer: no final copy
    }

    /// <param name="blob">Raw container bytes.</param>
    /// <param name="length">Payload length. <paramref name="blob"/> may be longer when it
    /// is a grown buffer.</param>
    /// <param name="path">Label used in error messages.</param>
    private static Safetensors Load(byte[] blob, int length, string path)
    {
        if (length < 8) throw new InvalidDataException($"{path}: too short for a safetensors header");

        var headerLength = BinaryPrimitives.ReadInt64LittleEndian(blob.AsSpan(0, 8));

        // Subtract, never add: `8 + headerLength` overflows on a malformed file and
        // wraps negative, letting the bound pass.
        if (headerLength <= 0 || headerLength > length - 8)
            throw new InvalidDataException($"{path}: header length {headerLength} does not fit the file");

        using var doc = JsonDocument.Parse(blob.AsMemory(8, (int)headerLength));
        var tensors = new Dictionary<string, Entry>(StringComparer.Ordinal);
        foreach (var prop in doc.RootElement.EnumerateObject())
        {
            if (prop.NameEquals("__metadata__")) continue;   // free-form; carries no tensor
            var dtype = prop.Value.GetProperty("dtype").GetString()
                        ?? throw new InvalidDataException($"{prop.Name}: missing dtype");
            var shape = prop.Value.GetProperty("shape").EnumerateArray().Select(e => e.GetInt64()).ToArray();
            var offsets = prop.Value.GetProperty("data_offsets");
            tensors[prop.Name] = new Entry(dtype, shape, offsets[0].GetInt64(), offsets[1].GetInt64());
        }
        return new Safetensors(blob, 8 + headerLength, tensors);
    }

    /// <summary>Materialise one tensor from the header's declared shape and dtype. A
    /// layout mismatch fails here, not as a mis-shaped forward pass.</summary>
    public Tensor Get(string name)
    {
        if (!Tensors.TryGetValue(name, out var e))
            throw new KeyNotFoundException($"tensor '{name}' not in checkpoint");

        var span = _blob.AsSpan((int)(_dataStart + e.Begin), (int)(e.End - e.Begin));
        return e.DType switch
        {
            "F32" => from_array(ToArray<float>(span), ScalarType.Float32).reshape(e.Shape),
            "I64" => from_array(ToArray<long>(span), ScalarType.Int64).reshape(e.Shape),
            _ => throw new NotSupportedException($"{name}: dtype {e.DType} not supported"),
        };
    }

    public bool Has(string name) => Tensors.ContainsKey(name);

    private static T[] ToArray<T>(ReadOnlySpan<byte> span) where T : unmanaged
    {
        var typed = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, T>(span);
        return typed.ToArray();
    }

    public void Dispose() { /* blob is managed; nothing native held */ }
}
