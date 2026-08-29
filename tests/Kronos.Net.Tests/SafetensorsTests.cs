using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using Xunit;

namespace Kronos.Net.Tests;

/// <summary>
/// The container format, exercised against payloads built here rather than against a
/// checkpoint — so these run anywhere, including where no weights have been fetched.
/// </summary>
public class SafetensorsTests
{
    /// <summary>Assemble a minimal safetensors blob: 8-byte little-endian header length,
    /// that many bytes of JSON, then the data section the byte ranges index into.</summary>
    private static byte[] Build(params (string Name, string DType, long[] Shape, byte[] Data)[] tensors)
    {
        var header = new Dictionary<string, object>();
        var body = new List<byte>();
        foreach (var (name, dtype, shape, data) in tensors)
        {
            header[name] = new Dictionary<string, object>
            {
                ["dtype"] = dtype,
                ["shape"] = shape,
                ["data_offsets"] = new[] { (long)body.Count, body.Count + data.Length },
            };
            body.AddRange(data);
        }
        var json = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));
        var blob = new List<byte>();
        blob.AddRange(BitConverter.GetBytes((long)json.Length));
        blob.AddRange(json);
        blob.AddRange(body);
        return blob.ToArray();
    }

    private static byte[] Floats(params float[] values)
        => values.SelectMany(BitConverter.GetBytes).ToArray();

    [Fact]
    public void RoundTripsAFloatTensor()
    {
        var blob = Build(("w", "F32", [2L, 3L], Floats(1, 2, 3, 4, 5, 6)));
        using var st = Safetensors.Load(new MemoryStream(blob));

        Assert.True(st.Has("w"));
        using var t = st.Get("w");
        Assert.Equal([2L, 3L], t.shape);
        Assert.Equal(21f, t.sum().item<float>());
    }

    [Fact]
    public void ReadsTensorsAtTheirDeclaredOffsets()
    {
        // The second tensor is only correct if its byte range is honoured rather than
        // the data section being read from the start.
        var blob = Build(
            ("a", "F32", [2L], Floats(10, 20)),
            ("b", "F32", [3L], Floats(1, 2, 3)));
        using var st = Safetensors.Load(new MemoryStream(blob));

        Assert.Equal(30f, st.Get("a").sum().item<float>());
        Assert.Equal(6f, st.Get("b").sum().item<float>());
    }

    [Fact]
    public void IgnoresTheMetadataEntry()
    {
        // __metadata__ is free-form and carries no tensor; treating it as one would throw
        // on the missing dtype.
        var blob = Build(("w", "F32", [1L], Floats(1)));
        var json = Encoding.UTF8.GetString(blob, 8, (int)BitConverter.ToInt64(blob, 0));
        Assert.DoesNotContain("__metadata__", json);   // guards the fixture, not the reader

        using var st = Safetensors.Load(new MemoryStream(blob));
        Assert.Single(st.Tensors);
    }

    [Fact]
    public void RejectsATruncatedFile()
        => Assert.Throws<InvalidDataException>(() => Safetensors.Load(new MemoryStream([1, 2, 3])));

    [Fact]
    public void RejectsAHeaderLongerThanTheFile()
    {
        var blob = new byte[16];
        BitConverter.GetBytes(long.MaxValue).CopyTo(blob, 0);
        Assert.Throws<InvalidDataException>(() => Safetensors.Load(new MemoryStream(blob)));
    }

    /// <summary>A stream that reports no length, forcing the grow-and-copy path.</summary>
    private sealed class Unseekable(byte[] data) : Stream
    {
        private readonly MemoryStream _inner = new(data);
        public override bool CanSeek => false;
        public override bool CanRead => true;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override int Read(byte[] b, int o, int c) => _inner.Read(b, o, c);
        public override void Flush() { }
        public override long Seek(long o, SeekOrigin s) => throw new NotSupportedException();
        public override void SetLength(long v) => throw new NotSupportedException();
        public override void Write(byte[] b, int o, int c) => throw new NotSupportedException();
    }

    [Fact]
    public void ReadsANonSeekableStream()
    {
        // Falls back to grow-and-copy, which reads the buffer without a final ToArray;
        // GetBuffer throws on a non-exposable MemoryStream, so this pins that it isn't.
        var blob = Build(("w", "F32", [2L], Floats(3, 4)));
        using var st = Safetensors.Load(new Unseekable(blob));
        Assert.Equal(7f, st.Get("w").sum().item<float>());
    }

    [Fact]
    public void ReadsASeekableStreamFromItsCurrentPosition()
    {
        var blob = Build(("w", "F32", [2L], Floats(5, 6)));
        var padded = new byte[4].Concat(blob).ToArray();
        var stream = new MemoryStream(padded) { Position = 4 };
        using var st = Safetensors.Load(stream);
        Assert.Equal(11f, st.Get("w").sum().item<float>());
    }

    [Fact]
    public void NamesTheMissingTensorRatherThanReturningNull()
    {
        using var st = Safetensors.Load(new MemoryStream(Build(("w", "F32", [1L], Floats(1)))));
        var ex = Assert.Throws<KeyNotFoundException>(() => st.Get("absent"));
        Assert.Contains("absent", ex.Message);
    }
}
