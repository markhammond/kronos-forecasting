namespace Tsfm.Forecasting;

/// <summary>
/// A source of pre-trained weights. Streams, not paths: an embedded resource is located
/// by the runtime rather than by configuration.
///
/// <para>Model and tokenizer are not independently valid — a decoder is trained against
/// one codebook, and mixing them yields plausible numbers from meaningless tokens — so a
/// checkpoint supplies both.</para>
/// </summary>
public interface ICheckpoint
{
    /// <summary>Repository name and revision, for provenance.</summary>
    string Name { get; }

    /// <summary>Longest context this checkpoint can attend over, in bars. A property of the
    /// checkpoint, not the architecture — it varies between published models. Supplying more
    /// is not an error upstream, which truncates, but the excess is simply not read.</summary>
    int MaxContext { get; }

    Stream OpenModel();
    string ModelConfigJson { get; }

    Stream OpenTokenizer();
    string TokenizerConfigJson { get; }
}

/// <summary>Load from the published snapshot layout: <c>config.json</c> beside
/// <c>model.safetensors</c>. Development and parity only.</summary>
public sealed class DirectoryCheckpoint(
    string modelDir, string tokenizerDir, int maxContext = 512, string? name = null)
    : ICheckpoint
{
    public string Name { get; } = name ?? Path.GetFileName(modelDir.TrimEnd(Path.DirectorySeparatorChar));

    /// <summary>Not discoverable from a snapshot directory, so the caller supplies it;
    /// 512 matches Kronos-small and Kronos-base.</summary>
    public int MaxContext { get; } = maxContext;

    public Stream OpenModel() => File.OpenRead(Path.Combine(modelDir, "model.safetensors"));
    public string ModelConfigJson => File.ReadAllText(Path.Combine(modelDir, "config.json"));

    public Stream OpenTokenizer() => File.OpenRead(Path.Combine(tokenizerDir, "model.safetensors"));
    public string TokenizerConfigJson => File.ReadAllText(Path.Combine(tokenizerDir, "config.json"));

    public static bool Exists(string modelDir, string tokenizerDir)
        => File.Exists(Path.Combine(modelDir, "model.safetensors"))
        && File.Exists(Path.Combine(tokenizerDir, "model.safetensors"));
}

/// <summary>Checkpoint carried as embedded resources. Derived types supply the assembly
/// and resource names.</summary>
public abstract class EmbeddedCheckpoint : ICheckpoint
{
    protected abstract System.Reflection.Assembly Host { get; }
    protected abstract string ModelResource { get; }
    protected abstract string TokenizerResource { get; }
    protected abstract string ModelConfigResource { get; }
    protected abstract string TokenizerConfigResource { get; }

    public abstract string Name { get; }

    /// <inheritdoc/>
    public abstract int MaxContext { get; }

    public Stream OpenModel() => Open(ModelResource);
    public Stream OpenTokenizer() => Open(TokenizerResource);
    public string ModelConfigJson => ReadText(ModelConfigResource);
    public string TokenizerConfigJson => ReadText(TokenizerConfigResource);

    private Stream Open(string resource)
        => Host.GetManifestResourceStream(resource)
           ?? throw new InvalidOperationException(
               $"resource '{resource}' missing from {Host.GetName().Name}; " +
               "the weights assembly was built without its payload");

    private string ReadText(string resource)
    {
        using var reader = new StreamReader(Open(resource));
        return reader.ReadToEnd();
    }
}
