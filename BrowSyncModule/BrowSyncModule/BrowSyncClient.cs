using Microsoft.Extensions.Logging;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace BrowSyncModule;

// ---------------------------------------------------------------------------
// JSON contract — mirrors the Python server's "brow" response exactly
// ---------------------------------------------------------------------------

internal sealed class BrowFrame
{
    [JsonPropertyName("type")]
    public string Type { get; set; } = "";

    [JsonPropertyName("ts")]
    public double Timestamp { get; set; }

    [JsonPropertyName("outputs")]
    public BrowOutputs? Outputs { get; set; }

    [JsonPropertyName("mode")]
    public string Mode { get; set; } = "unknown";
}

internal sealed class BrowOutputs
{
    // Unified Expressions uses per-side inner/outer split
    [JsonPropertyName("BrowInnerUpLeft")]   public float BrowInnerUpLeft   { get; set; }
    [JsonPropertyName("BrowInnerUpRight")]  public float BrowInnerUpRight  { get; set; }
    [JsonPropertyName("BrowOuterUpLeft")]   public float BrowOuterUpLeft   { get; set; }
    [JsonPropertyName("BrowOuterUpRight")]  public float BrowOuterUpRight  { get; set; }
    [JsonPropertyName("BrowLowererLeft")]   public float BrowLowererLeft   { get; set; }
    [JsonPropertyName("BrowLowererRight")]  public float BrowLowererRight  { get; set; }
    [JsonPropertyName("BrowPinchLeft")]     public float BrowPinchLeft     { get; set; }
    [JsonPropertyName("BrowPinchRight")]    public float BrowPinchRight    { get; set; }
}

internal sealed class PingMessage
{
    [JsonPropertyName("type")]
    public string Type { get; init; } = "ping";
}

// ---------------------------------------------------------------------------
// Status reported back to the module
// ---------------------------------------------------------------------------

internal enum WsConnectionState { Disconnected, Connecting, Connected }

// ---------------------------------------------------------------------------
// WebSocket client
// ---------------------------------------------------------------------------

/// <summary>
/// Maintains a persistent WebSocket connection to the BrowSync Python server.
/// Runs its own background thread. The latest received BrowOutputs is always
/// available via <see cref="LatestOutputs"/> (thread-safe read).
/// </summary>
internal sealed class BrowSyncClient : IDisposable
{
    // -- Config --------------------------------------------------------------
    private readonly Uri _serverUri;
    private readonly TimeSpan _reconnectDelay = TimeSpan.FromSeconds(3);
    private readonly TimeSpan _pingInterval   = TimeSpan.FromSeconds(5);
    private readonly int      _receiveBufferSize = 4096;

    // -- State ---------------------------------------------------------------
    private volatile BrowOutputs? _latestOutputs;
    private volatile WsConnectionState _state = WsConnectionState.Disconnected;
    private volatile string _lastMode = "disconnected";
    private ulong _framesReceived;

    private CancellationTokenSource? _cts;
    private Thread? _thread;

    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    // -- Public interface ----------------------------------------------------

    public BrowOutputs? LatestOutputs => _latestOutputs;
    public WsConnectionState State    => _state;
    public string LastMode            => _lastMode;
    public ulong FramesReceived       => _framesReceived;

    public BrowSyncClient(string host = "localhost", int port = 7720)
    {
        _serverUri = new Uri($"ws://{host}:{port}");
    }

    public void Start()
    {
        _cts = new CancellationTokenSource();
        _thread = new Thread(RunLoop)
        {
            Name         = "BrowSync-WS",
            IsBackground = true,
        };
        _thread.Start();
    }

    public void Stop()
    {
        _cts?.Cancel();
        _thread?.Join(TimeSpan.FromSeconds(5));
        _cts?.Dispose();
        _cts = null;
    }

    public void Dispose() => Stop();

    // -- Background loop -----------------------------------------------------

    private void RunLoop()
    {
        var token = _cts!.Token;

        while (!token.IsCancellationRequested)
        {
            try
            {
                ConnectAndReceive(token).GetAwaiter().GetResult();
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                BrowSyncModule.Log?.LogWarning($"[BrowSync] WS error: {ex.Message}. Reconnecting in {_reconnectDelay.TotalSeconds}s...");
            }

            _state = WsConnectionState.Disconnected;
            _latestOutputs = null;

            if (!token.IsCancellationRequested)
                Thread.Sleep(_reconnectDelay);
        }

        _state = WsConnectionState.Disconnected;
    }

    private async Task ConnectAndReceive(CancellationToken token)
    {
        _state = WsConnectionState.Connecting;
        BrowSyncModule.Log?.LogInformation($"[BrowSync] Connecting to {_serverUri}...");

        using var ws = new ClientWebSocket();
        ws.Options.KeepAliveInterval = TimeSpan.FromSeconds(10);

        await ws.ConnectAsync(_serverUri, token);
        _state = WsConnectionState.Connected;
        BrowSyncModule.Log?.LogInformation("[BrowSync] Connected to BrowSync server.");

        // Recalibrate all subsystems on (re)connect so state is clean
        await SendJsonAsync(ws, """{"type":"recalibrate","target":"all"}""", token);

        using var pingTimer = new PeriodicTimer(_pingInterval);
        var pingTask  = PingLoop(ws, pingTimer, token);
        var recvTask  = ReceiveLoop(ws, token);

        await Task.WhenAny(pingTask, recvTask);

        // Whichever finished first, cancel the other by closing the socket
        if (ws.State == WebSocketState.Open)
        {
            try { await ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "done", CancellationToken.None); }
            catch { /* ignore close errors */ }
        }

        await Task.WhenAll(pingTask, recvTask);
    }

    private async Task ReceiveLoop(ClientWebSocket ws, CancellationToken token)
    {
        var buffer = new byte[_receiveBufferSize];
        var sb     = new StringBuilder();

        while (ws.State == WebSocketState.Open && !token.IsCancellationRequested)
        {
            sb.Clear();
            WebSocketReceiveResult result;

            do
            {
                result = await ws.ReceiveAsync(buffer, token);

                if (result.MessageType == WebSocketMessageType.Close)
                    return;

                sb.Append(Encoding.UTF8.GetString(buffer, 0, result.Count));
            }
            while (!result.EndOfMessage);

            var json = sb.ToString();
            ProcessMessage(json);
        }
    }

    private async Task PingLoop(ClientWebSocket ws, PeriodicTimer timer, CancellationToken token)
    {
        var pingJson = JsonSerializer.Serialize(new PingMessage());

        try
        {
            while (await timer.WaitForNextTickAsync(token))
            {
                if (ws.State != WebSocketState.Open) break;
                await SendJsonAsync(ws, pingJson, token);
            }
        }
        catch (OperationCanceledException) { }
    }

    private static async Task SendJsonAsync(ClientWebSocket ws, string json, CancellationToken token)
    {
        var bytes = Encoding.UTF8.GetBytes(json);
        await ws.SendAsync(bytes, WebSocketMessageType.Text, endOfMessage: true, token);
    }

    private void ProcessMessage(string json)
    {
        try
        {
            var frame = JsonSerializer.Deserialize<BrowFrame>(json, JsonOpts);
            if (frame is null) return;

            if (frame.Type == "brow" && frame.Outputs is not null)
            {
                _latestOutputs = frame.Outputs;
                _lastMode      = frame.Mode;
                Interlocked.Increment(ref _framesReceived);
            }
            // pong / reset_ack / recalibrate_ack / mode_ack messages are silently ignored
        }
        catch (JsonException)
        {
            // Malformed message — skip
        }
    }
}
