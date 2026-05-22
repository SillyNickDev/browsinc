using System.Reflection;
using Microsoft.Extensions.Logging;
using VRCFaceTracking;
using VRCFaceTracking.Core.Library;
using VRCFaceTracking.Core.Params.Expressions;

namespace BrowSyncModule;

/// <summary>
/// BrowSync VRCFT Module — inferred eyebrow tracking for users without a Quest Pro.
///
/// Architecture:
///   1. This module connects as a WebSocket CLIENT to the BrowSync Python server
///      (ws://localhost:7720 by default).
///   2. The Python server receives eye/face data from the user's existing trackers,
///      runs the rule-based + GRU inference pipeline, and pushes brow AU estimates
///      back over the WebSocket.
///   3. This module writes those estimates to UnifiedTracking.Data.Shapes, which
///      VRCFT then forwards to VRChat over OSC as normal.
///
/// The module only handles brow expressions — it intentionally does not claim or
/// overwrite eye or lower-face expressions, leaving those to the user's existing
/// tracker modules.
/// </summary>
public class BrowSyncModule : ExtTrackingModule
{
    // -- Module metadata -----------------------------------------------------

    public override (bool SupportsEye, bool SupportsExpression) Supported => (false, true);
    // Eye: false  — we don't provide eye tracking, existing modules handle that
    // Expression: true — we provide brow expression data only

    // Static logger reference so BrowSyncClient can reach it without DI
    internal static ILogger? Log;

    // -- Configuration -------------------------------------------------------

    private const string DefaultHost = "localhost";
    private const int    DefaultPort = 7720;

    // How long to wait for initial connection before declaring init failed
    private static readonly TimeSpan InitTimeout = TimeSpan.FromSeconds(8);

    // -- State ---------------------------------------------------------------

    private BrowSyncClient? _client;
    private Thread?         _updateThread;
    private CancellationTokenSource? _updateCts;

    // Frame counter for VRCFT status display
    private ulong _localFrameCount;
    private BrowOutputs? _lastApplied;

    // ---------------------------------------------------------------------------
    // Lifecycle
    // ---------------------------------------------------------------------------

    public override (bool eyeSuccess, bool expressionSuccess) Initialize(
        bool eyeAvailable,
        bool expressionAvailable)
    {
        Log = Logger;
        Log?.LogInformation("[BrowSync] Initializing BrowSync module v0.1.0");

        // Set module display info
        ModuleInformation.Name = "BrowSync — Inferred Brow Tracking";

        // Embed logo
        var logoStream = GetType()
            .Assembly
            .GetManifestResourceStream("BrowSyncModule.Assets.browsync_logo.png");

        if (logoStream is not null)
            ModuleInformation.StaticImages = new List<Stream> { logoStream };

        // Start the WebSocket client
        _client = new BrowSyncClient(DefaultHost, DefaultPort);
        _client.Start();

        // Wait briefly for initial connection — if it fails we still load,
        // the module will just keep trying to reconnect in the background.
        var deadline = DateTime.UtcNow + InitTimeout;
        while (DateTime.UtcNow < deadline)
        {
            if (_client.State == WsConnectionState.Connected) break;
            Thread.Sleep(200);
        }

        bool connected = _client.State == WsConnectionState.Connected;

        if (connected)
            Log?.LogInformation("[BrowSync] Connected to Python inference server.");
        else
            Log?.LogWarning("[BrowSync] Could not connect to BrowSync server at startup. " +
                            $"Make sure browsync/ws_server/server.py is running on port {DefaultPort}. " +
                            "The module will keep retrying in the background.");

        // Always return true for expression — we provide valid (zero) values
        // even when disconnected, and reconnect automatically.
        return (eyeSuccess: false, expressionSuccess: true);
    }

    public override void Update()
    {
        // VRCFT calls this on a dedicated thread at its own cadence (~100Hz).
        // We just apply whatever the latest frame from the WS client is.

        if (Status != ModuleState.Active)
        {
            Thread.Sleep(10);
            return;
        }

        var outputs = _client?.LatestOutputs;

        if (outputs is not null)
        {
            ApplyToUnifiedTracking(outputs);
            _lastApplied = outputs;
            _localFrameCount++;
        }
        else
        {
            // No data yet (disconnected or waiting for first frame) — zero brows
            // so they don't get stuck in a non-neutral position
            ZeroBrows();
        }

        // Throttle: ~90fps ceiling matches the Python server's target rate
        Thread.Sleep(11);
    }

    public override void Teardown()
    {
        Log?.LogInformation("[BrowSync] Tearing down.");
        ZeroBrows();

        _client?.Stop();
        _client = null;
    }

    // ---------------------------------------------------------------------------
    // UnifiedTracking writes
    // ---------------------------------------------------------------------------

    /// <summary>
    /// Writes BrowSync output to UnifiedTracking.Data.Shapes.
    /// Only writes brow-related expressions — leaves all other shapes untouched.
    /// </summary>
    private static void ApplyToUnifiedTracking(BrowOutputs o)
    {
        var shapes = UnifiedTracking.Data.Shapes;

        // Inner brow — left and right separate in Unified Expressions
        shapes[(int)UnifiedExpressions.BrowInnerUpLeft].Weight  = Clamp(o.BrowInnerUpLeft);
        shapes[(int)UnifiedExpressions.BrowInnerUpRight].Weight = Clamp(o.BrowInnerUpRight);

        // Outer brow
        shapes[(int)UnifiedExpressions.BrowOuterUpLeft].Weight  = Clamp(o.BrowOuterUpLeft);
        shapes[(int)UnifiedExpressions.BrowOuterUpRight].Weight = Clamp(o.BrowOuterUpRight);

        // Lowerer (furrowed / pressed down)
        shapes[(int)UnifiedExpressions.BrowLowererLeft].Weight  = Clamp(o.BrowLowererLeft);
        shapes[(int)UnifiedExpressions.BrowLowererRight].Weight = Clamp(o.BrowLowererRight);

        // Pinch (inner scrunch)
        shapes[(int)UnifiedExpressions.BrowPinchLeft].Weight    = Clamp(o.BrowPinchLeft);
        shapes[(int)UnifiedExpressions.BrowPinchRight].Weight   = Clamp(o.BrowPinchRight);
    }

    private static void ZeroBrows()
    {
        var shapes = UnifiedTracking.Data.Shapes;
        shapes[(int)UnifiedExpressions.BrowInnerUpLeft].Weight  = 0f;
        shapes[(int)UnifiedExpressions.BrowInnerUpRight].Weight = 0f;
        shapes[(int)UnifiedExpressions.BrowOuterUpLeft].Weight  = 0f;
        shapes[(int)UnifiedExpressions.BrowOuterUpRight].Weight = 0f;
        shapes[(int)UnifiedExpressions.BrowLowererLeft].Weight  = 0f;
        shapes[(int)UnifiedExpressions.BrowLowererRight].Weight = 0f;
        shapes[(int)UnifiedExpressions.BrowPinchLeft].Weight    = 0f;
        shapes[(int)UnifiedExpressions.BrowPinchRight].Weight   = 0f;
    }

    private static float Clamp(float v) => Math.Clamp(v, 0f, 1f);
}
