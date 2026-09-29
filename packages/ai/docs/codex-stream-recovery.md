# Codex stream recovery

Codex requests use a five minute inactivity limit for both WebSocket and SSE.
The limit starts before connection setup and resets when a parsed `response.*`
event arrives. It covers waiting for connection establishment, HTTP headers,
and response events. Transport keepalives and vendor telemetry do not reset it.
There is no total duration limit while response events continue arriving.

Set `timeoutMs` in Codex stream options to choose a different inactivity limit.
Prime Agent forwards the existing `retry.provider.timeoutMs` setting to this
option. Positive finite values up to 2^31 - 1 milliseconds are accepted; omitted
or invalid values use the default. A longer limit can accommodate workloads
with unusually long silent reasoning pauses. Silence alone does not establish
that the remote model stopped computing.

On timeout, the transport is cancelled and the result reports an error named
`CodexStreamTimeoutError` with code `ETIMEDOUT`. It enters the existing provider
retry policy. A WebSocket failure before visible output may fall back to SSE;
a failure after output starts ends that attempt without replaying its output.
Caller cancellation continues to report an aborted request.

The `provider_stream_timeout` diagnostic records the transport, configured
limit, request start, last provider event, last content delta, last event type,
and response ID when available. It does not store content. Failed WebSocket
connections and their continuation state are removed from the session cache.
The separate five minute cache expiry still applies only to idle connections.

The inactivity limit applies to each transport attempt. Retries and fallback
can make total recovery take longer than this limit, and the existing recovery
budget is checked between attempts.
