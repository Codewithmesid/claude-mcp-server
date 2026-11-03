// MCP endpoint — stateless.
//
// The Render build kept a Map of live transports keyed by session id. That
// cannot work on serverless: the next request may land on a different
// instance and find an empty Map. So each request builds its own server and
// transport, handles the one message, and tears them down. The MCP spec
// supports this: with sessionIdGenerator undefined the transport issues no
// session id, and clients treat every POST as self-contained.
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { createServer, ready, AUTH_TOKEN } from "../lib/core.js";

export const config = { maxDuration: 300 };

export default async function handler(req, res) {
  // ── auth ──────────────────────────────────────────────────────
  const auth = req.headers["authorization"] || "";
  if (!AUTH_TOKEN || auth !== `Bearer ${AUTH_TOKEN}`) {
    res.status(401).json({
      jsonrpc: "2.0",
      error: { code: -32001, message: "Unauthorized" },
      id: null,
    });
    return;
  }

  // GET and DELETE exist in the spec for session-based servers. Stateless has
  // no session to resume or end, so answer plainly instead of 404ing.
  if (req.method === "GET" || req.method === "DELETE") {
    res.status(405).json({
      jsonrpc: "2.0",
      error: {
        code: -32000,
        message: "This server is stateless; use POST for every request. " +
                 "No session id is issued and none is required.",
      },
      id: null,
    });
    return;
  }

  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  let transport;
  let server;
  try {
    await ready();                      // departments, hierarchy, profile

    server = createServer();
    transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: undefined,    // ← stateless
      enableJsonResponse: true,         // plain JSON rather than an SSE frame
    });

    // free the pair when the connection ends, however it ends
    res.on("close", () => {
      transport?.close?.();
      server?.close?.();
    });

    await server.connect(transport);
    await transport.handleRequest(req, res, req.body);
  } catch (err) {
    console.error("MCP handler failed:", err);
    if (!res.headersSent) {
      res.status(500).json({
        jsonrpc: "2.0",
        error: { code: -32603, message: `Internal error: ${err?.message || err}` },
        id: null,
      });
    }
  }
}
