import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import Anthropic from "@anthropic-ai/sdk";
import express from "express";
import { randomUUID } from "crypto";
import { z } from "zod";

// ── Config ──────────────────────────────────────────────────────
const PORT          = parseInt(process.env.PORT || "3001", 10);
const AUTH_TOKEN    = process.env.AUTH_TOKEN;
const ANTHROPIC_KEY = process.env.ANTHROPIC_API_KEY;
const MODEL         = process.env.CLAUDE_MODEL || "claude-sonnet-4-20250514";

if (!ANTHROPIC_KEY) {
  console.error("ANTHROPIC_API_KEY is required.");
  process.exit(1);
}
if (!AUTH_TOKEN) {
  console.error("AUTH_TOKEN is required. Pick any secret string.");
  process.exit(1);
}

// ── Anthropic client ────────────────────────────────────────────
const anthropic = new Anthropic({ apiKey: ANTHROPIC_KEY });

// ── Build an MCP server instance ────────────────────────────────
function createServer() {
  const server = new McpServer({
    name: "claude-assistant",
    version: "1.0.0",
  });

  // Tool 1: General-purpose Claude
  server.tool(
    "ask_claude",
    "Send a question or task to Claude AI and get a detailed response. " +
    "Use when you need help with analysis, writing, research, summarisation, " +
    "translation, or any task you cannot complete yourself.",
    {
      prompt:  z.string().describe("The question or task for Claude"),
      context: z.string().optional().describe("Optional background info or context"),
    },
    async ({ prompt, context }) => {
      const userMsg = context
        ? `Context:\n${context}\n\nTask:\n${prompt}`
        : prompt;

      const res = await anthropic.messages.create({
        model: MODEL,
        max_tokens: 8192,
        messages: [{ role: "user", content: userMsg }],
      });

      const text = res.content
        .filter((b) => b.type === "text")
        .map((b) => b.text)
        .join("\n\n");
      return { content: [{ type: "text", text }] };
    }
  );

  // Tool 2: Coding assistant
  server.tool(
    "code_with_claude",
    "Send a coding task to Claude Code. Claude will write, review, refactor, " +
    "or debug code and return the complete solution with explanations.",
    {
      task:          z.string().describe("What code to write, fix, or review"),
      language:      z.string().optional().describe("Programming language (python, javascript, etc.)"),
      existing_code: z.string().optional().describe("Existing code to modify or build upon"),
      file_name:     z.string().optional().describe("Target file name for the output"),
    },
    async ({ task, language, existing_code, file_name }) => {
      const sys =
        "You are Claude Code, an expert software engineer. " +
        "Write clean, production-ready, well-documented code. " +
        "Always provide complete, working code — no placeholder TODOs. " +
        (language ? `Use ${language}. ` : "") +
        (file_name ? `The file is ${file_name}. ` : "") +
        "Structure: 1) brief explanation, 2) complete code in a fenced block, 3) how to run it.";

      const userMsg = existing_code
        ? `Existing code:\n\`\`\`\n${existing_code}\n\`\`\`\n\nTask: ${task}`
        : task;

      const res = await anthropic.messages.create({
        model: MODEL,
        max_tokens: 16384,
        system: sys,
        messages: [{ role: "user", content: userMsg }],
      });

      const text = res.content
        .filter((b) => b.type === "text")
        .map((b) => b.text)
        .join("\n\n");
      return { content: [{ type: "text", text }] };
    }
  );

  // Tool 3: Code review
  server.tool(
    "review_code",
    "Have Claude review code for bugs, security issues, performance, " +
    "and best practices. Returns a detailed review with suggested fixes.",
    {
      code:        z.string().describe("The code to review"),
      language:    z.string().optional().describe("Programming language"),
      focus_areas: z.string().optional().describe("Focus on: bugs, security, performance, style, all"),
    },
    async ({ code, language, focus_areas }) => {
      const sys =
        "You are a senior code reviewer. Review for bugs, security, performance, " +
        "and best practices. For each issue give the problem and a concrete fix. " +
        (focus_areas ? `Focus especially on: ${focus_areas}. ` : "") +
        "End with an overall assessment and the corrected code.";

      const userMsg = `Review this ${language || ""} code:\n\`\`\`${language || ""}\n${code}\n\`\`\``;

      const res = await anthropic.messages.create({
        model: MODEL,
        max_tokens: 8192,
        system: sys,
        messages: [{ role: "user", content: userMsg }],
      });

      const text = res.content
        .filter((b) => b.type === "text")
        .map((b) => b.text)
        .join("\n\n");
      return { content: [{ type: "text", text }] };
    }
  );

  return server;
}

// ── Express app ─────────────────────────────────────────────────
const app = express();
app.use(express.json());

// Health check (no auth needed)
app.get("/", (_req, res) => {
  res.json({
    status: "ok",
    server: "claude-assistant",
    tools: ["ask_claude", "code_with_claude", "review_code"],
    transport: "Streamable HTTP",
  });
});

// Auth middleware for /mcp
app.use("/mcp", (req, res, next) => {
  const auth = req.headers.authorization;
  if (auth !== `Bearer ${AUTH_TOKEN}`) {
    return res.status(401).json({ error: "Unauthorized" });
  }
  next();
});

// Session storage
const sessions = new Map();

// POST /mcp — handles all MCP JSON-RPC requests
app.post("/mcp", async (req, res) => {
  const sessionId = req.headers["mcp-session-id"];

  // Existing session
  if (sessionId && sessions.has(sessionId)) {
    const transport = sessions.get(sessionId);
    await transport.handleRequest(req, res, req.body);
    return;
  }

  // New session (initialize request)
  const transport = new StreamableHTTPServerTransport({
    sessionIdGenerator: () => randomUUID(),
  });

  const mcpServer = createServer();
  await mcpServer.connect(transport);

  // Clean up on close
  transport.onclose = () => {
    const sid = transport.sessionId;
    if (sid) sessions.delete(sid);
  };

  await transport.handleRequest(req, res, req.body);

  // Store after handling (sessionId is set during initialize)
  if (transport.sessionId) {
    sessions.set(transport.sessionId, transport);
  }
});

// GET /mcp — SSE stream for server-initiated messages
app.get("/mcp", async (req, res) => {
  const sessionId = req.headers["mcp-session-id"];
  if (!sessionId || !sessions.has(sessionId)) {
    res.status(400).json({ error: "No active session. POST /mcp first." });
    return;
  }
  const transport = sessions.get(sessionId);
  await transport.handleRequest(req, res);
});

// DELETE /mcp — close session
app.delete("/mcp", async (req, res) => {
  const sessionId = req.headers["mcp-session-id"];
  if (!sessionId || !sessions.has(sessionId)) {
    res.status(400).json({ error: "No active session." });
    return;
  }
  const transport = sessions.get(sessionId);
  await transport.close();
  sessions.delete(sessionId);
  res.status(200).json({ status: "closed" });
});

// ── Start ───────────────────────────────────────────────────────
app.listen(PORT, "0.0.0.0", () => {
  console.log(`Claude MCP Server running on port ${PORT}`);
  console.log(`MCP endpoint: http://localhost:${PORT}/mcp`);
  console.log(`Health check: http://localhost:${PORT}/`);
});
