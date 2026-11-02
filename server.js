import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import Anthropic from "@anthropic-ai/sdk";
import express from "express";
import { randomUUID } from "crypto";
import { z } from "zod";
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

// ── Config ──────────────────────────────────────────────────────
const PORT          = parseInt(process.env.PORT || "3001", 10);
const AUTH_TOKEN    = process.env.AUTH_TOKEN;
const ANTHROPIC_KEY = process.env.ANTHROPIC_API_KEY;
const MODEL         = process.env.CLAUDE_MODEL || "claude-sonnet-4-20250514";

if (!ANTHROPIC_KEY) { console.error("ANTHROPIC_API_KEY is required."); process.exit(1); }
if (!AUTH_TOKEN)    { console.error("AUTH_TOKEN is required."); process.exit(1); }

// ── Load knowledge base ─────────────────────────────────────────
const __filename = fileURLToPath(import.meta.url);
const __dirname  = dirname(__filename);
const KB = JSON.parse(readFileSync(join(__dirname, "knowledge.json"), "utf-8"));

// Build compact profile for system prompt (~2KB)
const PROFILE_CONTEXT = `You are Claude, acting as an AI assistant for ${KB.profile.name} (${KB.profile.name_ru}), ${KB.profile.role} of ${KB.profile.company}.

KEY CONTEXT:
- Company: ${KB.profile.company_description}
- HQ: ${KB.profile.hq}
- Co-founder: ${KB.profile.co_founder}
- Background: ${KB.profile.background}
- Languages: ${KB.profile.languages}
- Locations: ${KB.profile.locations}
- Commercial terms: ${KB.profile.commercial_terms}
- Dubai entity: ${KB.profile.dubai_entity}
- Group structure: ${Object.values(KB.group_structure).join("; ")}

You have access to Roman's business knowledge base covering: ${Object.keys(KB.topics).join(", ")}. When answering, draw on this context to give informed, specific responses that reflect understanding of MetalMarket's business.`;

// Build keyword index for search
function buildIndex() {
  const index = [];
  for (const [topicId, topic] of Object.entries(KB.topics)) {
    index.push({
      id: topicId,
      keywords: topic.keywords.map(k => k.toLowerCase()),
      summary: topic.summary,
      details: topic.details || null,
    });
  }
  return index;
}
const TOPIC_INDEX = buildIndex();

// Search knowledge by query string — returns matching topic summaries
function searchKnowledge(query, maxResults = 5) {
  const terms = query.toLowerCase().split(/\s+/).filter(t => t.length > 1);
  const scored = TOPIC_INDEX.map(topic => {
    let score = 0;
    for (const term of terms) {
      // Exact keyword match = 10 points
      if (topic.keywords.some(k => k === term)) score += 10;
      // Partial keyword match = 5 points
      else if (topic.keywords.some(k => k.includes(term) || term.includes(k))) score += 5;
      // Summary mention = 2 points
      else if (topic.summary.toLowerCase().includes(term)) score += 2;
      // Details mention = 1 point
      else if (topic.details && topic.details.toLowerCase().includes(term)) score += 1;
    }
    return { ...topic, score };
  })
  .filter(t => t.score > 0)
  .sort((a, b) => b.score - a.score)
  .slice(0, maxResults);

  return scored;
}

// Format search results as readable text
function formatResults(results) {
  if (results.length === 0) return "No matching topics found in the knowledge base.";
  return results.map(r => {
    let text = `## ${r.id.replace(/_/g, " ").toUpperCase()} (relevance: ${r.score})\n${r.summary}`;
    if (r.details) text += `\n\nDetails: ${r.details}`;
    return text;
  }).join("\n\n---\n\n");
}

// Format people info
function formatPeople(query) {
  const terms = query.toLowerCase().split(/\s+/);
  const matches = Object.entries(KB.people).filter(([key, person]) => {
    const searchable = `${key} ${person.name} ${person.aliases?.join(" ") || ""} ${person.role || ""} ${person.relation || ""}`.toLowerCase();
    return terms.some(t => searchable.includes(t));
  });
  if (matches.length === 0) return null;
  return matches.map(([, p]) =>
    `**${p.name}**${p.aliases ? ` (aka ${p.aliases.join(", ")})` : ""}: ${p.role || p.relation || ""}`
  ).join("\n");
}

// ── Anthropic client ────────────────────────────────────────────
const anthropic = new Anthropic({ apiKey: ANTHROPIC_KEY });

// ── Build an MCP server instance ────────────────────────────────
function createServer() {
  const server = new McpServer({
    name: "claude-assistant",
    version: "2.0.0",
  });

  // Tool 1: General-purpose Claude (now with business context)
  server.tool(
    "ask_claude",
    "Send a question or task to Claude AI and get a detailed response. " +
    "Claude knows Roman's business context — MetalMarket, deals, contacts, pipelines. " +
    "Use for analysis, writing, research, summarisation, translation, or any task.",
    {
      prompt:  z.string().describe("The question or task for Claude"),
      context: z.string().optional().describe("Optional additional context"),
    },
    async ({ prompt, context }) => {
      // Auto-search knowledge base for relevant context
      const kbResults = searchKnowledge(prompt, 3);
      const peopleResults = formatPeople(prompt);

      let systemMsg = PROFILE_CONTEXT;
      if (kbResults.length > 0) {
        systemMsg += `\n\nRELEVANT BUSINESS CONTEXT:\n${formatResults(kbResults)}`;
      }
      if (peopleResults) {
        systemMsg += `\n\nRELEVANT PEOPLE:\n${peopleResults}`;
      }

      const userMsg = context
        ? `Context:\n${context}\n\nTask:\n${prompt}`
        : prompt;

      const res = await anthropic.messages.create({
        model: MODEL,
        max_tokens: 8192,
        system: systemMsg,
        messages: [{ role: "user", content: userMsg }],
      });

      const text = res.content
        .filter((b) => b.type === "text")
        .map((b) => b.text)
        .join("\n\n");
      return { content: [{ type: "text", text }] };
    }
  );

  // Tool 2: Coding assistant (with company context)
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
        `You are Claude Code, an expert software engineer working for ${KB.profile.company}. ` +
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

  // Tool 4: Search knowledge base
  server.tool(
    "search_knowledge",
    "Search Roman's business knowledge base for information about MetalMarket, " +
    "deals, contacts, pipelines, products, and company context. " +
    "Use this to find specific business details before asking Claude a question.",
    {
      query:       z.string().describe("Search query — keywords about the topic (e.g. 'GPU servers', 'European pipeline', 'casting forging')"),
      max_results: z.number().optional().describe("Max results to return (default 5)"),
    },
    async ({ query, max_results }) => {
      const results = searchKnowledge(query, max_results || 5);
      const peopleResults = formatPeople(query);

      let text = formatResults(results);
      if (peopleResults) {
        text += `\n\n---\n\nPEOPLE:\n${peopleResults}`;
      }

      return { content: [{ type: "text", text }] };
    }
  );

  // Tool 5: Get full profile
  server.tool(
    "get_profile",
    "Get Roman's full business profile — identity, company info, group structure, " +
    "and a list of all knowledge base topics available.",
    {},
    async () => {
      const topicList = Object.entries(KB.topics).map(([id, t]) =>
        `- ${id}: ${t.keywords.slice(0, 5).join(", ")}`
      ).join("\n");

      const peopleList = Object.values(KB.people).map(p =>
        `- ${p.name}: ${p.role || p.relation || ""}`
      ).join("\n");

      const text = `# Profile: ${KB.profile.name}
${KB.profile.role}, ${KB.profile.company}
Website: ${KB.profile.website}
Email: ${KB.profile.email}
HQ: ${KB.profile.hq}
Background: ${KB.profile.background}
Languages: ${KB.profile.languages}

## Group Structure
${Object.entries(KB.group_structure).map(([k, v]) => `- ${k}: ${v}`).join("\n")}

## Knowledge Base Topics
${topicList}

## Key People
${peopleList}`;

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
    version: "2.0.0",
    tools: ["ask_claude", "code_with_claude", "review_code", "search_knowledge", "get_profile"],
    knowledge_topics: Object.keys(KB.topics).length,
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

  if (sessionId && sessions.has(sessionId)) {
    const transport = sessions.get(sessionId);
    await transport.handleRequest(req, res, req.body);
    return;
  }

  const transport = new StreamableHTTPServerTransport({
    sessionIdGenerator: () => randomUUID(),
  });

  const mcpServer = createServer();
  await mcpServer.connect(transport);

  transport.onclose = () => {
    const sid = transport.sessionId;
    if (sid) sessions.delete(sid);
  };

  await transport.handleRequest(req, res, req.body);

  if (transport.sessionId) {
    sessions.set(transport.sessionId, transport);
  }
});

// GET /mcp — SSE stream
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
  console.log(`Claude MCP Server v2.0.0 running on port ${PORT}`);
  console.log(`Knowledge base: ${Object.keys(KB.topics).length} topics, ${Object.keys(KB.people).length} people`);
  console.log(`MCP endpoint: http://localhost:${PORT}/mcp`);
});
