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

// Department list for validation
const DEPT_IDS = Object.keys(KB.departments);
const DEPT_NAMES = Object.fromEntries(
  Object.entries(KB.departments).map(([id, d]) => [id, d.name])
);

// Build compact system prompt
const PROFILE_CONTEXT = `You are Claude, acting as the AI backbone for The Secretary — the intelligent assistant managing all departments for Spartan (${KB.profile.name}, ${KB.profile.role} of ${KB.profile.company}).

HIERARCHY:
- Spartan (CEO): ${KB.hierarchy.spartan.name} — overall boss
- The Secretary: manages all 8 departments, routes queries, retrieves documents

DEPARTMENTS:
${DEPT_IDS.map(id => `- ${KB.departments[id].name}: ${KB.departments[id].scope}`).join("\n")}

COMPANY: ${KB.profile.company_description}
HQ: ${KB.profile.hq}
Co-founder: ${KB.profile.co_founder}
Commercial terms: ${KB.profile.commercial_terms}
Group: ${Object.values(KB.group_structure).join("; ")}

When answering, identify which department(s) a query belongs to and draw on the relevant knowledge. The Analytics Department has cross-department access to all files.`;

// ── Search engine ───────────────────────────────────────────────

// Flatten all topics across departments into a searchable index
function buildIndex() {
  const index = [];
  for (const [deptId, dept] of Object.entries(KB.departments)) {
    for (const [topicId, topic] of Object.entries(dept.topics)) {
      index.push({
        department: deptId,
        departmentName: dept.name,
        topicId,
        keywords: topic.keywords.map(k => k.toLowerCase()),
        summary: topic.summary,
        details: topic.details || null,
      });
    }
  }
  return index;
}
const TOPIC_INDEX = buildIndex();

// Search by query, optionally filtered to a department
function searchKnowledge(query, department = null, maxResults = 5) {
  const terms = query.toLowerCase().split(/\s+/).filter(t => t.length > 1);

  let candidates = TOPIC_INDEX;
  if (department) {
    // Allow matching by id or name
    const deptLower = department.toLowerCase();
    candidates = TOPIC_INDEX.filter(t =>
      t.department === deptLower ||
      t.departmentName.toLowerCase().includes(deptLower)
    );
  }

  const scored = candidates.map(topic => {
    let score = 0;
    for (const term of terms) {
      if (topic.keywords.some(k => k === term)) score += 10;
      else if (topic.keywords.some(k => k.includes(term) || term.includes(k))) score += 5;
      else if (topic.summary.toLowerCase().includes(term)) score += 2;
      else if (topic.details && topic.details.toLowerCase().includes(term)) score += 1;
    }
    return { ...topic, score };
  })
  .filter(t => t.score > 0)
  .sort((a, b) => b.score - a.score)
  .slice(0, maxResults);

  return scored;
}

function formatResults(results) {
  if (results.length === 0) return "No matching topics found in the knowledge base.";
  return results.map(r => {
    let text = `## ${r.topicId.replace(/_/g, " ").toUpperCase()}\n**Department:** ${r.departmentName} | **Relevance:** ${r.score}\n\n${r.summary}`;
    if (r.details) text += `\n\nDetails: ${r.details}`;
    return text;
  }).join("\n\n---\n\n");
}

function formatPeople(query) {
  const terms = query.toLowerCase().split(/\s+/);
  const matches = Object.entries(KB.people).filter(([key, person]) => {
    const searchable = `${key} ${person.name} ${person.aliases?.join(" ") || ""} ${person.role || ""} ${person.relation || ""}`.toLowerCase();
    return terms.some(t => searchable.includes(t));
  });
  if (matches.length === 0) return null;
  return matches.map(([, p]) =>
    `**${p.name}**${p.aliases ? ` (aka ${p.aliases.join(", ")})` : ""}: ${p.role || p.relation || ""} [${DEPT_NAMES[p.department] || ""}]`
  ).join("\n");
}

// ── Anthropic client ────────────────────────────────────────────
const anthropic = new Anthropic({ apiKey: ANTHROPIC_KEY });

// ── MCP server ──────────────────────────────────────────────────
function createServer() {
  const server = new McpServer({
    name: "claude-assistant",
    version: "3.0.0",
  });

  // ─── Tool 1: ask_claude (with department-aware context) ───────
  server.tool(
    "ask_claude",
    "Send a question or task to Claude AI. Claude knows Spartan's full business context " +
    "organized by department (Finance, Electronics, Chemicals, Health, Accounts, R&D, IT, Analytics). " +
    "Relevant context is auto-injected. Use for analysis, writing, research, or any task.",
    {
      prompt:     z.string().describe("The question or task for Claude"),
      department: z.string().optional().describe("Route to a specific department: finance, electronics, chemicals, health, accounts, rd, it, analytics"),
      context:    z.string().optional().describe("Optional additional context"),
    },
    async ({ prompt, department, context }) => {
      const kbResults = searchKnowledge(prompt, department, 4);
      const peopleResults = formatPeople(prompt);

      let systemMsg = PROFILE_CONTEXT;
      if (kbResults.length > 0) {
        systemMsg += `\n\nRELEVANT CONTEXT FROM DEPARTMENTS:\n${formatResults(kbResults)}`;
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

  // ─── Tool 2: code_with_claude ─────────────────────────────────
  server.tool(
    "code_with_claude",
    "Send a coding task to Claude Code — IT Department. " +
    "Writes, reviews, refactors, or debugs code with complete solutions.",
    {
      task:          z.string().describe("What code to write, fix, or review"),
      language:      z.string().optional().describe("Programming language"),
      existing_code: z.string().optional().describe("Existing code to modify"),
      file_name:     z.string().optional().describe("Target file name"),
    },
    async ({ task, language, existing_code, file_name }) => {
      const sys =
        `You are Claude Code, expert software engineer for ${KB.profile.company} (IT Department). ` +
        "Write clean, production-ready, well-documented code. No placeholder TODOs. " +
        (language ? `Use ${language}. ` : "") +
        (file_name ? `File: ${file_name}. ` : "") +
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

  // ─── Tool 3: review_code ──────────────────────────────────────
  server.tool(
    "review_code",
    "Have Claude review code for bugs, security, performance, best practices — IT Department.",
    {
      code:        z.string().describe("The code to review"),
      language:    z.string().optional().describe("Programming language"),
      focus_areas: z.string().optional().describe("Focus on: bugs, security, performance, style, all"),
    },
    async ({ code, language, focus_areas }) => {
      const sys =
        "You are a senior code reviewer. Review for bugs, security, performance, " +
        "and best practices. For each issue: problem + concrete fix. " +
        (focus_areas ? `Focus: ${focus_areas}. ` : "") +
        "End with overall assessment and corrected code.";

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

  // ─── Tool 4: search_knowledge ─────────────────────────────────
  server.tool(
    "search_knowledge",
    "Search Spartan's business knowledge base by keyword. Optionally filter by department: " +
    "finance, electronics, chemicals, health, accounts, rd, it, analytics. " +
    "Analytics Department has access to all files across all departments.",
    {
      query:       z.string().describe("Search keywords (e.g. 'GPU servers', 'catalyst EFKO', 'wheat Suslov')"),
      department:  z.string().optional().describe("Filter by department: finance, electronics, chemicals, health, accounts, rd, it, analytics"),
      max_results: z.number().optional().describe("Max results (default 5)"),
    },
    async ({ query, department, max_results }) => {
      // Analytics sees everything
      const dept = department?.toLowerCase() === "analytics" ? null : department;
      const results = searchKnowledge(query, dept, max_results || 5);
      const peopleResults = formatPeople(query);

      let text = formatResults(results);
      if (peopleResults) {
        text += `\n\n---\n\nPEOPLE:\n${peopleResults}`;
      }
      return { content: [{ type: "text", text }] };
    }
  );

  // ─── Tool 5: get_profile ──────────────────────────────────────
  server.tool(
    "get_profile",
    "Get Spartan's full business profile, group structure, org chart with all 8 departments, and a directory of people.",
    {},
    async () => {
      const deptList = DEPT_IDS.map(id => {
        const d = KB.departments[id];
        const topicCount = Object.keys(d.topics).length;
        return `### ${d.name}\nScope: ${d.scope}\nTopics (${topicCount}): ${Object.keys(d.topics).join(", ")}`;
      }).join("\n\n");

      const peopleList = Object.values(KB.people).map(p =>
        `- **${p.name}**${p.aliases ? ` (${p.aliases.join(", ")})` : ""}: ${p.role || p.relation || ""} → ${DEPT_NAMES[p.department] || "unassigned"}`
      ).join("\n");

      const text = `# SPARTAN — ${KB.profile.name}
${KB.profile.role}, ${KB.profile.company}
Website: ${KB.profile.website} | Email: ${KB.profile.email}
HQ: ${KB.profile.hq}
Background: ${KB.profile.background}
Languages: ${KB.profile.languages}

## Hierarchy
- **Spartan** (CEO): ${KB.hierarchy.spartan.description}
- **The Secretary**: ${KB.hierarchy.secretary.description}

## Group Structure
${Object.entries(KB.group_structure).map(([k, v]) => `- ${v}`).join("\n")}

## Departments
${deptList}

## Key People
${peopleList}`;

      return { content: [{ type: "text", text }] };
    }
  );

  // ─── Tool 6: list_departments ─────────────────────────────────
  server.tool(
    "list_departments",
    "List all departments with their scope and topic count. Use to understand the org structure before searching.",
    {},
    async () => {
      const lines = DEPT_IDS.map(id => {
        const d = KB.departments[id];
        const topics = Object.keys(d.topics);
        return `**${d.name}** (${id})\nScope: ${d.scope}\nTopics: ${topics.join(", ")} (${topics.length})`;
      });

      const text = `# Organization — 8 Departments\n\nSpartan (CEO) → The Secretary → All Departments\n\n${lines.join("\n\n")}`;
      return { content: [{ type: "text", text }] };
    }
  );

  return server;
}

// ── Express app ─────────────────────────────────────────────────
const app = express();
app.use(express.json());

app.get("/", (_req, res) => {
  res.json({
    status: "ok",
    server: "claude-assistant",
    version: "3.0.0",
    hierarchy: { ceo: "Spartan", assistant: "The Secretary" },
    departments: DEPT_IDS.map(id => ({ id, name: KB.departments[id].name, topics: Object.keys(KB.departments[id].topics).length })),
    tools: ["ask_claude", "code_with_claude", "review_code", "search_knowledge", "get_profile", "list_departments"],
    transport: "Streamable HTTP",
  });
});

app.use("/mcp", (req, res, next) => {
  const auth = req.headers.authorization;
  if (auth !== `Bearer ${AUTH_TOKEN}`) {
    return res.status(401).json({ error: "Unauthorized" });
  }
  next();
});

const sessions = new Map();

app.post("/mcp", async (req, res) => {
  const sessionId = req.headers["mcp-session-id"];
  if (sessionId && sessions.has(sessionId)) {
    const transport = sessions.get(sessionId);
    await transport.handleRequest(req, res, req.body);
    return;
  }
  const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: () => randomUUID() });
  const mcpServer = createServer();
  await mcpServer.connect(transport);
  transport.onclose = () => { const sid = transport.sessionId; if (sid) sessions.delete(sid); };
  await transport.handleRequest(req, res, req.body);
  if (transport.sessionId) { sessions.set(transport.sessionId, transport); }
});

app.get("/mcp", async (req, res) => {
  const sessionId = req.headers["mcp-session-id"];
  if (!sessionId || !sessions.has(sessionId)) { res.status(400).json({ error: "No active session. POST /mcp first." }); return; }
  const transport = sessions.get(sessionId);
  await transport.handleRequest(req, res);
});

app.delete("/mcp", async (req, res) => {
  const sessionId = req.headers["mcp-session-id"];
  if (!sessionId || !sessions.has(sessionId)) { res.status(400).json({ error: "No active session." }); return; }
  const transport = sessions.get(sessionId);
  await transport.close();
  sessions.delete(sessionId);
  res.status(200).json({ status: "closed" });
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Claude MCP Server v3.0.0 — Spartan Command`);
  console.log(`Departments: ${DEPT_IDS.length} | Topics: ${TOPIC_INDEX.length} | People: ${Object.keys(KB.people).length}`);
  console.log(`MCP endpoint: http://localhost:${PORT}/mcp`);
});
