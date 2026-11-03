import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import Anthropic from "@anthropic-ai/sdk";
import { z } from "zod";

// ── Config ──────────────────────────────────────────────────────
const AUTH_TOKEN    = process.env.AUTH_TOKEN;
const ANTHROPIC_KEY = process.env.ANTHROPIC_API_KEY;
// Model ids change over time. Ask the API which models this key can actually
// use rather than pinning a name that quietly 404s years later.
const MODEL_PREFERENCE = [/sonnet/i, /haiku/i, /opus/i];
let RESOLVED_MODEL = null;
let AVAILABLE_MODELS = [];
const SUPABASE_URL  = process.env.SUPABASE_URL;
const SUPABASE_KEY  = process.env.SUPABASE_SERVICE_KEY;
const VERSION       = "6.0.0";
const TOOL_NAMES    = ["ask_claude","code_with_claude","review_code","search_knowledge","get_profile","list_departments","add_knowledge","update_context","search_records","get_record","get_file_url","archive_stats","search_conversations","enrich_pending"];

if (!ANTHROPIC_KEY) { console.error("ANTHROPIC_API_KEY is required."); process.exit(1); }
if (!AUTH_TOKEN)    { console.error("AUTH_TOKEN is required."); process.exit(1); }
if (!SUPABASE_URL)  { console.error("SUPABASE_URL is required."); process.exit(1); }
if (!SUPABASE_KEY)  { console.error("SUPABASE_SERVICE_KEY is required."); process.exit(1); }

// ── Supabase REST client (native fetch, no extra deps) ─────────
const SB_REST = `${SUPABASE_URL}/rest/v1`;
const SB_HEADERS = {
  apikey: SUPABASE_KEY,
  Authorization: `Bearer ${SUPABASE_KEY}`,
  "Content-Type": "application/json",
  Prefer: "return=representation",
};

async function sbQuery(table, params = "") {
  const url = `${SB_REST}/${table}${params ? "?" + params : ""}`;
  const res = await fetch(url, { headers: SB_HEADERS });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Supabase ${table} query failed (${res.status}): ${err}`);
  }
  return res.json();
}

// ── Cached static data (loaded once at startup) ────────────────
let DEPARTMENTS = {};
let DEPT_IDS = [];
let DEPT_NAMES = {};
let HIERARCHY = {};
let PROFILE = {};
let GROUP_STRUCTURE = {};
let PROFILE_CONTEXT = "";

async function loadStaticData() {
  // Load departments
  const depts = await sbQuery("departments", "order=id");
  DEPARTMENTS = {};
  for (const d of depts) {
    DEPARTMENTS[d.id] = { name: d.name, scope: d.scope };
  }
  DEPT_IDS = Object.keys(DEPARTMENTS);
  DEPT_NAMES = Object.fromEntries(DEPT_IDS.map(id => [id, DEPARTMENTS[id].name]));

  // Load hierarchy
  const hier = await sbQuery("hierarchy");
  HIERARCHY = {};
  for (const h of hier) { HIERARCHY[h.id] = h; }

  // Load profile
  const profiles = await sbQuery("profile", "id=eq.main");
  PROFILE = profiles.length > 0 ? profiles[0].data : {};

  // Load group structure
  const groups = await sbQuery("group_structure");
  GROUP_STRUCTURE = {};
  for (const g of groups) { GROUP_STRUCTURE[g.id] = g.entity; }

  // Build system prompt
  PROFILE_CONTEXT = `You are Claude, acting as the AI backbone for The Secretary — the intelligent assistant managing all departments for Spartan (${PROFILE.name || "the boss"}, ${PROFILE.role || "CEO"} of ${PROFILE.company || "the company"}).

HIERARCHY:
- Spartan (CEO): ${HIERARCHY.spartan?.name || "Boss"} — overall boss
- The Secretary: manages all 8 departments, routes queries, retrieves documents

DEPARTMENTS:
${DEPT_IDS.map(id => `- ${DEPARTMENTS[id].name}: ${DEPARTMENTS[id].scope}`).join("\n")}

COMPANY: ${PROFILE.company_description || ""}
HQ: ${PROFILE.hq || ""}
Commercial terms: ${PROFILE.commercial_terms || ""}
Group: ${Object.values(GROUP_STRUCTURE).join("; ")}

When answering, identify which department(s) a query belongs to and draw on the relevant knowledge. The Analytics Department has cross-department access to all files.`;

  console.log(`Loaded: ${DEPT_IDS.length} departments, ${Object.keys(HIERARCHY).length} hierarchy entries, profile OK`);
}

// ── Search engine (queries Supabase live) ───────────────────────

async function searchKnowledge(query, department = null, maxResults = 5) {
  const terms = query.toLowerCase().split(/\s+/).filter(t => t.length > 1);

  // Build PostgREST full-text search query
  const ftsQuery = terms.join(" & ");

  let params = `select=id,department_id,keywords,summary,details&order=id&limit=${maxResults * 3}`;

  // Department filter
  if (department) {
    const deptLower = department.toLowerCase();
    // Match by id or name
    const matchedDept = DEPT_IDS.find(id =>
      id === deptLower || DEPARTMENTS[id]?.name.toLowerCase().includes(deptLower)
    );
    if (matchedDept) {
      params += `&department_id=eq.${matchedDept}`;
    }
  }

  // Try full-text search first
  let results = [];
  try {
    const ftsResults = await sbQuery("topics", `${params}&fts=fts.${encodeURIComponent(ftsQuery)}`);
    results = ftsResults;
  } catch {
    // FTS failed, fall back to keyword matching
  }

  // If FTS returned nothing, fall back to keyword array overlap
  if (results.length === 0) {
    try {
      const kwResults = await sbQuery("topics", `${params}&keywords=ov.{${terms.join(",")}}`);
      results = kwResults;
    } catch {
      // keyword overlap also failed
    }
  }

  // If still nothing, do a broad text search on summary
  if (results.length === 0 && terms.length > 0) {
    try {
      const ilikeResults = await sbQuery("topics", `${params}&or=(${terms.map(t => `summary.ilike.*${t}*`).join(",")})`);
      results = ilikeResults;
    } catch {
      // all search methods failed
    }
  }

  // Score results locally for ranking
  const scored = results.map(topic => {
    let score = 0;
    const kw = (topic.keywords || []).map(k => k.toLowerCase());
    for (const term of terms) {
      if (kw.some(k => k === term)) score += 10;
      else if (kw.some(k => k.includes(term) || term.includes(k))) score += 5;
      else if (topic.summary?.toLowerCase().includes(term)) score += 2;
      else if (topic.details?.toLowerCase().includes(term)) score += 1;
    }
    return {
      topicId: topic.id,
      department: topic.department_id,
      departmentName: DEPT_NAMES[topic.department_id] || topic.department_id,
      keywords: kw,
      summary: topic.summary,
      details: topic.details || null,
      score,
    };
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

async function searchPeople(query) {
  const terms = query.toLowerCase().split(/\s+/);

  // Search people by name, aliases, role
  let people = [];
  try {
    const orClauses = terms.map(t => `name.ilike.*${t}*,role.ilike.*${t}*`).join(",");
    people = await sbQuery("people", `or=(${orClauses})&limit=10`);
  } catch {
    // people search failed
  }

  if (people.length === 0) return null;
  return people.map(p =>
    `**${p.name}**${p.aliases?.length ? ` (aka ${p.aliases.join(", ")})` : ""}: ${p.role || p.relation || ""} [${DEPT_NAMES[p.department_id] || ""}]`
  ).join("\n");
}

// ── Document records (Shared Records archive, 3,577 rows) ──────
// Ranked full-text search runs server-side in Postgres via the
// search_records() function so relevance ordering is real, not guessed.
async function searchRecords({ query, department, area, docType, filesOnly, limit, loose }) {
  const res = await fetch(`${SB_REST}/rpc/search_records`, {
    method: "POST",
    headers: SB_HEADERS,
    body: JSON.stringify({
      q: query || null,
      dept: department || null,
      area_f: area || null,
      type_f: docType || null,
      files_only: !!filesOnly,
      lim: Math.min(Math.max(parseInt(limit || 10, 10), 1), 50),
      loose: !!loose,
    }),
  });
  if (!res.ok) throw new Error(`search_records failed (${res.status}): ${await res.text()}`);
  return res.json();
}

function formatRecords(rows) {
  if (!rows || rows.length === 0) return "No matching records.";
  return rows.map(r => {
    const bits = [
      `**${r.record_id}** — ${r.title || r.file_name || "(untitled)"}`,
      `Department: ${DEPT_NAMES[r.department_id] || r.department_id || "—"}` +
        (r.subfolder ? ` / ${r.subfolder}` : "") +
        (r.area === "personal" ? "  [personal]" : ""),
    ];
    if (r.doc_type) bits.push(`Type: ${r.doc_type}`);
    if (r.doc_date) bits.push(`Date: ${r.doc_date}`);
    if (r.counterparty) bits.push(`Counterparty: ${r.counterparty}`);
    if (r.doc_no) bits.push(`Document no: ${r.doc_no}`);
    if (r.amount) bits.push(`Amount: ${r.amount} ${r.currency || ""}`.trim());
    if (r.summary) bits.push(`\n${r.summary}`);
    if (r.key_fields) bits.push(`Key fields: ${r.key_fields}`);
    bits.push(
      r.storage_path
        ? `File: available — call get_file_url with record_id ${r.record_id}`
        : `File: not in the archive (metadata only)`
    );
    return bits.join("\n");
  }).join("\n\n---\n\n");
}

// ── Conversations (past Claude chats, synced every 4 hours) ────
// Goes through an RPC so punctuation in a question never collides with
// PostgREST's filter grammar, where commas are delimiters.
async function searchConversations(query, limit = 5, opts = {}) {
  const res = await fetch(`${SB_REST}/rpc/search_conversations`, {
    method: "POST",
    headers: SB_HEADERS,
    body: JSON.stringify({
      q: query || null,
      cat: opts.category || null,
      lim: Math.min(Math.max(parseInt(limit || 5, 10), 1), 25),
      loose: !!opts.loose,
    }),
  });
  if (!res.ok) throw new Error(`search_conversations failed (${res.status}): ${await res.text()}`);
  return res.json();
}

// Compact renderings for injection into a system prompt — full detail is a
// separate tool call away, so keep these short.
function formatDocsBrief(rows) {
  return rows.map(r => {
    const bits = [`• ${r.record_id} — ${r.title || r.file_name || "(untitled)"}`];
    const meta = [
      DEPT_NAMES[r.department_id] || r.department_id,
      r.subfolder, r.doc_type, r.doc_date,
      r.counterparty, r.doc_no,
      r.amount ? `${r.amount} ${r.currency || ""}`.trim() : null,
      r.storage_path ? "file available" : "metadata only",
    ].filter(Boolean).join(" | ");
    if (meta) bits.push(`  ${meta}`);
    if (r.summary) bits.push(`  ${String(r.summary).slice(0, 320)}`);
    if (r.key_fields) bits.push(`  Key: ${String(r.key_fields).slice(0, 200)}`);
    return bits.join("\n");
  }).join("\n");
}

function formatConvsBrief(rows) {
  return rows.map(c => {
    const bits = [`• ${c.title || c.chat_id}${c.category ? ` [${c.category}]` : ""}`];
    if (c.summary)    bits.push(`  ${String(c.summary).slice(0, 320)}`);
    if (c.key_points) bits.push(`  Points: ${String(c.key_points).slice(0, 300)}`);
    if (c.artifacts)  bits.push(`  Files discussed: ${String(c.artifacts).slice(0, 160)}`);
    return bits.join("\n");
  }).join("\n");
}

// Mint a time-limited download link for a stored document.
async function signedUrl(storagePath, expiresIn = 3600) {
  const res = await fetch(
    `${SUPABASE_URL}/storage/v1/object/sign/records/${storagePath.split("/").map(encodeURIComponent).join("/")}`,
    {
      method: "POST",
      headers: SB_HEADERS,
      body: JSON.stringify({ expiresIn }),
    }
  );
  if (!res.ok) throw new Error(`sign failed (${res.status}): ${await res.text()}`);
  const { signedURL } = await res.json();
  return `${SUPABASE_URL}/storage/v1${signedURL}`;
}

// ── Anthropic client ────────────────────────────────────────────
const anthropic = new Anthropic({ apiKey: ANTHROPIC_KEY });

// Ask the API what exists, then pick by preference. Cached after the first call.
async function listModels() {
  const r = await fetch("https://api.anthropic.com/v1/models?limit=100", {
    headers: { "x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01" },
  });
  if (!r.ok) throw new Error(`models list failed (${r.status}): ${(await r.text()).slice(0, 200)}`);
  const body = await r.json();
  return (body.data || []).map(m => m.id);
}

async function resolveModel(force = false) {
  if (RESOLVED_MODEL && !force) return RESOLVED_MODEL;
  if (process.env.CLAUDE_MODEL && !force) {
    RESOLVED_MODEL = process.env.CLAUDE_MODEL;
    return RESOLVED_MODEL;
  }
  AVAILABLE_MODELS = await listModels();
  if (!AVAILABLE_MODELS.length) throw new Error("API returned no models for this key");
  // the list comes back newest-first; take the first match by preference
  for (const rx of MODEL_PREFERENCE) {
    const hit = AVAILABLE_MODELS.find(id => rx.test(id));
    if (hit) { RESOLVED_MODEL = hit; break; }
  }
  if (!RESOLVED_MODEL) RESOLVED_MODEL = AVAILABLE_MODELS[0];
  console.log(`Model resolved to: ${RESOLVED_MODEL} (of ${AVAILABLE_MODELS.length} available)`);
  return RESOLVED_MODEL;
}

async function createMessage(params) {
  let model = await resolveModel();
  try {
    return await anthropic.messages.create({ ...params, model });
  } catch (e) {
    const missing = e?.status === 404 || /not_found_error/i.test(String(e?.message || e));
    if (!missing) throw e;
    // the chosen model vanished — re-discover once and retry
    model = await resolveModel(true);
    return await anthropic.messages.create({ ...params, model });
  }
}


// ── MCP server ──────────────────────────────────────────────────
function createServer() {
  const server = new McpServer({
    name: "claude-assistant",
    version: VERSION,
  });

  // ─── Tool 1: ask_claude (with department-aware context) ───────
  server.tool(
    "ask_claude",
    "Send a question or task to Claude AI. Claude knows Spartan's full business context " +
    "organized by department (Finance, Electronics, Chemicals, Health, Accounts, R&D, IT, Analytics). " +
    "Relevant context is auto-injected from the Supabase knowledge base. Use for analysis, writing, research, or any task.",
    {
      prompt:     z.string().describe("The question or task for Claude"),
      department: z.string().optional().describe("Route to a specific department: finance, electronics, chemicals, health, accounts, rd, it, analytics"),
      context:    z.string().optional().describe("Optional additional context"),
    },
    async ({ prompt, department, context }) => {
      // Pull from every source in parallel: the curated topics, the people
      // directory, the document archive, and past conversations. Without the
      // last two, Claude would answer from 33 summaries while 3,500+ indexed
      // documents sat unused.
      const [kbResults, peopleResults, docResults, convResults] = await Promise.all([
        searchKnowledge(prompt, department, 4).catch(() => []),
        searchPeople(prompt).catch(() => null),
        searchRecords({ query: prompt, department, limit: 6, loose: true }).catch(() => []),
        searchConversations(prompt, 3, { loose: true }).catch(() => []),
      ]);

      let systemMsg = PROFILE_CONTEXT;
      if (kbResults.length > 0) {
        systemMsg += `\n\nRELEVANT CONTEXT FROM DEPARTMENTS:\n${formatResults(kbResults)}`;
      }
      if (peopleResults) {
        systemMsg += `\n\nRELEVANT PEOPLE:\n${peopleResults}`;
      }
      if (docResults.length > 0) {
        systemMsg += `\n\nRELEVANT DOCUMENTS FROM THE ARCHIVE:\n${formatDocsBrief(docResults)}` +
          `\n\n(To open any of these, use get_record for full detail or get_file_url for the file.)`;
      }
      if (convResults.length > 0) {
        systemMsg += `\n\nRELEVANT PAST CONVERSATIONS:\n${formatConvsBrief(convResults)}`;
      }
      systemMsg += `\n\nGround your answer in the material above. If it does not contain ` +
        `what is needed, say so rather than guessing — more can be retrieved with ` +
        `search_records or search_conversations.`;

      const userMsg = context
        ? `Context:\n${context}\n\nTask:\n${prompt}`
        : prompt;

      const res = await createMessage({
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
        `You are Claude Code, expert software engineer for ${PROFILE.company || "the company"} (IT Department). ` +
        "Write clean, production-ready, well-documented code. No placeholder TODOs. " +
        (language ? `Use ${language}. ` : "") +
        (file_name ? `File: ${file_name}. ` : "") +
        "Structure: 1) brief explanation, 2) complete code in a fenced block, 3) how to run it.";

      const userMsg = existing_code
        ? `Existing code:\n\`\`\`\n${existing_code}\n\`\`\`\n\nTask: ${task}`
        : task;

      const res = await createMessage({
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

      const res = await createMessage({
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
    "Search Spartan's business knowledge base (Supabase) by keyword. Full-text search with keyword matching. " +
    "Optionally filter by department: finance, electronics, chemicals, health, accounts, rd, it, analytics. " +
    "Analytics Department has access to all files across all departments.",
    {
      query:       z.string().describe("Search keywords (e.g. 'GPU servers', 'catalyst EFKO', 'wheat Suslov')"),
      department:  z.string().optional().describe("Filter by department: finance, electronics, chemicals, health, accounts, rd, it, analytics"),
      max_results: z.number().optional().describe("Max results (default 5)"),
    },
    async ({ query, department, max_results }) => {
      // Analytics sees everything
      const dept = department?.toLowerCase() === "analytics" ? null : department;
      const results = await searchKnowledge(query, dept, max_results || 5);
      const peopleResults = await searchPeople(query);

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
    "Get Spartan's full business profile, group structure, org chart with all 8 departments, and a directory of people. Data served live from Supabase.",
    {},
    async () => {
      // Refresh data from DB
      const [depts, people, topics] = await Promise.all([
        sbQuery("departments", "order=id"),
        sbQuery("people", "order=name"),
        sbQuery("topics", "select=id,department_id&order=id"),
      ]);

      const topicsByDept = {};
      for (const t of topics) {
        if (!topicsByDept[t.department_id]) topicsByDept[t.department_id] = [];
        topicsByDept[t.department_id].push(t.id);
      }

      const deptList = depts.map(d => {
        const deptTopics = topicsByDept[d.id] || [];
        return `### ${d.name}\nScope: ${d.scope}\nTopics (${deptTopics.length}): ${deptTopics.join(", ")}`;
      }).join("\n\n");

      const peopleList = people.map(p =>
        `- **${p.name}**${p.aliases?.length ? ` (${p.aliases.join(", ")})` : ""}: ${p.role || p.relation || ""} → ${DEPT_NAMES[p.department_id] || "unassigned"}`
      ).join("\n");

      const text = `# SPARTAN — ${PROFILE.name || "Boss"}
${PROFILE.role || ""}, ${PROFILE.company || ""}
Website: ${PROFILE.website || ""} | Email: ${PROFILE.email || ""}
HQ: ${PROFILE.hq || ""}

## Hierarchy
- **Spartan** (CEO): ${HIERARCHY.spartan?.description || "Overall boss"}
- **The Secretary**: ${HIERARCHY.secretary?.description || "Manages all departments"}

## Group Structure
${Object.values(GROUP_STRUCTURE).map(v => `- ${v}`).join("\n")}

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
      const topics = await sbQuery("topics", "select=id,department_id&order=id");
      const topicsByDept = {};
      for (const t of topics) {
        if (!topicsByDept[t.department_id]) topicsByDept[t.department_id] = [];
        topicsByDept[t.department_id].push(t.id);
      }

      const lines = DEPT_IDS.map(id => {
        const d = DEPARTMENTS[id];
        const deptTopics = topicsByDept[id] || [];
        return `**${d.name}** (${id})\nScope: ${d.scope}\nTopics: ${deptTopics.join(", ")} (${deptTopics.length})`;
      });

      const text = `# Organization — ${DEPT_IDS.length} Departments\n\nSpartan (CEO) → The Secretary → All Departments\n\n${lines.join("\n\n")}`;
      return { content: [{ type: "text", text }] };
    }
  );

  // ─── Tool 7: add_knowledge ────────────────────────────────────
  server.tool(
    "add_knowledge",
    "Add a new knowledge topic to the database. Use when Spartan shares new business info " +
    "that should be persisted across sessions.",
    {
      id:          z.string().describe("Unique topic ID (e.g. 'new_deal_acme')"),
      department:  z.string().describe("Department: finance, electronics, chemicals, health, accounts, rd, it, analytics"),
      keywords:    z.array(z.string()).describe("Search keywords for this topic"),
      summary:     z.string().describe("Summary of the knowledge"),
      details:     z.string().optional().describe("Optional detailed information"),
    },
    async ({ id, department, keywords, summary, details }) => {
      try {
        const body = {
          id,
          department_id: department,
          keywords,
          summary,
          details: details || null,
        };
        await fetch(`${SB_REST}/topics`, {
          method: "POST",
          headers: SB_HEADERS,
          body: JSON.stringify(body),
        });
        return { content: [{ type: "text", text: `Knowledge topic "${id}" added to ${DEPT_NAMES[department] || department}.` }] };
      } catch (err) {
        return { content: [{ type: "text", text: `Failed to add topic: ${err.message}` }] };
      }
    }
  );

  // ─── Tool 8: update_context ───────────────────────────────────
  server.tool(
    "update_context",
    "Store or update a context key-value pair in the database. " +
    "Use for conversation memory, task state, or any persistent data.",
    {
      key:        z.string().describe("Context key (e.g. 'last_task', 'meeting_notes_2026_11')"),
      value:      z.any().describe("Value to store (JSON object)"),
      department: z.string().optional().describe("Optional department scope"),
    },
    async ({ key, value, department }) => {
      try {
        // Upsert: insert or update on conflict
        const body = {
          key,
          value: typeof value === "string" ? { text: value } : value,
          department_id: department || null,
        };
        await fetch(`${SB_REST}/context`, {
          method: "POST",
          headers: { ...SB_HEADERS, Prefer: "resolution=merge-duplicates" },
          body: JSON.stringify(body),
        });
        return { content: [{ type: "text", text: `Context "${key}" saved.` }] };
      } catch (err) {
        return { content: [{ type: "text", text: `Failed to save context: ${err.message}` }] };
      }
    }
  );

  // ─── Tool 9: search_records ───────────────────────────────────
  server.tool(
    "search_records",
    "Search the Shared Records document archive — 3,577 indexed records across all departments, " +
    "covering invoices, certificates of analysis, contracts, bank documents, KYC, purchase orders, " +
    "shipping papers and correspondence. Full-text ranked search over titles, counterparties, " +
    "document numbers, summaries, key fields and OCR'd document text. Works in English and Russian. " +
    "1,048 of these records have the actual file stored and downloadable via get_file_url. " +
    "Use this for ANY question about a specific document, invoice, supplier, certificate or deal paperwork.",
    {
      query:      z.string().describe("What to search for — supplier name, product, invoice number, document type, or free text. Russian terms work."),
      department: z.string().optional().describe("Filter by department: finance, accounts, chemicals, electronics, health, it, rd, secretary"),
      area:       z.string().optional().describe("Filter by area: 'company' or 'personal'"),
      doc_type:   z.string().optional().describe("Filter by document type, partial match (e.g. 'invoice', 'bank statement', 'certificate')"),
      files_only: z.boolean().optional().describe("If true, only return records whose file is actually stored and downloadable"),
      limit:      z.number().optional().describe("Max results, 1-50 (default 10)"),
    },
    async ({ query, department, area, doc_type, files_only, limit }) => {
      try {
        const rows = await searchRecords({
          query, department, area, docType: doc_type, filesOnly: files_only, limit,
        });
        const header = `Found ${rows.length} record(s) for "${query}".\n\n`;
        return { content: [{ type: "text", text: header + formatRecords(rows) }] };
      } catch (err) {
        return { content: [{ type: "text", text: `Record search failed: ${err.message}` }] };
      }
    }
  );

  // ─── Tool 10: get_record ──────────────────────────────────────
  server.tool(
    "get_record",
    "Fetch one document record in full by its Record ID (e.g. REC-0455), including the complete " +
    "OCR'd text excerpt, all metadata, and whether the file is downloadable.",
    {
      record_id: z.string().describe("The Record ID, e.g. REC-0455"),
    },
    async ({ record_id }) => {
      try {
        const rows = await sbQuery(
          "records",
          `record_id=eq.${encodeURIComponent(record_id.trim().toUpperCase())}&limit=1`
        );
        if (!rows.length) {
          return { content: [{ type: "text", text: `No record found with ID ${record_id}.` }] };
        }
        const r = rows[0];
        const lines = [
          `# ${r.title || r.file_name || r.record_id}`,
          ``,
          `- **Record ID:** ${r.record_id}`,
          `- **Area / Department:** ${r.area} / ${DEPT_NAMES[r.department_id] || r.department_id || "—"}${r.subfolder ? ` / ${r.subfolder}` : ""}`,
          `- **Document type:** ${r.doc_type || "—"}`,
          `- **Document date:** ${r.doc_date || "—"}`,
          `- **Counterparty:** ${r.counterparty || "—"}`,
          `- **Document no:** ${r.doc_no || "—"}`,
          `- **Amount:** ${r.amount ? `${r.amount} ${r.currency || ""}`.trim() : "—"}`,
          `- **Status / Version:** ${r.status || "—"}${r.version ? ` / ${r.version}` : ""}`,
          `- **File name:** ${r.file_name || "—"}${r.file_size ? ` (${r.file_size} bytes)` : ""}`,
          `- **File stored:** ${r.storage_path ? "yes — use get_file_url" : "no (metadata only)"}`,
          `- **Original owner:** ${r.owner_email || "—"}`,
          `- **See also:** ${r.see_also || "—"}`,
        ];
        if (r.summary)      lines.push(``, `## Summary`, r.summary);
        if (r.key_fields)   lines.push(``, `## Key fields`, r.key_fields);
        if (r.text_excerpt) lines.push(``, `## Document text`, r.text_excerpt.slice(0, 6000));
        return { content: [{ type: "text", text: lines.join("\n") }] };
      } catch (err) {
        return { content: [{ type: "text", text: `Failed to fetch record: ${err.message}` }] };
      }
    }
  );

  // ─── Tool 11: get_file_url ────────────────────────────────────
  server.tool(
    "get_file_url",
    "Get a time-limited download link for a stored document, by Record ID. " +
    "Only works for records whose file is in the archive (search_records shows which). " +
    "The link expires, so fetch it when you actually need the file.",
    {
      record_id:  z.string().describe("The Record ID, e.g. REC-0455"),
      expires_in: z.number().optional().describe("Link lifetime in seconds, 60-86400 (default 3600)"),
    },
    async ({ record_id, expires_in }) => {
      try {
        const rows = await sbQuery(
          "records",
          `record_id=eq.${encodeURIComponent(record_id.trim().toUpperCase())}&select=record_id,file_name,storage_path,file_size&limit=1`
        );
        if (!rows.length) {
          return { content: [{ type: "text", text: `No record found with ID ${record_id}.` }] };
        }
        const r = rows[0];
        if (!r.storage_path) {
          return { content: [{ type: "text", text:
            `${r.record_id} has no stored file — it is one of the metadata-only records. ` +
            `Its summary and text are still searchable via get_record.` }] };
        }
        const ttl = Math.min(Math.max(parseInt(expires_in || 3600, 10), 60), 86400);
        const url = await signedUrl(r.storage_path, ttl);
        return { content: [{ type: "text", text:
          `${r.record_id} — ${r.file_name}\nExpires in ${ttl}s\n\n${url}` }] };
      } catch (err) {
        return { content: [{ type: "text", text: `Failed to create download link: ${err.message}` }] };
      }
    }
  );

  // ─── Tool 12: archive_stats ───────────────────────────────────
  server.tool(
    "archive_stats",
    "Overview of the document archive: how many records exist per department, how many have " +
    "downloadable files, and the most common document types. Use this to orient before searching.",
    {
      department: z.string().optional().describe("Optional: restrict the breakdown to one department"),
    },
    async ({ department }) => {
      try {
        // Aggregate in Postgres. Counting client-side is wrong here: PostgREST
        // caps result rows (default 1000), so a row-by-row tally silently
        // under-reports a 3,577-row table.
        const res = await fetch(`${SB_REST}/rpc/archive_stats`, {
          method: "POST",
          headers: SB_HEADERS,
          body: JSON.stringify({ dept: department || null }),
        });
        if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
        const s = await res.json();
        const deptLines = (s.by_department || []).map(d =>
          `  ${(DEPT_NAMES[d.department_id] || d.department_id || "—").padEnd(24)} ${String(d.total).padStart(5)} records, ${d.files} with files`
        );
        const typeLines = (s.by_type || []).map(t =>
          `  ${String(t.doc_type).slice(0, 40).padEnd(42)} ${t.n}`
        );
        return { content: [{ type: "text", text:
          `ARCHIVE: ${s.total} records, ${s.with_files} with downloadable files ` +
          `(${s.storage_mb} MB in storage), ${s.personal} personal.\n\n` +
          `BY DEPARTMENT\n${deptLines.join("\n")}\n\nTOP DOCUMENT TYPES\n${typeLines.join("\n")}` }] };
      } catch (err) {
        return { content: [{ type: "text", text: `Failed to read archive stats: ${err.message}` }] };
      }
    }
  );

  // ─── Tool 13: search_conversations ────────────────────────────
  server.tool(
    "search_conversations",
    "Search past Claude conversations — what was discussed, decided and produced. " +
    "Use this for questions about reasoning and history rather than paperwork: " +
    "'what did we decide about X', 'what was agreed with Y', 'what did we conclude'. " +
    "Documents live in search_records; this is the thinking behind them.",
    {
      query:    z.string().optional().describe("What to look for. Leave empty for the most recent conversations."),
      category: z.string().optional().describe("Filter: 'business' or 'personal'"),
      limit:    z.number().optional().describe("Max results, 1-25 (default 5)"),
    },
    async ({ query, category, limit }) => {
      try {
        const rows = await searchConversations(query, limit || 5, { category });
        if (!rows.length) {
          return { content: [{ type: "text", text: `No conversations matched "${query || "(recent)"}".` }] };
        }
        const body = rows.map(c => {
          const lines = [
            `## ${c.title || c.chat_id}`,
            `Category: ${c.category || "—"}${c.department_id ? ` | Department: ${DEPT_NAMES[c.department_id] || c.department_id}` : ""}` +
              `${c.src_updated_at ? ` | Last active: ${String(c.src_updated_at).slice(0, 10)}` : ""}`,
          ];
          if (c.summary)       lines.push("", c.summary);
          if (c.key_points)    lines.push("", `**Key points:** ${c.key_points}`);
          if (c.entities)      lines.push(`**Mentioned:** ${c.entities}`);
          if (c.artifacts)     lines.push(`**Files discussed:** ${c.artifacts}`);
          if (c.url)           lines.push(`${c.url}`);
          return lines.join("\n");
        }).join("\n\n---\n\n");
        return { content: [{ type: "text", text: `Found ${rows.length} conversation(s).\n\n${body}` }] };
      } catch (err) {
        return { content: [{ type: "text", text: `Conversation search failed: ${err.message}` }] };
      }
    }
  );

  // ─── Tool 14: enrich_pending ──────────────────────────────────
  server.tool(
    "enrich_pending",
    "Classify newly ingested documents. Finds records flagged needs_review, reads each file " +
    "out of Storage (PDFs and scans included), and fills in department, subfolder, document type, " +
    "counterparty, date, amount, summary and key fields using the archive's existing taxonomy. " +
    "Called automatically by the inbox ingest job; can also be run manually.",
    {
      limit: z.number().optional().describe("Max records to classify in one pass, 1-25 (default 10)"),
    },
    async ({ limit }) => {
      const max = Math.min(Math.max(parseInt(limit || 10, 10), 1), 25);
      try {
        const pending = await sbQuery(
          "records",
          `needs_review=is.true&storage_path=not.is.null&select=record_id,file_name,original_filename,storage_path,file_size,source_note&order=ingested_at.asc&limit=${max}`
        );
        if (!pending.length) {
          return { content: [{ type: "text", text: "Nothing pending — all ingested records are classified." }] };
        }
        const tax = await (await fetch(`${SB_REST}/rpc/taxonomy`, {
          method: "POST", headers: SB_HEADERS, body: "{}",
        })).json();

        const done = [], failed = [];
        for (const r of pending) {
          try {
            const result = await classifyOne(r, tax);
            done.push(`${r.record_id} → ${result.department_id}/${result.subfolder} (${result.doc_type}, ${result.confidence})`);
          } catch (e) {
            failed.push(`${r.record_id}: ${e.message}`);
            await sbPatch(r.record_id, { classify_error: String(e.message).slice(0, 400) });
          }
        }
        const lines = [`Classified ${done.length} of ${pending.length} pending record(s).`];
        if (done.length)   lines.push("", ...done);
        if (failed.length) lines.push("", "FAILED:", ...failed);
        return { content: [{ type: "text", text: lines.join("\n") }] };
      } catch (err) {
        return { content: [{ type: "text", text: `enrich_pending failed: ${err.message}` }] };
      }
    }
  );

  return server;
}

// ── Classification helpers ──────────────────────────────────────
async function sbPatch(recordId, fields) {
  const res = await fetch(
    `${SB_REST}/records?record_id=eq.${encodeURIComponent(recordId)}`,
    { method: "PATCH", headers: { ...SB_HEADERS, Prefer: "return=minimal" }, body: JSON.stringify(fields) }
  );
  if (!res.ok) throw new Error(`patch ${recordId} failed (${res.status}): ${await res.text()}`);
}

const DOC_MIME = {
  pdf: "application/pdf",
};
const IMG_MIME = {
  png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg",
  gif: "image/gif", webp: "image/webp",
};
const TEXT_EXT = new Set(["txt", "md", "csv", "json", "xml", "html", "htm", "log", "eml"]);
const MAX_BYTES = 20 * 1024 * 1024; // keep well inside the API's document limit

async function classifyOne(rec, tax) {
  // Pull the actual bytes back out of Storage.
  const dl = await fetch(
    `${SUPABASE_URL}/storage/v1/object/records/${rec.storage_path.split("/").map(encodeURIComponent).join("/")}`,
    { headers: { apikey: SUPABASE_KEY, Authorization: `Bearer ${SUPABASE_KEY}` } }
  );
  if (!dl.ok) throw new Error(`download failed (${dl.status})`);
  const buf = Buffer.from(await dl.arrayBuffer());

  const name = rec.original_filename || rec.file_name || "";
  const ext = (name.split(".").pop() || "").toLowerCase();

  const blocks = [];
  let readable = true;
  if (buf.length > MAX_BYTES) {
    readable = false;
  } else if (DOC_MIME[ext]) {
    blocks.push({ type: "document", source: { type: "base64", media_type: DOC_MIME[ext], data: buf.toString("base64") } });
  } else if (IMG_MIME[ext]) {
    blocks.push({ type: "image", source: { type: "base64", media_type: IMG_MIME[ext], data: buf.toString("base64") } });
  } else if (TEXT_EXT.has(ext)) {
    blocks.push({ type: "text", text: "FILE CONTENTS:\n" + buf.toString("utf-8").slice(0, 40000) });
  } else {
    readable = false;
  }

  const subfolderList = (tax.subfolders || [])
    .map(s => `${s.dept}/${s.subfolder}`).join(", ");
  const deptList = (tax.departments || [])
    .map(d => `${d.id} — ${d.name}: ${d.scope}`).join("\n");

  const instruction =
`Classify this business document for the MetalMarket archive.

Prefer the EXISTING taxonomy. Invent a new subfolder only when nothing fits; use lowercase-hyphenated slugs.

DEPARTMENTS:
${deptList}

EXISTING SUBFOLDERS (department/subfolder):
${subfolderList}

EXISTING DOCUMENT TYPES: ${(tax.doc_types || []).join(", ")}

KNOWN COUNTERPARTIES: ${(tax.counterparties || []).join(", ")}

FILE NAME: ${name}
${readable ? "" : "NOTE: the file could not be read (unsupported type or too large). Classify from the filename alone and set confidence to low."}

Reply with ONLY a JSON object, no prose, no code fences:
{"department_id":"...","subfolder":"...","title":"...","doc_type":"...","counterparty":null,"doc_no":null,"doc_date":null,"amount":null,"currency":null,"summary":"...","key_fields":"...","text_excerpt":"...","confidence":"high|medium|low"}

Rules:
- department_id MUST be one of the department ids listed above.
- doc_date must be YYYY-MM-DD or null. amount must be a number or null.
- summary: 2-3 factual sentences. Never invent facts not in the document.
- key_fields: the important verbatim values — batch numbers, references, quantities, dates.
- text_excerpt: up to 4000 characters of the document's actual text as you read it.
- confidence: low if you are guessing from the filename.`;

  blocks.push({ type: "text", text: instruction });

  const res = await createMessage({
    max_tokens: 4096,
    messages: [{ role: "user", content: blocks }],
  });
  let raw = res.content.filter(b => b.type === "text").map(b => b.text).join("").trim();
  raw = raw.replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/, "").trim();
  const m = raw.match(/\{[\s\S]*\}/);
  if (!m) throw new Error("classifier returned no JSON");
  const c = JSON.parse(m[0]);

  const validDepts = new Set((tax.departments || []).map(d => d.id));
  if (!validDepts.has(c.department_id)) {
    throw new Error(`classifier chose unknown department "${c.department_id}"`);
  }

  const lowConf = String(c.confidence || "").toLowerCase() === "low";
  await sbPatch(rec.record_id, {
    department_id: c.department_id,
    subfolder:     c.subfolder || "_general",
    title:         c.title || name,
    doc_type:      c.doc_type || null,
    counterparty:  c.counterparty || null,
    doc_no:        c.doc_no || null,
    doc_date:      c.doc_date || null,
    amount:        (typeof c.amount === "number" ? c.amount : null),
    currency:      c.currency || null,
    summary:       c.summary || null,
    key_fields:    c.key_fields || null,
    text_excerpt:  c.text_excerpt || null,
    needs_review:  lowConf,            // low confidence stays flagged for a human
    classify_error: null,
    source_note:   `auto-classified (${c.confidence || "unknown"} confidence)`,
  });
  return c;
}


// ── Lazy, cached bootstrap ──────────────────────────────────────
// On a long-lived server this ran once at startup. On serverless each cold
// start gets a fresh process, so load on first use and reuse while the
// instance stays warm. Concurrent callers share one in-flight promise.
let _bootstrap = null;
export function ready() {
  if (!_bootstrap) {
    _bootstrap = loadStaticData().catch(err => {
      _bootstrap = null;            // let the next request retry
      throw err;
    });
  }
  return _bootstrap;
}

export function statusPayload() {
  return {
    status: "ok",
    server: "claude-assistant",
    version: VERSION,
    database: "supabase",
    platform: "vercel",
    model: RESOLVED_MODEL || "(not yet resolved)",
    models_available: AVAILABLE_MODELS.length || null,
    hierarchy: { ceo: "Spartan", assistant: "The Secretary" },
    departments: DEPT_IDS.map(id => ({ id, name: DEPARTMENTS[id]?.name })),
    tools: TOOL_NAMES,
    transport: "Streamable HTTP (stateless)",
  };
}

export { createServer, AUTH_TOKEN };
