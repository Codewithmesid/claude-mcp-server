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
const SUPABASE_URL  = process.env.SUPABASE_URL;
const SUPABASE_KEY  = process.env.SUPABASE_SERVICE_KEY;

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
async function searchRecords({ query, department, area, docType, filesOnly, limit }) {
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

// ── MCP server ──────────────────────────────────────────────────
function createServer() {
  const server = new McpServer({
    name: "claude-assistant",
    version: "5.0.0",
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
      const kbResults = await searchKnowledge(prompt, department, 4);
      const peopleResults = await searchPeople(prompt);

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
        `You are Claude Code, expert software engineer for ${PROFILE.company || "the company"} (IT Department). ` +
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
        const filter = department ? `&department_id=eq.${encodeURIComponent(department)}` : "";
        const rows = await sbQuery(
          "records",
          `select=department_id,area,doc_type,storage_path&limit=5000${filter}`
        );
        const byDept = {}, byType = {};
        let withFile = 0, personal = 0;
        for (const r of rows) {
          const d = r.department_id || "—";
          byDept[d] = byDept[d] || { total: 0, files: 0 };
          byDept[d].total++;
          if (r.storage_path) { byDept[d].files++; withFile++; }
          if (r.area === "personal") personal++;
          const t = r.doc_type || "(none)";
          byType[t] = (byType[t] || 0) + 1;
        }
        const deptLines = Object.entries(byDept)
          .sort((a, b) => b[1].total - a[1].total)
          .map(([d, v]) => `  ${(DEPT_NAMES[d] || d).padEnd(24)} ${String(v.total).padStart(5)} records, ${v.files} with files`);
        const typeLines = Object.entries(byType)
          .sort((a, b) => b[1] - a[1]).slice(0, 12)
          .map(([t, n]) => `  ${t.slice(0, 40).padEnd(42)} ${n}`);
        return { content: [{ type: "text", text:
          `ARCHIVE: ${rows.length} records, ${withFile} with downloadable files, ${personal} personal.\n\n` +
          `BY DEPARTMENT\n${deptLines.join("\n")}\n\nTOP DOCUMENT TYPES\n${typeLines.join("\n")}` }] };
      } catch (err) {
        return { content: [{ type: "text", text: `Failed to read archive stats: ${err.message}` }] };
      }
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
    version: "5.0.0",
    database: "supabase",
    hierarchy: { ceo: "Spartan", assistant: "The Secretary" },
    departments: DEPT_IDS.map(id => ({ id, name: DEPARTMENTS[id]?.name })),
    tools: ["ask_claude", "code_with_claude", "review_code", "search_knowledge", "get_profile", "list_departments", "add_knowledge", "update_context", "search_records", "get_record", "get_file_url", "archive_stats"],
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

// ── Start ───────────────────────────────────────────────────────
async function start() {
  try {
    console.log("Loading knowledge base from Supabase...");
    await loadStaticData();
    app.listen(PORT, "0.0.0.0", () => {
      console.log(`Claude MCP Server v4.0.0 — Spartan Command (Supabase)`);
      console.log(`Database: ${SUPABASE_URL}`);
      console.log(`Departments: ${DEPT_IDS.length}`);
      console.log(`MCP endpoint: http://localhost:${PORT}/mcp`);
    });
  } catch (err) {
    console.error("Failed to start:", err);
    process.exit(1);
  }
}

start();
