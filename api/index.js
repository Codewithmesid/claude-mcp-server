// Status endpoint. Mirrors what the Render build served at "/".
import { ready, statusPayload } from "../lib/core.js";

export const config = { maxDuration: 60 };

export default async function handler(_req, res) {
  try {
    await ready();
    res.setHeader("Cache-Control", "no-store");
    res.status(200).json(statusPayload());
  } catch (err) {
    res.status(503).json({ status: "error", message: String(err?.message || err) });
  }
}
