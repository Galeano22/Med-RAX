#!/usr/bin/env node
/*
 * Minimal MedRAX API client.
 * Loads an image from disk (CR.png by default), asks you for a text prompt,
 * sends both to the FastAPI backend (api.py), and prints the agent's answer.
 *
 * Usage:
 *   node ask_medrax.js                    # uses ./CR.png, prompts for text
 *   node ask_medrax.js path/to/xray.png   # a different image
 *   MEDRAX_API=http://host:8686 node ask_medrax.js
 */

const fs = require("fs");
const path = require("path");
const readline = require("readline");

const API = (process.env.MEDRAX_API || "http://localhost:8686").replace(/\/$/, "");
const IMAGE = process.argv[2] || path.join(__dirname, "CR.png");

function ask(question) {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  return new Promise((resolve) =>
    rl.question(question, (answer) => {
      rl.close();
      resolve(answer);
    })
  );
}

function mimeFor(file) {
  const ext = path.extname(file).toLowerCase();
  return ext === ".jpg" || ext === ".jpeg" ? "image/jpeg" : "image/png";
}

async function main() {
  if (!fs.existsSync(IMAGE)) {
    console.error(`Image not found: ${IMAGE}`);
    process.exit(1);
  }

  // 1. Load the image and encode it as a data URI.
  const b64 = fs.readFileSync(IMAGE).toString("base64");
  const dataUri = `data:${mimeFor(IMAGE)};base64,${b64}`;
  console.log(`Loaded ${path.basename(IMAGE)} (${(b64.length / 1024).toFixed(0)} KB base64)`);

  // 2. Ask for the text prompt.
  const message =
    (await ask("Prompt for MedRAX: ")).trim() ||
    "Analyze this chest X-ray: classify the findings and summarize.";

  // 3. Send both to the API.
  console.log("\nSending to MedRAX… (this can take a while on first tool use)\n");
  const res = await fetch(`${API}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, image_b64: dataUri }),
  });

  if (!res.ok) {
    console.error(`API error ${res.status}: ${await res.text()}`);
    process.exit(1);
  }

  const data = await res.json();

  // 4. Show the result.
  console.log("──────── MedRAX reply ────────");
  console.log(data.reply);
  if (data.tool_calls?.length) {
    console.log("\nTools used:", data.tool_calls.map((t) => t.name).join(", "));
  }
  if (data.images?.length) {
    console.log("\nImages produced (download from the API):");
    data.images.forEach((u) => console.log(`  ${API}${u}`));
  }
  console.log(`\nthread_id: ${data.thread_id}  (pass it back to continue the conversation)`);
}

main().catch((e) => {
  console.error("Request failed:", e.message);
  console.error(`Is the backend running?  uvicorn api:app --host 0.0.0.0 --port 8686`);
  process.exit(1);
});
