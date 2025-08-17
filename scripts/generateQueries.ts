import fs from "fs";
import path from "path";
import OpenAI from "openai";
import readline from "readline";

interface Hit {
  id: string;
  name: string;
  markdownContent?: string;
  summary?: string;
  [k: string]: any;
}

interface Dataset {
  hits: Hit[];
  [k: string]: any;
}

interface GeneratedQuerySet {
  keywords: string[]; // short keyword/grouped terms
  entityPhrases: string[]; // proper nouns / named concepts
  conceptualPhrases: string[]; // higher-level intent phrases
  naturalQuestions: string[]; // natural language user questions
  problemStatements: string[]; // "How do I ..." / "What is ..." style
}

interface EnrichedHit {
  id: string; // original hit id
  rule: string; // name / number
  content: string; // markdown
  searchQueries?: GeneratedQuerySet;
}

const openaiApiKey = process.env.OPENAI_API_KEY;

// Heuristic removal: we now require OpenAI API key; no fallback provided.

async function generateQueriesForHit(
  client: OpenAI,
  hit: Hit
): Promise<GeneratedQuerySet> {
  const content = hit.markdownContent || hit.summary || "";
  const truncated = content.length > 8000 ? content.slice(0, 8000) : content;
  const system = `You generate diverse search queries for a search index. Return concise JSON with keys: keywords, entityPhrases, conceptualPhrases, naturalQuestions, problemStatements. Avoid duplicates. Tailor to the provided markdown excerpt. Keywords should be 1-3 word phrases. Provide 8-15 naturalQuestions.`;
  const user = `Section ID: ${hit.id}\nName: ${hit.name}\nMarkdown:\n${truncated}`;
  const response = await client.chat.completions.create({
    model: "gpt-4o-mini",
    temperature: 0.4,
    messages: [
      { role: "system", content: system },
      { role: "user", content: user },
    ],
    response_format: { type: "json_object" },
  });
  const raw = response.choices[0].message.content || "{}";
  const parsed = JSON.parse(raw);
  return {
    keywords: parsed.keywords || [],
    entityPhrases: parsed.entityPhrases || [],
    conceptualPhrases: parsed.conceptualPhrases || [],
    naturalQuestions: parsed.naturalQuestions || [],
    problemStatements: parsed.problemStatements || [],
  };
}

function formatDuration(ms: number) {
  const s = ms / 1000;
  if (s < 60) return s.toFixed(1) + "s";
  const m = Math.floor(s / 60);
  const r = Math.round(s % 60);
  return `${m}m ${r}s`;
}

function updateStatus(line: string) {
  readline.clearLine(process.stdout, 0);
  readline.cursorTo(process.stdout, 0);
  process.stdout.write(line);
}

async function main() {
  const [, , inputFile, outputFile = "queries-output.json", ...rest] =
    process.argv;
  if (!inputFile) {
    console.error(
      "Usage: tsx scripts/generateQueries.ts <input.json> [output.json] [--limit=N] [--batchSize=K] [--concurrency=C] [--saveEvery=B] [--resume]"
    );
    process.exit(1);
  }

  // Parse flags
  const flags: Record<string, string | boolean> = {};
  rest.forEach((arg) => {
    if (arg.startsWith("--")) {
      const [k, v] = arg.replace(/^--/, "").split("=");
      flags[k] = v === undefined ? true : v;
    }
  });
  const limit = flags.limit ? parseInt(flags.limit as string, 10) : undefined;
  const batchSize = flags.batchSize
    ? parseInt(flags.batchSize as string, 10)
    : 5;
  const concurrency = flags.concurrency
    ? parseInt(flags.concurrency as string, 10)
    : batchSize; // default same as batch
  const saveEvery = flags.saveEvery
    ? parseInt(flags.saveEvery as string, 10)
    : 1; // save after each batch by default
  const resume = !!flags.resume;

  const absInput = path.resolve(process.cwd(), inputFile);
  if (!fs.existsSync(absInput)) {
    console.error("Input file not found:", absInput);
    process.exit(1);
  }
  const dataset: Dataset = JSON.parse(fs.readFileSync(absInput, "utf-8"));
  const hits = dataset.hits || [];
  const total = limit ? Math.min(limit, hits.length) : hits.length;

  if (!openaiApiKey) {
    console.error(
      "OPENAI_API_KEY is required (heuristic fallback removed). Aborting."
    );
    process.exit(1);
  }
  const client = new OpenAI({ apiKey: openaiApiKey });

  const absOutput = path.resolve(process.cwd(), outputFile);
  let enriched: EnrichedHit[] = [];
  let processedIds = new Set<string>();
  if (resume && fs.existsSync(absOutput)) {
    try {
      const existing = JSON.parse(fs.readFileSync(absOutput, "utf-8"));
      if (Array.isArray(existing.hits)) {
        enriched = existing.hits;
        enriched.forEach((h: EnrichedHit) => processedIds.add(h.id));
        console.log(`Resumed: preloaded ${enriched.length} hits.`);
      }
    } catch (e) {
      console.warn(
        "Resume requested but failed to parse existing output. Starting fresh."
      );
    }
  }

  const startIndex = enriched.length
    ? hits.findIndex((h) => !processedIds.has(h.id))
    : 0;
  const startTime = Date.now();
  let errors = 0;
  console.log(
    `Starting generation: total=${total} model=gpt-4o-mini batchSize=${batchSize} concurrency=${concurrency} resume=${resume}`
  );

  async function processBatch(batchHits: Hit[], globalOffset: number) {
    // Concurrency control: chunk into groups of size 'concurrency'
    const results: EnrichedHit[] = new Array(batchHits.length);
    let idx = 0;
    async function worker(i: number, hit: Hit) {
      const t0 = Date.now();
      try {
        const queries = await generateQueriesForHit(client, hit);
        results[i] = {
          id: hit.id,
          rule: hit.name,
          content: hit.markdownContent || hit.summary || "",
          searchQueries: queries,
        };
        return Date.now() - t0;
      } catch (e) {
        errors++;
        throw e; // fail fast
      }
    }
    while (idx < batchHits.length) {
      const slice = batchHits.slice(idx, idx + concurrency);
      const durations = await Promise.all(
        slice.map((h, si) => worker(idx + si, h))
      );
      idx += slice.length;
      const completed = globalOffset + idx;
      const elapsed = Date.now() - startTime;
      const avg = elapsed / (enriched.length + idx);
      const remaining = total - (enriched.length + idx);
      const eta = remaining * avg;
      const pct = (((enriched.length + idx) / total) * 100).toFixed(1);
      updateStatus(
        `Batch progress ${completed}/${total} (${pct}%) | lastChunkAvg ${Math.round(
          durations.reduce((a, b) => a + b, 0) / durations.length
        )}ms | elapsed ${formatDuration(elapsed)} | ETA ${formatDuration(
          eta
        )} | errors ${errors}`
      );
    }
    enriched.push(...results);
  }

  for (let offset = startIndex; offset < total; offset += batchSize) {
    const slice = hits
      .slice(offset, Math.min(offset + batchSize, total))
      .filter((h) => !processedIds.has(h.id));
    if (!slice.length) continue; // all already processed
    await processBatch(slice, offset + slice.length);
    const batchesDone = Math.ceil(
      (enriched.length - processedIds.size) / batchSize
    );
    if (batchesDone % saveEvery === 0) {
      const tempPath = absOutput + ".tmp";
      const meta = {
        generatedAt: new Date().toISOString(),
        count: enriched.length,
        errors,
      };
      fs.writeFileSync(
        tempPath,
        JSON.stringify({ hits: enriched, meta }, null, 2)
      );
      fs.renameSync(tempPath, absOutput); // atomic swap on most systems
    }
  }

  process.stdout.write("\n");
  const meta = {
    generatedAt: new Date().toISOString(),
    count: enriched.length,
    errors,
  };
  fs.writeFileSync(
    absOutput,
    JSON.stringify({ hits: enriched, meta }, null, 2)
  );
  console.log(
    `Done. Wrote enriched dataset with ${enriched.length} hits -> ${absOutput}`
  );
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
