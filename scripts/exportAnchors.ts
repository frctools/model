import fs from "fs";
import path from "path";

interface GeneratedQuerySet {
  keywords: string[];
  entityPhrases: string[];
  conceptualPhrases: string[];
  naturalQuestions: string[];
  problemStatements: string[];
}

interface EnrichedHit {
  id: string;
  rule: string;
  content: string;
  searchQueries?: GeneratedQuerySet;
}

interface EnrichedDataset {
  hits: EnrichedHit[];
  meta?: any;
}

function escapeCsv(value: string): string {
  const needsQuotes = /[",\n]/.test(value);
  const escaped = value.replace(/"/g, '""');
  return needsQuotes ? `"${escaped}"` : escaped;
}

function collectAnchors(hit: EnrichedHit): string[] {
  if (!hit.searchQueries) return [];
  const {
    keywords = [],
    entityPhrases = [],
    conceptualPhrases = [],
    naturalQuestions = [],
    problemStatements = [],
  } = hit.searchQueries;
  return [
    ...keywords,
    ...entityPhrases,
    ...conceptualPhrases,
    ...naturalQuestions,
    ...problemStatements,
  ].filter(Boolean);
}

function main() {
  const [, , inputFile, outputFile = "anchors-output.csv", ...rest] =
    process.argv;
  if (!inputFile) {
    console.error(
      "Usage: tsx scripts/exportAnchors.ts <queries.json> [output.csv] [--maxContentChars=N]"
    );
    process.exit(1);
  }
  const flags: Record<string, string | boolean> = {};
  rest.forEach((arg) => {
    if (arg.startsWith("--")) {
      const [k, v] = arg.replace(/^--/, "").split("=");
      flags[k] = v === undefined ? true : v;
    }
  });
  const maxContentChars = flags.maxContentChars
    ? parseInt(flags.maxContentChars as string, 10)
    : 4000;

  const absInput = path.resolve(process.cwd(), inputFile);
  if (!fs.existsSync(absInput)) {
    console.error("Input file not found:", absInput);
    process.exit(1);
  }
  const dataset: EnrichedDataset = JSON.parse(
    fs.readFileSync(absInput, "utf-8")
  );
  const rows: string[] = [];
  rows.push(["anchor", "content"].map(escapeCsv).join(","));
  let count = 0;
  for (const hit of dataset.hits) {
    const anchors = collectAnchors(hit);
    const content = (hit.content || "").slice(0, maxContentChars);
    for (const anchor of anchors) {
      rows.push(
        [
          escapeCsv(anchor),
          escapeCsv(content),
        ].join(",")
      );
      count++;
    }
  }
  const absOutput = path.resolve(process.cwd(), outputFile);
  fs.writeFileSync(absOutput, rows.join("\n"));
  console.log(`Wrote ${count} anchor rows -> ${absOutput}`);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
