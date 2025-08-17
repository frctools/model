# Search Query Generation

Generate enriched search phrases and natural questions for each hit in the dataset JSON files.

## Usage

1. Install dependencies:

```bash
npm install
```

2. Set OpenAI key (required):

```bash
export OPENAI_API_KEY=sk-...
```

3. Run generation:

```bash
npm run generate:2025
```

Outputs `queries-2025.json` with a `searchQueries` object per hit. An OpenAI API key is required; the script will exit if it's missing.

### Advanced Options

You can control batching, concurrency, and incremental saving:

```bash
tsx scripts/generateQueries.ts 2025.json queries-2025.json --limit=100 --batchSize=10 --concurrency=5 --saveEvery=2
```

Flags:

- `--limit=N` Process only first N hits.
- `--batchSize=K` Number of hits grouped per batch (default 5).
- `--concurrency=C` Parallel requests inside a batch (default = batchSize).
- `--saveEvery=B` Persist to disk every B batches (default 1).
- `--resume` Resume from an existing output file (matched by `id`).

Progress is streamed in-place with elapsed time, ETA, and error count. A `.tmp` file is atomically swapped for durability on each save.

### Export Anchors to CSV

After generation you can flatten all generated phrases/questions to a CSV (one row per anchor with associated positive content snippet):

```bash
npm run export:2025
# or custom
tsx scripts/exportAnchors.ts queries-2025.json anchors-2025.csv --maxContentChars=3000
```

CSV Columns:

- hit_id: original section id
- rule: section name/number
- anchor: generated phrase or question
- content: associated markdown snippet (truncated)
