# Example Library Guide

## Overview

The Example Library stores verified correct extraction results with their screenshots. Claude can reference these examples to improve accuracy on new screenshots (few-shot learning).

## Directory Structure

```
/home/admin/ocr-testing/ocr-new/examples/
├── 20241210_120530/
│   ├── screenshot.jpg          # Original screenshot
│   ├── result.json             # Verified correct extraction
│   └── metadata.json           # Tags, notes, stats
├── 20241210_120545/
│   ├── screenshot.jpg
│   ├── result.json
│   └── metadata.json
└── ...
```

## API Endpoints

### 1. Save an Example
```
POST /ocr/examples/save

Request:
{
  "screenshot_base64": "...",
  "extraction_result": {
    "teams": [...]
  },
  "tags": ["glare", "partial_card", "high_kills"],
  "notes": "Example with glare effect on top 3 cards"
}

Response:
{
  "success": true,
  "example_id": "20241210_120530",
  "tags": ["glare", "partial_card", "high_kills"],
  "notes": "..."
}
```

### 2. List All Examples
```
GET /ocr/examples/list

Response:
{
  "success": true,
  "count": 5,
  "examples": [
    {
      "example_id": "20241210_120530",
      "timestamp": "2024-12-10T12:05:30",
      "tags": ["glare", "partial_card"],
      "notes": "...",
      "num_teams": 12,
      "total_kills": 87
    },
    ...
  ]
}
```

### 3. Get Specific Example
```
GET /ocr/examples/{example_id}

Response:
{
  "success": true,
  "example": {
    "example_id": "20241210_120530",
    "screenshot_base64": "...",
    "result": {...},
    "metadata": {...}
  }
}
```

## Recommended Tags

Use these tags to categorize examples:

- **`glare`** — Top 3 cards have shine/glare effect
- **`partial_card`** — Card is cut off (only 2 players visible)
- **`high_kills`** — Team total > 30 kills
- **`low_kills`** — Team total < 5 kills
- **`ambiguous_digits`** — Contains confusing digits (0/8/9/6)
- **`white_placement`** — Placement 2-12 (white colored)
- **`gold_placement`** — Placement 1 (yellow/gold colored)
- **`scrollable_view`** — Partial screenshot (5-7 cards visible)
- **`full_view`** — All 12 cards visible
- **`edge_case`** — Unusual or tricky scenario

## Workflow

### Step 1: Extract with Claude
Use the Claude Extractor page to extract a screenshot.

### Step 2: Verify Results
Manually check the extraction is 100% correct.

### Step 3: Save as Example
```javascript
// From claude_extractor.html, after successful extraction:
const response = await fetch('http://3.149.239.69:8000/ocr/examples/save', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    screenshot_base64: screenshotBase64,
    extraction_result: extractionResult.data,
    tags: ['glare', 'partial_card'],
    notes: 'Example with glare on top 3 cards and partial card at bottom'
  })
});
```

### Step 4: Build Few-Shot Prompt (Future)
When extracting new screenshots, include 1-2 relevant examples in the prompt:

```python
# In full_result_extractor.py (future enhancement)
relevant_examples = example_library.find_examples_by_tag('glare', limit=1)
few_shot_section = example_library.build_few_shot_prompt(tags=['glare'], limit=1)
user_prompt = few_shot_section + original_user_prompt
```

## Example Metadata

Each example stores:
- `example_id` — Timestamp-based ID (YYYYMMDD_HHMMSS)
- `timestamp` — ISO format creation time
- `tags` — List of category tags
- `notes` — Human notes about the example
- `screenshot_size_bytes` — Image size
- `num_teams` — Number of teams extracted
- `total_kills` — Sum of all team kills

## Best Practices

1. **Verify before saving** — Only save 100% correct extractions
2. **Use descriptive tags** — Help identify relevant examples later
3. **Add notes** — Explain what makes this example special
4. **Build diverse library** — Include edge cases (glare, partial cards, etc.)
5. **Regular cleanup** — Delete incorrect or outdated examples

## Cost Impact

- **Storage:** ~500KB per example (screenshot + JSON)
- **Retrieval:** No extra cost (local file system)
- **Few-shot tokens:** ~1500-2000 tokens per example included in prompt (~$0.005-0.01)

## Future Enhancements

1. **Auto-include examples** — Detect edge cases and auto-inject relevant examples
2. **Similarity matching** — Find visually similar examples
3. **Confidence scoring** — Track which examples improve accuracy most
4. **Example versioning** — Track extraction quality over time
