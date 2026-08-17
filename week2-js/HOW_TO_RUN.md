# How to run everything (week2-js)

All commands run from the `week2-js/` folder. Run `npm install` once first if
you haven't already, and make sure `OPENROUTER_API_KEY` is set in `.env`.

## Exercise 1 - Prompt Templates

```
npm run prompt-templates-demo
```

## Exercise 2 - Model Invocation Methods

```
npm run invocation-demo
```

## Exercise 3 - Review Analysis Pipeline

```
npm run review-pipeline
```

Results are saved to `output/review_analysis_with_structured_output.json` and
`output/review_analysis_with_output_parser.json`.

## Bonus - tool/function calling demo (multi-tool)

Runs 3 scenarios showing the model choosing 1 tool, then 2 tools (across 2
rounds), then 1 tool again - not always the same tool set:
```
npm run tool-calling-demo
```

## All commands at a glance

| What | Command |
|---|---|
| Install deps | `npm install` |
| Ex. 1 prompt templates demo | `npm run prompt-templates-demo` |
| Ex. 2 invocation methods demo | `npm run invocation-demo` |
| Ex. 3 review analysis pipeline | `npm run review-pipeline` |
| Bonus tool-calling demo | `npm run tool-calling-demo` |
| Type-check everything | `npm run typecheck` |
