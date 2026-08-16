

async function streamQuestion(question: string, baseUrl = "http://127.0.0.1:8000"): Promise<void> {
  const start = performance.now();
  let event = "message";

  const response = await fetch(`${baseUrl}/ask/stream?${new URLSearchParams({ question })}`);
  if (!response.ok || !response.body) {
    throw new Error(`request failed: ${response.status} ${response.statusText}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? ""; // keep the last (possibly incomplete) line for next chunk

    for (const rawLine of lines) {
      const line = rawLine.replace(/\r$/, ""); // fetch doesn't strip \r before \n like httpx does
      if (line === "") {
        event = "message";
        continue;
      }
      if (line.startsWith("event:")) {
        event = line.slice("event:".length).trim();
        continue;
      }
      if (!line.startsWith("data:")) {
        continue;
      }

      const data = line.slice("data:".length).trim();
      if (event === "start") {
        process.stdout.write(`[start] ${data}\n\n`);
      } else if (event === "done") {
        process.stdout.write(`\n\n[done] ${data}\n`);
      } else if (event === "error") {
        process.stdout.write(`\n\n[error] ${data}\n`);
      } else {
        process.stdout.write(extractToken(data));
      }
    }
  }

  const elapsed = Math.round((performance.now() - start) / 10) / 100;
  process.stdout.write(`\nClient-side wall time: ${elapsed}s\n`);
}

function extractToken(dataJson: string): string {
  try {
    return JSON.parse(dataJson).token ?? "";
  } catch {
    return "";
  }
}

const questionText = process.argv.slice(2).join(" ") || "Explain what Server-Sent Events are in three sentences.";
streamQuestion(questionText).catch((error) => {
  console.error(error);
  process.exit(1);
});
