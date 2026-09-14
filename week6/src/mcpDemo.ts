/**
 * Week 6 - MCP, told as one story: building a tiny "personal assistant" AI.
 *
 * A personal assistant needs two basic abilities: it should be able to check your to-do list
 * (a real file on disk), and it should remember things about you between conversations (like
 * ChatGPT's own "memory" feature). Instead of writing that file-reading code and that
 * remembering code ourselves (which is what week5 did for its weather/calculator tools), this
 * week plugs in two ready-made helper programs - MCP servers - that already know how to do
 * those two jobs. That's the whole point of MCP: reuse an existing helper instead of building
 * one from scratch for every new ability you want.
 *
 * The two helpers used here (both official, from the team that created MCP itself, zero API
 * keys, zero network access - see README for why these were picked over the assignment's other
 * suggestions like "fetch a webpage", which turned out to have a real unpatched security hole):
 *   - @modelcontextprotocol/server-filesystem - reads/writes files, locked to sandbox/ only
 *   - @modelcontextprotocol/server-memory     - remembers facts across separate runs of this script
 *
 * Run: npm run mcp-demo
 */

import fs from "node:fs";
import path from "node:path";
import { AIMessage, HumanMessage, type BaseMessage } from "@langchain/core/messages";
import { MultiServerMCPClient } from "@langchain/mcp-adapters";
import { createAgent } from "langchain";

import { PROJECT_ROOT } from "./config.js";
import { getCloudChatModel } from "./models.js";
import { printPanel, printSectionHeader } from "./utils.js";

const SANDBOX_DIR = path.join(PROJECT_ROOT, "sandbox");
const TODO_FILE = path.join(SANDBOX_DIR, "todo.txt");

// A real, live incident: an earlier test run of this exact script was left running in the
// background and never shut down, which kept mcp-memory/memory.jsonl locked - the NEXT run then
// froze forever on its first memory tool call, with zero explanation, waiting on a lock that
// was never going to clear. Without a limit, "the assistant is thinking" and "something is stuck
// forever" look identical from the terminal. This bounds every assistant call so a stuck run
// fails loudly with a clear message instead of hanging silently.
const ASSISTANT_TIMEOUT_MS = 60_000;

// The memory helper saves what it remembers into this file, so it survives between separate
// runs of this script - just like your phone remembers your contacts even after you restart it.
// By default this helper tries to save inside its own node_modules folder instead (checked this
// directly - it's a bad spot, since that folder gets wiped every time you reinstall packages),
// so MEMORY_FILE_PATH below is pointed here instead, and mkdirSync makes sure the folder exists
// before the helper tries to write to it (it doesn't create its own folder and just fails if
// it's missing - found that out by actually hitting the error).
const MEMORY_FILE = path.join(PROJECT_ROOT, "mcp-memory", "memory.jsonl");
fs.mkdirSync(path.dirname(MEMORY_FILE), { recursive: true });

// This is the actual "connecting to MCP servers" step. Each entry below is the same as typing a
// command into your own terminal - the filesystem line is equivalent to running
// `npx mcp-server-filesystem "...\week6\sandbox"` yourself, which starts that helper as its own
// separate running program, locked to that one folder. MultiServerMCPClient just starts both
// helpers for you and gives you one remote control for talking to either of them.
const mcpClient = new MultiServerMCPClient({
  mcpServers: {
    filesystem: { command: "npx", args: ["-y", "mcp-server-filesystem", SANDBOX_DIR] },
    memory: { command: "npx", args: ["-y", "mcp-server-memory"], env: { MEMORY_FILE_PATH: MEMORY_FILE } },
  },
});

// Used after every question so you can SEE the assistant actually using a helper, instead of
// just trusting that its answer is correct. Same idea as double-checking a friend's homework by
// asking "ok but show me how you got that number," not just accepting the final answer.
function logToolCalls(messages: BaseMessage[]): void {
  for (const message of messages) {
    if (AIMessage.isInstance(message) && message.tool_calls && message.tool_calls.length > 0) {
      for (const toolCall of message.tool_calls) {
        console.log(`  -> assistant used ${toolCall.name}(${JSON.stringify(toolCall.args)})`);
      }
    }
  }
}

async function exercise1SeeWhatTheAssistantCanDo(): Promise<void> {
  printSectionHeader("Exercise 1: what can our assistant's two helpers actually do?");

  // Starting both helper programs and asking each one, separately, "what are you capable of?" -
  // like reading the manual for two gadgets right after unboxing them, before plugging them
  // into anything. initializeConnections() keeps each helper's list separate (instead of one
  // mixed list) specifically so you can see which ability came from which helper.
  const toolsByServer = await mcpClient.initializeConnections();

  for (const [serverName, tools] of Object.entries(toolsByServer)) {
    const plainEnglish = serverName === "filesystem" ? "reading/writing files" : "remembering facts";
    console.log(`\n${serverName} helper (${tools.length} abilities, all about ${plainEnglish}):`);
    for (const tool of tools) {
      console.log(`  - ${tool.name}: ${tool.description}`);
    }
  }
}

async function exercise2AssistantRemembersYou(): Promise<void> {
  printSectionHeader("Exercise 2: teach the assistant something about you, using ONLY the memory helper");

  // getTools("memory") hands the assistant ONLY the remembering abilities - on purpose, not the
  // file-reading ones. This mirrors how ChatGPT's own "memory" feature works: you tell it
  // something once, and it can bring that fact back up in a totally separate conversation later.
  const tools = await mcpClient.getTools("memory");
  const chat = getCloudChatModel();
  const assistant = createAgent({ model: chat, tools });

  const question =
    "Please remember two things about me: my name is Rishit, and I'm currently learning AI " +
    "engineering. Then tell me what you now remember about me.";
  console.log(`You say: ${question}`);

  const result = await assistant.invoke({ messages: [new HumanMessage(question)] }, { timeout: ASSISTANT_TIMEOUT_MS });
  logToolCalls(result.messages);
  const finalMessage = result.messages[result.messages.length - 1];
  console.log(`\nAssistant replies:\n${finalMessage.content}`);
}

async function exercise3AssistantChecksYourTodoListAndRemembers(): Promise<void> {
  printSectionHeader("Exercise 3 (optional): one assistant, both helpers, one real task");

  // No specific helper named here - the assistant gets BOTH the file-reading and the
  // remembering abilities at once, and decides for itself which one(s) it actually needs.
  const tools = await mcpClient.getTools();
  const chat = getCloudChatModel();
  const assistant = createAgent({ model: chat, tools });

  // A real personal assistant would need to do exactly this: check a real file, then remember
  // the important part of it. Neither helper alone could answer this - reading the file needs
  // the filesystem helper, and remembering the priority afterward needs the memory helper.
  const question =
    `Check my to-do list at ${TODO_FILE}, remember my #1 priority task from it, and then tell ` +
    "me both what my priority task is and what else you remember about me.";
  console.log(`You say: ${question}`);

  const result = await assistant.invoke({ messages: [new HumanMessage(question)] }, { timeout: ASSISTANT_TIMEOUT_MS });
  logToolCalls(result.messages);
  const finalMessage = result.messages[result.messages.length - 1];
  console.log(`\nAssistant replies:\n${finalMessage.content}`);
}

async function main(): Promise<void> {
  printPanel(
    "Week 6: MCP - Building a Tiny Personal Assistant",
    "1) see its abilities  2) it remembers you  3) it checks a real file AND remembers"
  );

  try {
    await exercise1SeeWhatTheAssistantCanDo();
    await exercise2AssistantRemembersYou();
    await exercise3AssistantChecksYourTodoListAndRemembers();
  } catch (error) {
    // Confirmed live, twice: a "stuck" run has two real, different causes, not one -
    // don't guess which one it is, tell the user how to check both.
    console.error(`\nSomething went wrong: ${error instanceof Error ? error.message : error}`);
    console.error(
      "\nThis is most likely one of two things:\n" +
        "1. The free OpenRouter model itself is slow or temporarily overloaded right now (this " +
        "is\n" +
        "   an OpenRouter/model issue, not a bug here - confirmed by testing the model directly " +
        "with\n" +
        "   no MCP involved at all and seeing the exact same hang). Just try again in a bit.\n" +
        "2. A leftover copy of this demo from an earlier attempt is still running and holding\n" +
        "   mcp-memory/memory.jsonl locked. Check for it:\n" +
        "     Windows (PowerShell): Get-CimInstance Win32_Process -Filter \"Name='node.exe'\" | " +
        "Select-Object ProcessId, CommandLine\n" +
        "     macOS/Linux:           ps aux | grep mcp-server\n" +
        "   ...and close any match before trying again."
    );
    process.exitCode = 1;
    return;
  } finally {
    // Both helpers are real separate programs still running in the background - this shuts
    // them down so your terminal actually returns control to you instead of hanging forever.
    await mcpClient.close();
  }
}

main();
