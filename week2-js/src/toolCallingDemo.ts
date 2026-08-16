/**
 * Bonus demo for the 'function / tool calling' topic (not one of the 3 graded exercises,
 * same role as week2's tool_calling_demo.py). Two tools are bound at once, and the demo
 * runs 3 different questions to show that the model decides for itself, per question, how
 * many tools (zero, one, or both) it actually needs before answering - the same tool set
 * doesn't mean the same tool calls every time. Run: npm run tool-calling-demo
 */

import { z } from "zod";
import { tool } from "@langchain/core/tools";
import { SystemMessage, HumanMessage, ToolMessage, type BaseMessage } from "@langchain/core/messages";

import * as config from "./config.js";
import * as models from "./models.js";

// Two fixed lookup tables standing in for what would normally be separate database/API calls.
const PRIORITY_POLICY: Record<string, string> = {
  billing: "Billing issues are handled within 2 business days, unless a duplicate charge is involved - then same-day.",
  technical: "Technical issues blocking work are 'urgent' and handled within 1 hour; non-blocking ones within 1 day.",
  account: "Account access issues (e.g. can't log in) are 'urgent' - handled within 1 hour.",
  feature_request: "Feature requests are logged for the roadmap and are never urgent.",
  bug: "Bugs are triaged within 1 business day and escalated to 'urgent' if they cause data loss.",
  other: "Unclassified issues default to 'medium' priority pending manual review.",
};

const PLAN_SLA_POLICY: Record<string, string> = {
  free: "Free plan customers have no guaranteed response SLA - best-effort support, typically within 3 business days.",
  pro: "Pro plan customers have a 4 business-hour response SLA.",
  enterprise: "Enterprise plan customers have a 30-minute response SLA, 24/7.",
};

const lookupPriorityPolicy = tool(
  async ({ issue_type }: { issue_type: string }) => {
    return PRIORITY_POLICY[issue_type] ?? "No policy on file for this issue type; default to 'medium' priority.";
  },
  {
    name: "lookup_priority_policy",
    description:
      "Return this company's support priority policy for a given issue type " +
      "(one of: billing, technical, account, feature_request, bug, other).",
    schema: z.object({ issue_type: z.string() }),
  }
);

const lookupPlanSla = tool(
  async ({ plan_tier }: { plan_tier: string }) => {
    return PLAN_SLA_POLICY[plan_tier] ?? "No SLA on file for this plan tier.";
  },
  {
    name: "lookup_plan_sla",
    description: "Return this company's response-time SLA commitment for a given customer plan tier (one of: free, pro, enterprise).",
    schema: z.object({ plan_tier: z.string() }),
  }
);

// Dispatch table so the loop below can execute whichever tool(s) the model actually asked
// for, by name, instead of assuming it's always the same single tool. Loosely typed on
// purpose - the two tools have different zod arg schemas, and at this point we're just
// routing a JSON-ish args object from the model to whichever tool's name matched.
interface CallableTool {
  invoke: (args: Record<string, unknown>) => Promise<string>;
}
const TOOLS = [lookupPriorityPolicy, lookupPlanSla];
const toolsByName: Record<string, CallableTool> = Object.fromEntries(TOOLS.map((t) => [t.name, t as unknown as CallableTool]));

const SYSTEM_PROMPT = new SystemMessage(
  "You are a support triage assistant. Use the lookup_priority_policy tool when you need to " +
    "decide a ticket's priority, and the lookup_plan_sla tool when the question involves a " +
    "customer's plan tier or SLA commitment. Use only the tools you actually need - not every " +
    "question needs both, and some need neither."
);

interface Scenario {
  label: string;
  question: string;
}

const SCENARIOS: Scenario[] = [
  {
    label: "Scenario A - expect ONE tool call (priority only, no plan mentioned)",
    question: "A customer says they can't log in and have a demo in an hour. What priority should this ticket get, and why?",
  },
  {
    label: "Scenario B - expect TWO tool calls (priority + SLA, both needed)",
    question:
      "An Enterprise plan customer says they were charged twice on their last invoice and wants to know if " +
      "we're within our SLA. What priority should this ticket get, and are we meeting our SLA commitment?",
  },
  {
    label: "Scenario C - expect ONE tool call (SLA only, no issue to prioritize)",
    question: "What is our response SLA for Pro plan customers?",
  },
];

const MAX_ROUNDS = 4;

/**
 * A single "round" is one request/response turn. Some questions need more than one round:
 * the model calls tool A, reads the result, and only THEN realizes it also needs tool B -
 * it doesn't always ask for every tool it'll need up front in one shot. So this loops
 * "send -> if tool_calls, run them and send again" until the model responds with plain
 * text and no further tool_calls (or MAX_ROUNDS is hit, as a safety cap against a model
 * that never stops requesting tools).
 */
type BoundChat = ReturnType<ReturnType<typeof models.getChatModel>["bindTools"]>;

/** The free OpenRouter model occasionally returns a malformed/empty response once a
 * conversation has several tool-result messages in it - a real API flakiness, not a bug
 * in the loop logic. Wrapped so one bad response ends this scenario cleanly instead of
 * crashing the whole multi-scenario script. */
async function safeInvoke(
  chatWithTools: BoundChat,
  messages: BaseMessage[],
  context: string
): Promise<Awaited<ReturnType<BoundChat["invoke"]>> | null> {
  try {
    return await chatWithTools.invoke(messages);
  } catch (error) {
    console.log(
      `\n(model call failed ${context}: ${error instanceof Error ? error.message : error} - ` +
        "stopping this scenario; this is API flakiness from the free model, not a loop bug.)"
    );
    return null;
  }
}

async function runScenario(chatWithTools: BoundChat, scenario: Scenario): Promise<void> {
  console.log(`\n${"=".repeat(70)}\n${scenario.label}\nQuestion: ${scenario.question}\n${"=".repeat(70)}`);

  const messages: BaseMessage[] = [SYSTEM_PROMPT, new HumanMessage(scenario.question)];
  const toolsUsed: string[] = [];

  console.log("Step 1 - sending the question with 2 tools available...");
  let aiMessage = await safeInvoke(chatWithTools, messages, "on the initial request");
  if (!aiMessage) return;
  messages.push(aiMessage);

  let round = 1;
  while (aiMessage.tool_calls && aiMessage.tool_calls.length > 0 && round <= MAX_ROUNDS) {
    console.log(`\nRound ${round} - model requested ${aiMessage.tool_calls.length} tool call(s):`);
    for (const toolCall of aiMessage.tool_calls) {
      console.log(`  - ${toolCall.name}(${JSON.stringify(toolCall.args)})`);
    }

    for (const toolCall of aiMessage.tool_calls) {
      const matchedTool = toolsByName[toolCall.name];
      const result = matchedTool
        ? await matchedTool.invoke(toolCall.args as never)
        : `Unknown tool requested: ${toolCall.name}`;
      console.log(`  executed ${toolCall.name} locally -> ${JSON.stringify(result)}`);
      messages.push(new ToolMessage({ content: result as string, tool_call_id: toolCall.id! }));
      toolsUsed.push(toolCall.name);
    }

    console.log("\nSending the tool result(s) back to the model...");
    const next = await safeInvoke(chatWithTools, messages, `after round ${round}`);
    if (!next) return;
    aiMessage = next;
    messages.push(aiMessage);
    round += 1;
  }

  if (toolsUsed.length === 0) {
    console.log(
      "\nThe model answered directly without calling any tool (this can happen if the free " +
        "model doesn't reliably support tool calling on OpenRouter, or if it judged no tool " +
        "was needed)."
    );
  } else {
    console.log(`\nUsed ${toolsUsed.length} tool call(s) across ${round - 1} round(s): ${toolsUsed.join(", ")}`);
  }

  console.log("\nFinal answer:");
  console.log(aiMessage.content || "(model stopped without producing a final text answer - see MAX_ROUNDS cap above)");
}

async function runDemo(): Promise<void> {
  const chat = models.getChatModel(config.TOOL_DEMO_MODEL, { temperature: 0 });
  const chatWithTools = chat.bindTools(TOOLS);

  for (const scenario of SCENARIOS) {
    await runScenario(chatWithTools, scenario);
  }
}

runDemo();
