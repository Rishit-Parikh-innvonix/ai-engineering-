/**
 * Shared logic for "bind ONE tool directly to the chat model and hand-run the request ->
 * tool_call -> tool_result -> final answer loop ourselves" - what exercises 1 and 2 both do.
 * This is what "bind the tool to llm" means before an agent framework is involved at all;
 * exercise 3's createReactAgent runs this same kind of loop internally instead.
 */

import type { StructuredToolInterface } from "@langchain/core/tools";
import { HumanMessage, SystemMessage, type BaseMessage } from "@langchain/core/messages";
import type { ChatOpenAI } from "@langchain/openai";

const MAX_ROUNDS = 3;

const GROUNDING_SYSTEM_PROMPT = new SystemMessage(
  "You have exactly one tool available. Answer the user's question using ONLY the tool's " +
    "result - never from your own memory, even if you think you already know the answer. " +
    "Always reply in the same language the user's question was written in."
);

export async function runBoundToolDemo(chat: ChatOpenAI, singleTool: StructuredToolInterface, question: string): Promise<void> {
  console.log(`Question: ${question}`);

  // The first call is forced to use the tool (tool_choice: "required") - otherwise a free
  // model can just skip a bound tool entirely and answer from its own (unverified) memory,
  // silently defeating the point of giving it a tool at all. Once a tool result exists in
  // the conversation, later calls go back to "auto" so the model can produce a normal final
  // answer instead of being forced to call the tool again forever.
  const forceToolCall = chat.bindTools([singleTool], { tool_choice: "required" });
  const chatWithTool = chat.bindTools([singleTool]);

  const messages: BaseMessage[] = [GROUNDING_SYSTEM_PROMPT, new HumanMessage(question)];
  let aiMessage = await forceToolCall.invoke(messages);
  messages.push(aiMessage);

  let round = 1;
  while (aiMessage.tool_calls && aiMessage.tool_calls.length > 0 && round <= MAX_ROUNDS) {
    for (const toolCall of aiMessage.tool_calls) {
      console.log(`  -> model called ${toolCall.name}(${JSON.stringify(toolCall.args)})`);
      // Passing the whole ToolCall (not just its .args) is the documented LangChain
      // convention: the tool then returns a ready-to-push ToolMessage itself (with
      // tool_call_id already set), instead of us hand-building one from a raw string result.
      const toolMessage = await singleTool.invoke(toolCall);
      console.log(`     result: ${toolMessage.content}`);
      messages.push(toolMessage);
    }
    aiMessage = await chatWithTool.invoke(messages);
    messages.push(aiMessage);
    round += 1;
  }

  // If the model is still issuing tool_calls after MAX_ROUNDS, it never produced a plain-text
  // answer - aiMessage.content would be empty/irrelevant, so say so explicitly instead of
  // printing a blank "Final answer:".
  if (aiMessage.tool_calls && aiMessage.tool_calls.length > 0) {
    console.log(`\n(Gave up after ${MAX_ROUNDS} rounds - model kept calling tools instead of answering.)`);
    return;
  }

  console.log(`\nFinal answer:\n${aiMessage.content}`);
}
