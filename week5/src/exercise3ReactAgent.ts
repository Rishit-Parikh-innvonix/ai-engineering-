/**
 * Exercise 3: "Provide 2-4 tools to llm and create a react agent."
 *
 * createAgent (from the `langchain` package) is given all 3 tools (Wikipedia, the custom
 * weather tool, and a hand-built calculator) at once. Unlike exercises 1-2, nothing here
 * manually loops "call model -> run tool -> call model again" - createAgent builds and runs
 * that reason/act/observe loop itself; we just hand it a model and a tool list, and it decides
 * per question how many tools (zero, one, or several, in sequence) it actually needs.
 *
 * This used to be createReactAgent from @langchain/langgraph/prebuilt, which carries a real
 * @deprecated tag pointing here (checked node_modules/langchain's own type declarations to
 * confirm createAgent's minimal shape is a near drop-in: {model, tools} instead of {llm, tools},
 * same agent.invoke({messages}) / result.messages shape) - migrated rather than kept, since an
 * actual non-deprecated replacement exists and needs no new dependency or API key.
 *
 * Run: npm run exercise3
 */

import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { createAgent } from "langchain";

import { getCloudChatModel } from "./models.js";
import { wikipedia, getCurrentWeather, calculator } from "./tools.js";
import { printSectionHeader, runIfMain } from "./utils.js";

const QUESTIONS: string[] = [
  "What is the current weather in Tokyo, and what is that temperature multiplied by 2?",
  "Search for 'Alan Turing' and tell me what he's best known for.",
  "What is 47 times 89?",
];

export async function runExercise3(): Promise<void> {
  printSectionHeader("Exercise 3: ReAct agent with 3 tools (wikipedia, get_current_weather, calculator)", "#");

  const chat = getCloudChatModel();
  const agent = createAgent({ model: chat, tools: [wikipedia, getCurrentWeather, calculator] });

  for (const question of QUESTIONS) {
    printSectionHeader(`Question: ${question}`, "-");

    const result = await agent.invoke({ messages: [new HumanMessage(question)] });

    for (const message of result.messages) {
      // AIMessage.isInstance is LangChain's current (non-deprecated) type guard - it narrows
      // `message` to AIMessage with a real, correctly-typed tool_calls property, no cast.
      if (AIMessage.isInstance(message) && message.tool_calls && message.tool_calls.length > 0) {
        for (const toolCall of message.tool_calls) {
          console.log(`  -> agent called ${toolCall.name}(${JSON.stringify(toolCall.args)})`);
        }
      } else if (message.type === "tool") {
        console.log(`     observed: ${String(message.content).slice(0, 200)}`);
      }
    }

    const finalMessage = result.messages[result.messages.length - 1];
    console.log(`\nFinal answer:\n${finalMessage.content}`);
  }
}

runIfMain(import.meta.url, runExercise3);
