/**
 * Exercise 1 - Prompt Experiment. Same underlying task, four different ways of
 * phrasing the prompt, run against the same model at the same temperature so the
 * only variable is the prompt itself. Run: npm run prompt-experiment
 */

import { HumanMessage, SystemMessage } from "@langchain/core/messages";

import * as config from "./config.js";
import * as models from "./models.js";

interface PromptVariant {
  label: string;
  technique: string;
  build: () => (SystemMessage | HumanMessage)[];
}

const VARIANTS: PromptVariant[] = [
  {
    label: "A - Naive / bare prompt",
    technique: "No persona, no format, no constraints - just the raw question.",
    build: () => [new HumanMessage(config.PROMPT_EXPERIMENT_TASK)],
  },
  {
    label: "B - Persona + audience",
    technique: "Adds a role for the model and names a specific audience.",
    build: () => [
      new SystemMessage("You are a patient computer science teacher explaining concepts to complete beginners."),
      new HumanMessage(`${config.PROMPT_EXPERIMENT_TASK} The learner has never written code before.`),
    ],
  },
  {
    label: "C - Format-constrained",
    technique: "Forces a specific length/structure instead of letting the model choose.",
    build: () => [
      new HumanMessage(
        `${config.PROMPT_EXPERIMENT_TASK} Answer in exactly 3 sentences: sentence 1 defines the ` +
          "concept in plain English, sentence 2 gives a real-world (non-code) analogy, sentence 3 " +
          "states the one thing beginners most often get wrong about it. No code, no bullet points."
      ),
    ],
  },
  {
    label: "D - Few-shot (worked example first)",
    technique: "Shows the model one worked example of the desired style/depth before asking for the real answer.",
    build: () => [
      new HumanMessage(
        "Here is an example of the style I want:\n\n" +
          'Q: Explain what a variable is to a beginner.\n' +
          'A: A variable is a labeled box you can put a value in and look at or change later. ' +
          "Think of it like a labeled jar in your kitchen - you can put sugar in it now and swap it " +
          "for flour later, and the label stays the same. Beginners often forget that assigning a new " +
          "value replaces what was there before, it doesn't add to it.\n\n" +
          `Now answer in the same style:\nQ: ${config.PROMPT_EXPERIMENT_TASK}\nA:`
      ),
    ],
  },
];

async function main(): Promise<void> {
  console.log(`Task: "${config.PROMPT_EXPERIMENT_TASK}"`);
  console.log(`Model: ${config.CLOUD_MODEL} (temperature 0.7, same for every variant)\n`);

  const chat = models.getCloudChatModel(0.7);

  for (const variant of VARIANTS) {
    console.log("=".repeat(70));
    console.log(`${variant.label}`);
    console.log(`Technique: ${variant.technique}`);
    console.log("=".repeat(70));

    const response = await chat.invoke(variant.build());
    console.log(response.content);
    console.log();
  }

  console.log(
    "Observations to fill in by hand after reading the four responses above:\n" +
      "- Which response was clearest for a total beginner?\n" +
      "- Which changed length/structure the most - the persona prompt, or the explicit format constraint?\n" +
      "- Did the few-shot example actually make the model match that style, or did it ignore it?\n" +
      "- Same model, same temperature, same underlying question - the only variable was wording.\n"
  );
}

main();
