import { fileURLToPath } from "node:url";
import chalk from "chalk";

// Lets an exercise file export its logic as a plain function (importable and reusable by
// runAll.ts) while still running that function automatically when the file is executed
// directly (`tsx src/exerciseN.ts`) - the standard ESM "is this the entry module" check.
export function runIfMain(moduleUrl: string, fn: () => Promise<void>): void {
  if (process.argv[1] === fileURLToPath(moduleUrl)) {
    fn();
  }
}

export function printSectionHeader(text: string, char: string = "="): void {
  const border = char.repeat(70);
  console.log(`\n${border}\n${text}\n${border}`);
}

export function printPanel(title: string, text: string): void {
  const width = Math.min(100, Math.max(title.length, ...text.split("\n").map((line) => line.length)) + 4);
  const border = "─".repeat(width);
  console.log(chalk.cyan(`┌${border}┐`));
  console.log(chalk.cyan("│ ") + chalk.bold(title));
  console.log(chalk.cyan(`├${border}┤`));
  for (const line of text.split("\n")) {
    console.log(chalk.cyan("│ ") + line);
  }
  console.log(chalk.cyan(`└${border}┘`));
}
