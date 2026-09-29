import chalk from "chalk";

export function printSectionHeader(text: string, char: string = "="): void {
  const border = char.repeat(70);
  console.log(`\n${border}\n${text}\n${border}`);
}

export const since = (startedAt: number): string => `${((Date.now() - startedAt) / 1000).toFixed(1)}s`;

// One line per graph step, so the flowchart is visible while it runs.
export function logStep(agent: string, message: string): void {
  console.log(`${chalk.cyan(`[${agent}]`)} ${message}`);
}
