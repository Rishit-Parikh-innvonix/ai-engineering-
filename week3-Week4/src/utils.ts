import Table from "cli-table3";
import chalk from "chalk";

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

export function printComparisonTable(title: string, head: string[], rows: (string | number)[][]): void {
  const table = new Table({ head });
  for (const row of rows) {
    table.push(row.map((cell) => String(cell)));
  }
  console.log(chalk.bold.cyan(`\n${title}`));
  console.log(table.toString());
}
