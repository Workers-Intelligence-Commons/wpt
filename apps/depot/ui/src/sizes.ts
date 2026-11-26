// Sizes and lengths of time in words, the same on every page: bytes in
// thousands (1.5 MB), tokens short (120M, 1.2B), a duration rounded to the minute.

import { plural } from "./parts";

const UNITS = ["bytes", "KB", "MB", "GB", "TB", "PB"];
export const bytes = (n: number) => {
  let value = n, unit = 0;
  while (value >= 1000 && unit < UNITS.length - 1) { value /= 1000; unit += 1; }
  return unit === 0 ? plural(n, "byte") : `${value.toFixed(value < 10 ? 1 : 0)} ${UNITS[unit]}`;
};
export const duration = (seconds: number) => {
  if (seconds < 60) return "less than 1 minute";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return plural(minutes, "minute");
  const hours = Math.floor(minutes / 60);
  return `${plural(hours, "hour")}${minutes % 60 ? ` ${plural(minutes % 60, "minute")}` : ""}`;
};
// A number of tokens, short: 950, 12K, 120M, 1.5B.
export const tokensShort = (n: number) => {
  for (const [size, unit] of [[1e12, "T"], [1e9, "B"], [1e6, "M"], [1e3, "K"]] as const)
    if (n >= size) { const v = n / size; return `${v < 10 && v % 1 ? v.toFixed(1) : Math.round(v)}${unit}`; }
  return n.toLocaleString("en");
};
