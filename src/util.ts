export const filterNils = <T>(arr: (T | null | undefined)[]): T[] =>
  arr.filter(x => x !== null && x !== undefined) as T[];
