/** Joins truthy class names. Small helper used by all UI components. */
export function cn(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(' ')
}
