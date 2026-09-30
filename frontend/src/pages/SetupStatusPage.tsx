import { CheckCircle2 } from 'lucide-react'
import { env } from '@/config/env'

const checks = [
  { label: 'React + TypeScript + Vite', detail: 'Application is rendering' },
  { label: 'Tailwind CSS', detail: 'Styles and theme tokens loaded' },
  { label: 'React Router', detail: 'Routing is active' },
  { label: 'Environment', detail: `Mode: ${env.mode}` },
  { label: 'API base URL', detail: env.apiBaseUrl },
]

/** Temporary Module 0 page. Replaced by the real layout and dashboard in Modules 1 and 2. */
export default function SetupStatusPage() {
  return (
    <main className="mx-auto max-w-2xl px-6 py-16">
      <h1 className="text-2xl font-semibold tracking-tight">{env.appName}</h1>
      <p className="mt-2 text-[var(--muted)]">
        Module 0 complete. The project foundation is set up and ready for the design system.
      </p>
      <ul className="mt-8 divide-y divide-[var(--border)] rounded-lg border border-[var(--border)] bg-[var(--surface)]">
        {checks.map((c) => (
          <li key={c.label} className="flex items-start gap-3 px-4 py-3">
            <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-brand-600" aria-hidden />
            <div className="min-w-0">
              <p className="font-medium">{c.label}</p>
              <p className="truncate text-sm text-[var(--muted)]">{c.detail}</p>
            </div>
          </li>
        ))}
      </ul>
    </main>
  )
}
