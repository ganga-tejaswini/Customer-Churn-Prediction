import { Link } from 'react-router-dom'

export default function NotFoundPage() {
  return (
    <main className="mx-auto max-w-md px-6 py-24 text-center">
      <h1 className="text-2xl font-semibold">Page not found</h1>
      <p className="mt-2 text-[var(--muted)]">The page you requested does not exist.</p>
      <Link to="/" className="mt-6 inline-block font-medium text-brand-600 underline underline-offset-4">
        Go to home
      </Link>
    </main>
  )
}
