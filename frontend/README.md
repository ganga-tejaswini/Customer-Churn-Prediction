# Churn Insight: Frontend

Frontend for **Customer Churn Prediction in Telecom Industry Using Ensemble-Based Classifiers** (CSBS Team 03).
The ML models and prediction APIs are built by teammates; this app only consumes them.

## Stack
React 19 · TypeScript · Vite · Tailwind CSS v4 · React Router · Axios · Recharts · Lucide React · React Hook Form · Zod

## Prerequisites
- Node.js 20.19+ or 22+ (`node -v`)
- npm 10+

## Setup
```bash
npm install
cp .env.example .env      # Windows PowerShell: Copy-Item .env.example .env
npm run dev
```
Open http://localhost:5173.

## Scripts
| Command | Purpose |
|---|---|
| `npm run dev` | Start the dev server |
| `npm run build` | Type-check and create a production build |
| `npm run lint` | Lint the source |
| `npm run preview` | Serve the production build locally |

## Environment variables
| Variable | Default | Description |
|---|---|---|
| `VITE_APP_NAME` | Churn Insight | App name shown in the UI |
| `VITE_API_BASE_URL` | http://localhost:8000/api | Backend base URL (confirm with the backend team) |
| `VITE_API_TIMEOUT_MS` | 15000 | Request timeout |
| `VITE_USE_MOCKS` | false | Use labeled mock data while the backend is unavailable |

Variables are validated with Zod in `src/config/env.ts`. Restart the dev server after editing `.env`.

## Structure
```
src/
  app/          App root and router
  config/       Validated environment config
  layouts/      App shell layouts (Module 1)
  pages/        Route-level pages
  components/   Shared UI: ui/, charts/, tables/, forms/
  features/     One folder per module (dashboard, datasets, customers, predictions,
                models, explainability, analytics, retention, settings)
  services/     Axios client and API calls (no API code inside components)
  hooks/        Shared hooks
  types/        Shared TypeScript types
  schemas/      Zod schemas
  utils/        Helpers
  mocks/        Clearly labeled mock data
  styles/       Global CSS and theme tokens
```
Import from `src` using the `@` alias, e.g. `import { env } from '@/config/env'`.

## Conventions
- API calls live in `services/` (or `features/*/api`), never in components.
- Reuse components from `components/ui` before creating new ones.
- No ML logic or invented results in the frontend; mock data must be labeled as mock.
