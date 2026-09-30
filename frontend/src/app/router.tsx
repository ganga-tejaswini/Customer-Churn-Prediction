import { createBrowserRouter } from 'react-router-dom'
import SetupStatusPage from '@/pages/SetupStatusPage'
import NotFoundPage from '@/pages/NotFoundPage'

/** Central route table. Later modules add their routes here (nested under the app layout). */
export const router = createBrowserRouter([
  { path: '/', element: <SetupStatusPage /> },
  { path: '*', element: <NotFoundPage /> },
])
