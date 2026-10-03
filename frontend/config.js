// DepthWizard frontend configuration.
// If served separately on port 5173 (local frontend dev server), points to the backend at http://127.0.0.1:8765.
// If deployed live (e.g. Hugging Face Spaces, Render) or served from the main port, uses window.location.origin.
export const API_BASE_URL =
  typeof window !== 'undefined' && window.location.port === '5173'
    ? 'http://127.0.0.1:8765'
    : (typeof window !== 'undefined' ? window.location.origin : 'http://127.0.0.1:8765');

