/*
 * frontend/src/App.tsx
 * ─────────────────────────────────────────────────────────────────────────────
 * PURPOSE:
 *     Root application component. Sets up React Router routes, global
 *     providers (TanStack Query, auth context), and the app shell layout.
 *
 * WHAT GOES HERE:
 *
 *     PROVIDERS (wrap everything):
 *         QueryClientProvider  — TanStack Query with 30s staleTime
 *         AuthProvider         — JWT token storage + user context
 *         RouterProvider       — React Router v6 with routes below
 *
 *     ROUTES:
 *         /                    → <Dashboard />    (requires auth)
 *         /claims/:claimId     → <ClaimDetail />  (requires auth)
 *         /review/:claimId     → <ReviewForm />   (requires REVIEWER role)
 *         /audit/:claimId      → <AuditTrail />   (requires auth)
 *         /settings            → <Settings />     (requires ADMIN role)
 *         /login               → <Login />        (public)
 *         *                    → <NotFound />
 *
 *     PROTECTED ROUTES:
 *         <ProtectedRoute role="REVIEWER"> wraps reviewer/admin pages.
 *         Redirects to /login if not authenticated.
 *         Redirects to / if authenticated but wrong role.
 *
 *     APP SHELL LAYOUT:
 *         <Sidebar />  — navigation links + user info + logout
 *         <TopBar />   — page title + notifications bell + SLA alert count
 *         <main>       — page content rendered here
 *
 *     GLOBAL NOTIFICATIONS:
 *         Toast system for: success actions, API errors, SLA alerts.
 *         Toasts appear in top-right corner.
 */
