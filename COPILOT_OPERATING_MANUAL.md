# COPILOT_OPERATING_MANUAL.md
> Paste this into **VS Code → Copilot Chat** as the first message in the repo.  
> Copilot: **you are the principal engineer**. The human is non-technical. You must generate **fully functional code (no placeholders)**, wire all logic end-to-end, and keep everything editable from the Admin Panel. Always **read `PROMPT.md` and `REQUIREMENTS.md`** before any task and keep a live status log.

---

## 0) SOURCE OF TRUTH
- Always read **`PROMPT.md`** and **`REQUIREMENTS.md`** first.
- Keep **`STATUS.md`** updated (what’s done, in-progress, next).
- Keep **`CHANGELOG.md`** updated (user-friendly entries).
- Respect **Definition of Done** in requirements.

---

## 1) WHAT YOU (COPILOT) MUST DO
- Generate **complete, production-grade code** with real logic. **No TODOs, no placeholders**.
- Use the opinionated stack from `REQUIREMENTS.md`: Next.js 14 (App Router), TS, Tailwind + shadcn/ui, next-intl, Prisma + PostgreSQL, tRPC, next-auth (email OTP + optional Google) + **TOTP 2FA for Admin**, Workbox PWA (offline + background sync), MapLibre, Recharts, S3 storage, SMTP/SMS/Web Push, etc.
- **Admin Panel edits EVERYTHING** (pages/blocks, media, menus, colors/fonts, strings/translations, SEO, PWA assets, notifications, datasets). Implement schema-driven blocks and theme editor (colors, fonts) that write to DB + CSS variables.
- **Accessibility** (WCAG 2.2 AA), **Security** (OWASP), **Performance** (Lighthouse ≥95 for PWA/Perf/SEO/A11y).
- Implement **all workflows**: complaints+SLA, eligibility checker (rules engine), projects+budgets+milestones map, pledges, events+RSVP, notices, directory+jobs, open-data exports, notifications center, translations admin, backups, feature flags, etc.
- Provide **tests**: unit (Jest), component (RTL), E2E (Playwright), a11y (axe), perf (Lighthouse CI).
- Provide **CI/CD** (GitHub Actions) and **deployment** (Vercel + Docker Compose for any DigitalOcean droplet).
- Use **open-source icons/images** (Heroicons/Lucide/Tabler for icons; unDraw/SVGs packaged locally with attributions). Add `public/ATTRIBUTIONS.md`.

---

## 2) STYLE & QUALITY RULES
- Type-safe end-to-end with Zod + tRPC. No `any`.
- Split code by domains; **no massive files**. Keep server-only code in server files.
- Strong RBAC checks **server-side** + UI gating. All mutating endpoints log **AuditLog**.
- Secure uploads (signed URLs, MIME/size checks). No secret leakage. Add CSP headers.
- All visible copy via i18n keys (Hindi default). Language toggle persists via cookie.
- **Admin Theme Editor** writes tokens to DB; layout injects CSS variables at runtime.
- **No dead code**, **no unused deps**. Run lint/typecheck before finishing a PR.

---

## 3) BRANCH/PR WORKFLOW (ALWAYS)
- Branch per PR: `feat/<short-scope>` or `chore/<scope>` or `fix/<scope>`.
- Commit convention: `type(scope): message` (types: feat, fix, chore, refactor, test, docs, ci).
- Each PR includes: description, screenshots, **checklist**, and links to passing CI.
- Update `STATUS.md` and `CHANGELOG.md` in PR.
- Do not merge if Lighthouse CI < 95 or a11y has critical issues.

**PR Plan (sequential)**
Follow the full plan from `REQUIREMENTS.md §25`:
PR0 Bootstrap → PR1 DB/Auth → PR2 PWA → PR3 Design System → PR4 Admin Shell → PR5 Content Manager → PR6 Form Builder → PR7 Geo & Maps → PR8 Projects & Budgets → PR9 Smart Mission → PR10 Schemes & Eligibility → PR11 Services & Requests → PR12 News/Notices/Events → PR13 Directory & Economy → PR14 Health/Education/Social → PR15 Tourism → PR16 Volunteer/Donate → PR17 Open Data → PR18 Notifications Center → PR19 Translations Admin → PR20 Theming & Settings → PR21 Analytics/SEO → PR22 Seeds → PR23 Hardening/Tests → PR24 Docs & Release.

---

## 4) FILES YOU MUST CREATE & MAINTAIN
- `STATUS.md` — live checklist per PR (Done/In Progress/Next).
- `CHANGELOG.md` — human-readable changes.
- `docs/` — README, Admin Guide, Content Guide, Data Import, I18N, Design System, Ops Runbook.
- `.env.example` — full env list.
- `docker-compose.yml` — Postgres + MinIO + Mailhog (dev) and **production variant** for DigitalOcean (Caddy reverse proxy + auto TLS optional).
- `vercel.json` — if needed for headers/caching.
- `scripts/{seed.ts,backup.ts,restore.ts}` — working scripts (no placeholders).
- `prisma/schema.prisma` + migrations + seeds with real demo data per §18.
- `public/ATTRIBUTIONS.md` — licenses/credits for icons/images.

---

## 5) HUMAN STEPS (NON-PROGRAMMER)
> Run these in the Terminal inside VS Code. Copilot generates the code.

### Step A — Bootstrap the repo (let Copilot do PR0 end-to-end)
1. **Create empty repo** and open in VS Code.  
2. Open **Copilot Chat** and paste this entire manual.  
3. Send this command to Copilot:

SYSTEM: Read PROMPT.md and REQUIREMENTS.md. Create PR0 as specified: bootstrap Next.js 14 + TS + Tailwind + shadcn/ui + next-intl (hi-IN default, en-IN), basic layout with a11y, ESLint/Prettier/Husky/commitlint, initial PWA manifest/icons, and docs/STATUS scaffolding. Provide exact terminal commands I must run. No placeholders. Then open a PR with screenshots and checklist.

4. Follow the terminal steps Copilot outputs (e.g., `pnpm install`, `pnpm dev`).

### Step B — Set up database locally (dev)
- Copilot will output a `.env` and `docker-compose.yml`. Run:

docker compose up -d pnpm prisma migrate deploy pnpm seed pnpm dev

- Open: http://localhost:3000 and Admin at `/admin`.

### Step C — Set up GitHub + CI
- Commit & push. Copilot will create GitHub Actions. Ensure CI runs and is green.

### Step D — DigitalOcean droplet deploy (production with Docker)
Ask Copilot for the **production** compose and commands:

SYSTEM: Generate a prod-ready docker-compose.yml for a single DO droplet with:

app (Next.js build then run),

postgres,

minio (S3),

caddy (reverse proxy + auto HTTPS via Let's Encrypt). Include healthchecks, volumes, envs, and a one-time init script to run prisma migrations + seeds. Provide exact shell commands to build/start, how to set DNS, and how to rotate secrets/backups. No placeholders; working defaults.


Run the commands it outputs on your droplet (via SSH).

---

## 6) “RUN” PROMPTS (YOU WILL USE THESE IN ORDER)

### PR0 — Bootstrap & DX

SYSTEM: Execute PR0. Create Next.js 14 TS app (App Router), Tailwind, shadcn/ui themed with tokens, next-intl (hi/en with cookie persist), accessibility foundations (skip links, focus rings), PWA manifest+icons, ESLint/Prettier/Husky/commitlint. Add STATUS.md & CHANGELOG.md. Provide commands to run and screenshots. Verify: pnpm build passes, Lighthouse PWA shell ≥ 90, axe no critical issues.

### PR1 — Database & Auth (OTP + TOTP 2FA + RBAC)

SYSTEM: Execute PR1. Implement Prisma schema (all entities in REQUIREMENTS.md §8 + extra Notification/DeliveryStatus/AuditLog/etc.), migrations, seeds. Add next-auth with email OTP and optional Google, add TOTP 2FA for Admin roles. Implement server-side RBAC helpers and middleware. Add secure session config. Provide seeds for an Admin account. Tests for RBAC and auth flows. No placeholders.

### PR2 — PWA & Service Worker

SYSTEM: Execute PR2. Implement Workbox strategies (CacheFirst static, StaleWhileRevalidate pages/data, NetworkOnly+BG Sync for forms). Add offline page, install prompt, offline indicator, and queued submission replays. Add tests and Lighthouse CI config to enforce ≥95.

### PR3 — Design System

SYSTEM: Execute PR3. Theme shadcn/ui, Tailwind tokens per brand. Build component gallery (announcement bar, hero, KPIs, progress rings, cards, tabs, accordions, timeline, breadcrumbs, share, emergency CTA, feedback button). Respect prefers-reduced-motion. Add docs/Design System with tokens and usage.

### PR4 — Admin Shell

SYSTEM: Execute PR4. Secure /admin with RBAC + 2FA mgmt. Sidebar, breadcrumbs, profile, audit log viewer, roles manager. Tests for guards and audit logging.

### PR5 — Content Manager (Blocks) + Media Library

SYSTEM: Execute PR5. Build schema-driven Page→Section→Block editor with versioning, preview, staging→approval→publish→rollback. Media library with alt/caption required, signed uploads (MinIO/S3), PDF viewer. Home page built entirely from blocks. Tests included.

### PR6 — Form Builder & Submissions

SYSTEM: Execute PR6. Visual builder → JSON schema + Zod; conditional logic; uploads; hCaptcha; rate limits. Public endpoints auto-generated; queues with SLA config and assignment rules. "My Requests" dashboard. End-to-end tests for Complaint flow.

### PR7 — Geo & Maps

SYSTEM: Execute PR7. MapLibre with ward boundaries, projects, issues, resources. Admin GeoJSON CRUD, CSV/GeoJSON import, photo geotagging. Ensure map is fast and accessible. Tests for geo import/export.

### PR8 — Projects & Budgets

SYSTEM: Execute PR8. Public project list/detail; budgets vs spent; milestones; ward map; contractors; docs; moderated comments; change log. Budget Explorer (Recharts) + CSV export. Admin CRUD. Tests.

### PR9 — Smart & Carbon-Free Mission

SYSTEM: Execute PR9. Carbon Calculator with real formulas (household energy, LPG, transport); Solar Wizard (roof size→kW→CAPEX→payback with scheme linkage); Tree Pledge Wall (moderated); Waste Segregation drag-drop mini-game; Water Tracker (RWH planner). Tests and a11y checks.

### PR10 — Schemes & Eligibility

SYSTEM: Execute PR10. Schemes cards; Eligibility Checker wizard using rules engine (criteria JSON); results with next steps/docs; 1-click start application (prefill). Admin rules editor. Tests for rule outcomes.

### PR11 — Services & Requests

SYSTEM: Execute PR11. Complaints, suggestions, RTI, certificates, waste pickup, water tanker, grievance appeal. SLA timers, escalations, multi-channel notifications. Full E2E tests.

### PR12 — News/Notices/Events

SYSTEM: Execute PR12. News/blog, Notices with deadlines (PDF viewer), Events calendar with RSVP, reminders, ICS export. Tests.

### PR13 — Directory & Economy

SYSTEM: Execute PR13. SHGs/artisans/businesses listings; profiles; products; enquiries; job board; trainings. Admin approval flow. Tests.

### PR14 — Health/Education/Social

SYSTEM: Execute PR14. Content pages for clinics, ANM/ASHA, schools/anganwadi, scholarships. Global emergency CTA persistent and accessible. Tests.

### PR15 — Tourism & Culture

SYSTEM: Execute PR15. Attractions/treks/homestays with route map; Responsible Tourism tips; gallery. A11y/perf tests.

### PR16 — Volunteer & Donate

SYSTEM: Execute PR16. Opportunities; signup; donation intents (materials/tools/trees) with transparency ledger; optional UPI intent deep link. Tests.

### PR17 — Open Data & Reports

SYSTEM: Execute PR17. CSV/JSON datasets; dashboards; village profile PDF; monthly progress report generator; API tokens. Tests.

### PR18 — Notifications Center

SYSTEM: Execute PR18. Compose templates per channel (email/SMS/WhatsApp/web push), audience segmentation by ward/role/interest, scheduling, delivery tracking. Adapter pattern for SMS vendor. Tests.

### PR19 — Translations Admin

SYSTEM: Execute PR19. String catalog manager, side-by-side translator, missing keys detector, CSV import/export. Enforce externalized strings. Tests.

### PR20 — Theming & Settings (Admin-Editable)

SYSTEM: Execute PR20. Theme editor (colors/fonts/logos/menus/SEO defaults/OG/PWA assets/cookie text). Persist to DB; inject CSS variables at runtime. No redeploy required. Tests.

### PR21 — Analytics & SEO

SYSTEM: Execute PR21. Privacy-friendly analytics; JSON-LD schemas; sitemaps; robots; canonical; OG/Twitter. Rich Results validation tests.

### PR22 — Seeding & Fixtures

SYSTEM: Execute PR22. Seed data as per REQUIREMENTS.md §18. Ensure coherent cross-links (projects↔milestones↔map, schemes↔eligibility forms). Screenshots in PR.

### PR23 — Hardening & Tests

SYSTEM: Execute PR23. Security headers (CSP/HSTS/XFO/XSS/Referrer-Policy), upload sanitization, rate limiting, CSRF, brute force guard. Expand unit/component/E2E coverage. Lighthouse budgets enforced. Document threat model briefly.

### PR24 — Docs & Release

SYSTEM: Execute PR24. Complete docs (README, Admin Guide, Content Guide, Data Import, I18N, Design System, Ops Runbook). Fill .env.example. Produce Vercel & DigitalOcean Docker deployment steps. Tag release. Attach acceptance checklist and final Lighthouse reports.

---

## 7) DIGITALOCEAN DEPLOYMENT (EXPECTATIONS FOR COPILOT OUTPUT)
- **Prod `docker-compose.yml`** with:
  - `app`: Multi-stage build (pnpm install/build) + run; env injection; healthcheck.
  - `postgres`: Volume; init; backups.
  - `minio`: S3-compatible storage; buckets for public/private; init user.
  - `caddy`: Reverse proxy with automatic HTTPS (map `SERVER_NAME` env), HTTP/2; security headers.
  - Optional `mailhog` for test email in staging.
- **Scripts**:
  - `scripts/migrate-and-start.sh` to run `prisma migrate deploy && pnpm start`.
  - `scripts/backup.sh` and `scripts/restore.sh` for DB + objects.
- **Docs**:
  - DNS setup steps, firewall ports (80/443/22), system updates, rotate secrets guide.

---

## 8) OPEN-SOURCE ASSETS & LICENSING
- Icons: **Heroicons**, **Lucide**, or **Tabler** (MIT). Bundle via npm and tree-shake.
- Illustrations: **unDraw** SVGs (credit in `public/ATTRIBUTIONS.md`).
- Photos: Provide local sample photos with clear attribution file; real content replaceable from Admin.
- Include `public/ATTRIBUTIONS.md` and show credits in `/about`.

---

## 9) ACCEPTANCE CHECKS (ENFORCE PER PR)
- `pnpm build` passes; typecheck/lint clean.
- a11y: axe shows no critical issues.
- Perf: Lighthouse CI budgets met (LCP < 2.5s, CLS < 0.1, TTI < 3.5s).
- Security: RBAC enforced; CSRF; rate limits; audit logs present; uploads safe.
- i18n: Strings externalized; Hindi default; toggle persists.
- Admin: Can edit corresponding features without code changes.
- Docs: Updated for any new feature.

---

## 10) STATUS FORMAT (UPDATE EVERY PR)
`STATUS.md` should look like:
```md
# STATUS — Ummid Se Hari
## Completed
- PR0 Bootstrap ✅
- PR1 DB/Auth ✅
## In Progress
- PR2 PWA & SW ⏳
## Up Next
- PR3 Design System


---

11) QUICK COMMANDS (DEV)

Copilot must generate and confirm these work:

pnpm i
docker compose up -d         # starts Postgres, MinIO, etc.
pnpm prisma migrate dev
pnpm seed
pnpm dev
pnpm test
pnpm build


---

12) GUARANTEES

No placeholders; provide working code/config/scripts.

Everything editable via Admin Panel (content, blocks, strings, theme, SEO, menus, PWA).

App is PWA installable with offline home + key pages and background sync for forms.

Easy deployment to Vercel and a single DigitalOcean droplet via Docker.


> Begin with PR0 now. Read PROMPT.md + REQUIREMENTS.md. Produce a step-by-step plan, create files, generate code, and return the exact terminal commands the human must run.
