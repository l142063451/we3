# REQUIREMENTS.md — “उम्मीद से हरी | Ummid Se Hari” (Smart & Carbon-Free Village PWA)

**Audience:** GitHub Copilot & contributors (engineers, designers, PMs, QA).  
**Objective:** End-to-end, production-ready PWA for Damday–Chuanala Gram Panchayat with a full-power Admin Panel that edits **every** public component.  
**Locales:** `hi-IN` (default), `en-IN` (secondary).  
**Compliance:** WCAG 2.2 AA, privacy-first, secure by design, offline-first.  
**Deployment target:** Vercel (preferred) or Docker on VPS.

---

## 0) Context & Vision

**Jurisdiction**
- Village: **Damday** · Post: **Chuanala** · Tehsil: **Gangolihat** · District: **Pithoragarh** · State: **Uttarakhand, India**
- Authority: **Gram Panchayat, led by Gram Pradhan**

**Vision:** Inspire (“उम्मीद”) and enable transparent, participatory, carbon-conscious governance.

**Measurable Goals**
1. 3× civic participation in 6 months (feedback, ideas, volunteering).
2. 15% household carbon-footprint proxy reduction in 12 months (pledges + actions).
3. 90% complaints resolved within SLA (7/14/30 days by category).
4. 100% projects tracked (budget, progress, geotagged milestones).
5. 80% content maintainers onboarded to Admin Panel within 30 days.

**Guiding Principles:** Hope-driven design, inclusivity, transparency, low data usage, offline-first, privacy by default.

---

## 1) Scope

### In-Scope (MVP+)
- Public website with all sections in **§6**.
- **Admin Panel (`/admin`)**: Content Manager, Form Builder, Geo Manager, Projects/Budgets, Schemes/Eligibility, Users & Roles, Moderation, Translations, Notifications, Theme & Settings, Reports, Directory, Backups/Integrations.
- PWA capabilities: installable, offline shell + key pages, background sync for forms.
- Auth: Email OTP (passwordless) + optional Google; **TOTP 2FA for admins**.
- DB: PostgreSQL, Prisma ORM.
- API: Next.js Route Handlers with **tRPC** (typed), REST fallback where needed (uploads, webhooks).
- Notifications: Email, SMS (India-ready adapter), optional WhatsApp, Web Push.
- Maps: MapLibre GL (OSM tiles), GeoJSON layers (wards, projects, issues, trees).
- Charts: Recharts (budgets, progress, carbon pledges).
- i18n: `next-intl` or `i18next`.
- CI/CD: GitHub Actions with lint/typecheck/test/E2E/Lighthouse/CodeQL/Deploy.

### Out-of-Scope (initial)
- Native mobile apps (PWA only).
- Payment gateway beyond UPI intent for donation intents (no funds custody).
- Large-scale GIS analysis (basic layers + pins only).

---

## 2) Personas & Key User Stories

**Citizens (anonymous/authenticated)**
- File/track complaints with photos & geotags.
- Check scheme eligibility, apply, and track.
- Explore projects, budgets, milestones on a map.
- Take green pledges; see pledge wall.
- RSVP to events; get reminders.
- Browse directory (SHGs, artisans, businesses), jobs, trainings.

**Panchayat Staff (Admin, Approver, Editor, DataEntry, Viewer)**
- Create/edit/publish content with versioning and preview.
- Build/modify forms, set SLAs, assign tickets, moderate content.
- Manage projects, budgets, payments, contractors; publish updates.
- Manage schemes and eligibility rules.
- Send announcements to segments (ward/role/interest).
- Manage translations, theming, integrations; view analytics.

---

## 3) Architecture & Tech Stack

- **Frontend**: Next.js 14 (App Router, RSC) + TypeScript + Tailwind CSS + shadcn/ui + Framer Motion (≤150ms) + TanStack Query.
- **PWA**: Workbox; manifest; offline fallback; background sync queue.
- **Forms/State**: React Hook Form + Zod; Zustand for light global state.
- **Maps/Viz**: MapLibre GL + Recharts.
- **Backend**: Next.js Route Handlers + **tRPC** routers (typed end-to-end).
- **DB/ORM**: PostgreSQL + Prisma.
- **Auth**: next-auth (Email OTP + Google optional), **TOTP 2FA** for admin roles.
- **Storage**: S3 compatible (MinIO for dev), signed URLs, Next/Image optimization.
- **Notifications**: SMTP; SMS adapter; WhatsApp optional; Web Push.
- **i18n**: next-intl / i18next; cookie persistence; language tags in HTML.
- **Analytics/SEO**: Privacy-friendly analytics; JSON-LD; OG/Twitter; sitemap; robots.
- **Testing**: Jest/RTL (unit), Playwright (E2E), Lighthouse CI, axe a11y.
- **CI/CD**: GitHub Actions (Verify/E2E/Lighthouse/CodeQL/Deploy), Vercel.

---

## 4) Design System

**Theme**: “Ummid Se Hari” (Hope-Green)
- Colors: Primary `#16A34A`, Accent `#F59E0B`, Trust `#2563EB`, Text `#1F2937`, BG `#F8FAFC`, Surface `#FFFFFF`, Success `#22C55E`, Warning `#F59E0B`, Error `#DC2626`, Info `#0EA5E9`.
- Typography: Headers — Poppins/Inter + Noto Sans Devanagari; Body — Inter + Noto Sans Devanagari.
- UI Tokens: 8px grid, radius 12–16px, gentle shadows, motion ≤ 150ms, respects `prefers-reduced-motion`.
- Components: Announcement bar, hero, KPIs, cards, tabs, accordions, timeline, progress rings, breadcrumb, search, share, footer, emergency CTA, feedback button, mega-menu.
- Microcopy tone: respectful, encouraging, bilingual labels.

**Acceptance**
- Tailwind theme tokens defined; shadcn components themed.
- A11y verified with axe; contrast ≥ 4.5:1 where appropriate.

---

## 5) Navigation & Sitemap

Global nav with mega-menu + persistent search + language switcher + contrast toggle.

- Home (डैशबोर्ड सारांश)
- About Village
- Gram Panchayat & Governance
- Smart & Carbon-Free Mission
- Schemes & Benefits
- Services & Requests
- Projects & Budgets
- News, Notices & Events
- Directory & Economy
- Health, Education & Social
- Tourism & Culture
- Volunteer & Donate
- Open Data & Reports
- Contact & Help
- Account

**Deep Links/Flows**  
- Scheme → Eligibility → Prefilled Application → Uploads → Track.  
- Carbon Calculator → Solar Wizard / Tree Pledge / Segregation Tips.  
- Project → Tenders + Budgets + Map + Beneficiary Surveys.  
- SHG Directory → Products → Enquire/Events.  
- Event → RSVP → Calendar → Volunteer → Reminders.

---

## 6) Feature Requirements (Public)

### 6.1 Home
- Hero (slogan + Gram Pradhan welcome), notices ticker.
- Quick actions: File Complaint, Check Scheme, Pledge Tree, Track Project.
- KPI Row (complaints resolved %, on-time projects %, pledge count, trees planted, water saved).
- Mini-map (projects/issues/resources).
- This week: events, tenders, new schemes, success stories.
- CTA: Join the Mission (volunteer/donate/suggest).

### 6.2 Smart & Carbon-Free Mission
- Roadmap timeline.
- **Carbon Footprint Calculator**: household inputs → tips.
- **Solar Adoption Wizard**: roof → kW → costs/benefits → schemes.
- **Tree Pledge Wall**: moderated.
- **Waste Segregation Game** (simple drag-drop).
- **Water Use Tracker** (RWH planning checklist).
- Monthly progress charts.

### 6.3 Schemes & Benefits
- Filterable cards by sector; detail pages.
- **Eligibility Checker** (wizard, rule engine).
- How-to apply, docs checklist, FAQs.

### 6.4 Services & Requests
- Forms: complaint, suggestion, RTI, certificate, waste pickup, water tanker, grievance appeal.
- Status tracker (ticket ID, timeline, SLA indicator).
- My Requests dashboard (for logged-in users).

### 6.5 Projects & Budgets
- Project listing with filters/tags.
- Detail: description, sanction, budget vs spent, milestones (dates/photos), ward map, contractor info, docs, moderated comments, change log.
- **Budget Explorer**: Sankey/Bar + CSV export.

### 6.6 News, Notices & Events
- News/blog/press cards.
- Notices (tenders, office orders) with deadlines; PDF viewer.
- **Events** calendar (RSVP, reminders, ICS export).

### 6.7 Directory & Economy
- SHGs & artisans: profiles, products/services, contact.
- Local businesses: categories, map pins, enquiry.
- Job board: posts, apply/enquire.
- Trainings/workshops: signups.

### 6.8 Health, Education & Social
- Clinic days, ANM/ASHA info; schools/anganwadi; scholarships; emergency numbers (global CTA).

### 6.9 Tourism & Culture
- Attractions, treks, homestays; map; responsible tourism tips; gallery.

### 6.10 Volunteer & Donate
- Skills/time signup; opportunities list.
- Donation intents (materials/tools/trees); transparency ledger.

### 6.11 Open Data & Reports
- Downloads (CSV/JSON); dashboards; village profile PDF.
- Monthly progress report generator.

### 6.12 Account
- Profile, language, notifications (email/SMS/WhatsApp), subscriptions.
- Request history; saved items.

---

## 7) Admin Panel (`/admin`) — Edit **Everything**

**RBAC Roles:** Admin, Approver, Editor, DataEntry, Viewer  
**Matrix:** See §15

### 7.1 Content Manager
- Pages → Sections → Blocks (schema-driven).
- WYSIWYG + block editor, drag-drop reorder, show/hide by locale/device.
- Media library with required alt/captions; PDF viewer.
- Versioning, preview, staging → approval → publish; rollback.

### 7.2 Form Builder
- Field palette; conditional logic; Zod validators; file upload controls.
- Auto-generate public endpoints; hCaptcha; rate limiting.
- SLA per category; assignment rules; queues.

### 7.3 Map & Geo Manager
- CRUD layers: wards, projects, issues, trees, resources (GeoJSON).
- Bulk CSV/GeoJSON import; photo geotagging.

### 7.4 Projects & Budgets
- CRUD projects, milestones, contractors.
- Budgets, payments; docs/attachments; public/private notes.

### 7.5 Schemes & Eligibility
- Rule editor; docs required; process steps; link to forms.

### 7.6 Users & Roles
- Invite users; set roles; enforce 2FA for admins; session invalidation.

### 7.7 Moderation
- Queues: complaints, suggestions, comments, directory entries, pledge wall.
- Approve/reject with reasons; auto-notify.

### 7.8 Translations
- String catalogs; page copy; side-by-side translation editor; missing string detector.

### 7.9 Notifications Center
- Compose announcements (email/SMS/WhatsApp/web push).
- Targeting by ward/role/interest; schedule; delivery tracking.

### 7.10 Theme & Settings
- Logos, colors, fonts, header/footer, menu, social links.
- SEO defaults, OG images, PWA icons, offline pages, cookie/consent text.

### 7.11 Reports & Analytics
- SLA compliance, participation, project progress, carbon pledges, site traffic.
- Export CSV/PDF.

### 7.12 Directory & Economy
- Approve listings; manage job posts, trainings.

### 7.13 Backups & Integrations
- One-click backup/restore (admin only).
- Configure SMTP/SMS/WhatsApp, map tiles, storage, feature flags, UPI intent.

---

## 8) Data Model (Prisma Outline)

Implement enums, indexes, foreign keys, soft deletes as needed. JSON fields validated via Zod.

- User(id, name, email, phone, locale, roles[], twoFAEnabled, createdAt)
- Role(id, name, permissions[])
- Page(id, slug, title, locale, status, blocks[], seo, createdBy, updatedBy, version)
- Media(id, url, alt, caption, meta, createdBy)
- Form(id, name, schemaJSON, slaDays, workflowJSON, active)
- Submission(id, formId, userId?, dataJSON, files[], status, assignedTo?, geo?, createdAt, updatedAt)
- Project(id, title, type, ward?, budget, spent, status, startDate, endDate, milestones[], geo, contractors[], docs[])
- Milestone(id, projectId, title, date, progress, notes, photos[])
- Scheme(id, title, category, criteriaJSON, docsRequired[], processSteps[], links[])
- EligibilityRun(id, schemeId, userId?, answersJSON, resultJSON, createdAt)
- Event(id, title, start, end, location, rsvpEnabled, description, attachments[])
- Notice(id, title, category, deadline?, body, attachments[])
- DirectoryEntry(id, type[SHG|Business|Artisan], name, contact, description, products[], geo, approved)
- Complaint(id, category, details, geo, media[], status, slaDue, history[])
- Pledge(id, userId?, pledgeType[tree|solar|waste], amount, geo?, approved)
- CarbonCalcRun(id, userId?, inputsJSON, outputJSON, createdAt)
- Donation(id, donorName?, type[materials|funds|trees], value, publicConsent)
- TranslationKey(id, key, defaultText, module)
- TranslationValue(id, keyId, locale, text)
- Notification(id, channel, templateId, audienceJSON, scheduledAt, status)
- DeliveryStatus(id, notificationId, target, channel, status, error?)
- AuditLog(id, actorId, action, entity, entityId, diffJSON, createdAt)
- Contractor(id, name, contact, docs[], ratings?)
- Payment(id, projectId, amount, date, voucherNo, notes)
- BudgetLine(id, projectId, head, amount, period)
- Attachment(id, url, type, meta)
- FeatureFlag(id, key, enabled, audienceJSON)

**Acceptance**  
- Prisma schema migrates cleanly; seed script creates demo data in §18.  
- Sensitive fields indexed where needed; rate-limited queries.

---

## 9) APIs (tRPC Routers)

Implement routers with input/output Zod schemas, server auth guards, and audit logging.

- `auth`: login OTP, verify, TOTP setup, roles.
- `content`: pages/blocks CRUD, preview, publish, versions.
- `media`: signed upload urls, listing, delete (with usage checks).
- `forms`: builder CRUD; submissions create/list/update; SLA timers.
- `projects`: CRUD; milestones; budgets/payments; public listing.
- `schemes`: CRUD; rules engine; eligibility runs; link to forms.
- `geo`: layers list/CRUD; import/export; photo geotag.
- `moderation`: queues; approve/reject; reasons; notifications.
- `directory`: entries CRUD/approve.
- `events`: CRUD; RSVP; reminders.
- `notices`: CRUD; categories, deadlines.
- `pledges`: CRUD; wall listing (approved only).
- `notifications`: compose/schedule/send; delivery tracking.
- `i18n`: keys/values CRUD; missing key scans.
- `analytics`: SLA metrics, participation, pledges, traffic (proxy).
- `admin`: backups, restore, feature flags, settings.

**Acceptance**  
- All routers have unit tests for validation & RBAC; error handling standardized.

---

## 10) Workflows

### 10.1 Complaint Filing
1. Public form → category, ward, geo, photos/videos; Zod validation; hCaptcha.
2. Create ticket; SLA timer (category-based).
3. Moderation/assignment; citizen notified (email/SMS/Web Push).
4. Status updates; evidence at resolution; citizen feedback; close.

### 10.2 Eligibility Checker
1. Wizard questions → rule engine evaluates.
2. Result with steps & docs; 1-click start application (prefill).
3. Track status in Account.

### 10.3 Project Publishing
1. Draft by DataEntry → Editor review → Approver publish.
2. Version snapshot; map/charts auto-update; audit log entry.

### 10.4 Carbon Pledge
1. Pledge (tree/solar/waste) with optional geo.
2. Admin moderation; appears on pledge wall; counters increment.
3. Follow-up reminders & how-to links.

### 10.5 Events
1. Admin creates; citizens RSVP.
2. Reminders via notifications; attendance export.

### 10.6 Notifications
- Templates per channel; audience targeting by ward/role/interest; schedule.
- Delivery status stored per target/channel.

---

## 11) PWA & Offline

- Web manifest (Hindi/English name), icons, theme colors.
- Workbox:
  - `CacheFirst` static assets.
  - `StaleWhileRevalidate` for pages/data.
  - `NetworkOnly` + Background Sync queue for POST forms.
- Offline fallback page; badge/indicator when offline.
- Install prompt; deferred install UX.

**Acceptance**  
- Lighthouse PWA score ≥ 95; installable; offline home + key pages.

---

## 12) Security & Privacy

- OWASP controls; secure headers (CSP, HSTS, XFO deny, XSS, Referrer-Policy).
- Auth hardening: next-auth session fixation mitigation; email OTP throttles; TOTP 2FA for admin roles.
- RBAC server-enforced; route handlers check permissions.
- CSRF protection; rate limiting per IP/user/form; brute force protection.
- File uploads: signed URLs, MIME/size validation, virus scan hook (optional), private bucket for sensitive files.
- PII minimization; explicit consent for public display (pledge wall, directory).
- Audit logs on all mutating actions.
- Secrets only via env; no secrets in repo.

---

## 13) Accessibility

- WCAG 2.2 AA: keyboard navigation, skip links, focus visibly, ARIA roles/labels, structural landmarks.
- Hindi content with correct `lang` attributes.
- Form error announcements (aria-live), labels, descriptions, hint text.
- Motion respect; high contrast mode toggle.

**Acceptance**  
- axe checks pass (no critical issues) in CI.  
- Screen reader smoke tests for key flows.

---

## 14) Performance

- Core Web Vitals green; image optimization; RSC data loading; ISR/SSG where suitable.
- Route segment caching; CDN headers; TanStack Query for client caches.
- Performance budgets in Lighthouse CI (LCP < 2.5s, CLS < 0.1, TTI < 3.5s).

---

## 15) RBAC Matrix

| Module/Action                  | Viewer | DataEntry | Editor | Approver | Admin |
|--------------------------------|:------:|:---------:|:------:|:--------:|:-----:|
| View public content            |   ✓    |     ✓     |   ✓    |    ✓     |   ✓   |
| Create/Edit drafts             |        |     ✓     |   ✓    |    ✓     |   ✓   |
| Publish content                |        |           |        |    ✓     |   ✓   |
| Manage forms/workflows         |        |     ✓     |   ✓    |    ✓     |   ✓   |
| Moderate submissions/comments  |        |     ✓     |   ✓    |    ✓     |   ✓   |
| Projects/Budgets write         |        |     ✓     |   ✓    |    ✓     |   ✓   |
| Users/Roles/2FA settings       |        |           |        |          |   ✓   |
| Theme & Settings               |        |           |        |          |   ✓   |
| Backups/Integrations           |        |           |        |          |   ✓   |
| View analytics/reports         |   ✓    |     ✓     |   ✓    |    ✓     |   ✓   |

Server-side guards + UI gating required.

---

## 16) SEO & Discoverability

- Bilingual metadata; JSON-LD types: Organization, GovernmentService, Event, NewsArticle, Place.
- Clean URLs; breadcrumbs; sitemap.xml; robots.txt; canonical tags.
- Structured content for schemes/projects to surface in search.

---

## 17) Repo Structure

app/(public)/* app/admin/* app/api/* components/*              # headless + composed components/admin/* lib/{auth,db,i18n,validation,map,charts,uploads,notifications,moderation,rbac,featureFlags}.ts styles/*                  # tailwind theme, globals public/*                  # icons, manifest, images content/*                 # seed markdown/json prisma/{schema.prisma,migrations,seeds.ts} tests/{unit,component,e2e,fixtures} docs/{README.md,ADMIN_GUIDE.md,CONTENT_GUIDE.md,DATA_IMPORT.md,I18N.md,DESIGN-SYSTEM.md,OPS_RUNBOOK.md} scripts/{seed.ts,backup.ts,restore.ts}

---

## 18) Sample Data & Seeding

Seed on `pnpm seed`:
- **Projects (4–6)**: solar street lights, rainwater harvesting, road repair, school upgrade (+ milestones, budgets, photos).
- **Schemes (8–10)** with criteria + FAQs.
- **Events (3)**, **Notices (5)**.
- **Directory (6)** SHGs/businesses (approved/pending mix).
- **Pledges (10)**, **Complaints (6)** with varied statuses.
- Ward/hamlet names; boundary sketch (GeoJSON).
- Gram Pradhan message (Hindi first); office timings/contacts; map pin.

---

## 19) Testing Strategy

**Unit (Jest/RTL)**
- Eligibility rules engine.
- Carbon calculator functions.
- Zod schemas for forms.
- RBAC helpers; notifications templating.

**Component**
- Forms (complaint, eligibility), block editor, tables, map widgets.

**E2E (Playwright)**
- Complaint flow (file → assign → update → close).
- Eligibility wizard → application → track status.
- Project publishing (draft → review → approve → live).
- Login (email OTP) + Admin TOTP 2FA.
- i18n toggle persists across sessions.
- PWA offline: offline home + queued form sync.

**A11y**
- axe checks for key pages.

**Perf**
- Lighthouse CI against preview; budgets enforced.

---

## 20) CI/CD (GitHub Actions)

Jobs:
1. **verify**: `pnpm i`, `pnpm typecheck`, `pnpm lint`, `pnpm test`, `pnpm build`.
2. **e2e**: Start Postgres service; run Playwright across Chromium/WebKit/Firefox.
3. **lighthouse**: Lighthouse CI on Preview.
4. **codeql**: Security analysis.
5. **deploy**: Vercel deploy on `main` (or Docker build/push on tags).
6. **seed-preview**: Seed demo data in preview environments.
7. **dependabot**: Weekly updates.

Artifacts: test reports, coverage, screenshots/videos, LW budgets.

---

## 21) Environments & Configuration

**Env Vars (`.env.example`)**

DATABASE_URL=postgres://... NEXTAUTH_URL=http://localhost:3000 NEXTAUTH_SECRET=... EMAIL_SERVER_HOST=... EMAIL_SERVER_PORT=... EMAIL_SERVER_USER=... EMAIL_SERVER_PASSWORD=... EMAIL_FROM="Ummid Se Hari <noreply@...>" GOOGLE_CLIENT_ID=... GOOGLE_CLIENT_SECRET=... S3_ENDPOINT=... S3_ACCESS_KEY_ID=... S3_SECRET_ACCESS_KEY=... S3_BUCKET_PUBLIC=... S3_BUCKET_PRIVATE=... SMS_PROVIDER=twilio|textlocal|... SMS_API_KEY=... WHATSAPP_TOKEN=...               # optional WEB_PUSH_PUBLIC_KEY=... WEB_PUSH_PRIVATE_KEY=... MAP_TILES_URL=https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png HCAPTCHA_SITE_KEY=... HCAPTCHA_SECRET=... ANALYTICS_URL=... FEATURE_FLAGS=...

**Envs**
- **Local**: Docker Compose (Postgres, MinIO), dev server; seeded data.
- **Preview**: Vercel preview; seeded demo; protected Admin.
- **Production**: Vercel; scheduled jobs (cron) for SLA checks/backups.

---

## 22) Deployment

**Vercel**
- Build command: `pnpm build`  
- Output: Next.js (serverless functions + edge as configured)  
- Env vars configured; storage bucket external.

**Docker (optional)**
- `Dockerfile` for Next.js app (multi-stage).
- `compose.yml` for Postgres, MinIO, mailhog (dev), app.

**Acceptance**
- Cold start acceptable for serverless APIs (optimize where hot paths exist).
- Healthcheck endpoint responds with app/version/migrations status.

---

## 23) Observability & Ops

- Structured logs with request IDs; error boundaries with user-friendly fallbacks.
- Admin dashboards with basic metrics (SLA, submissions, notif delivery).
- Backup/restore scripts; scheduled backups; restore test documented.
- Data retention & deletion tools; export SRRs (subject requests).

---

## 24) Acceptance Criteria (Global)

- **Build**: `pnpm install && pnpm build` passes; ESLint/typecheck clean.
- **PWA**: Installable; offline home + key pages; background sync functional.
- **Admin**: Edits everything (text/media/blocks/menus/colors/SEO/strings).
- **Forms**: Validation (client/server), uploads, spam protection, rate limits, confirmations.
- **Maps**: Layers load quickly; project milestones on map.
- **i18n**: All strings externalized; toggle persists; Hindi default.
- **Security**: RBAC enforced; 2FA for admins; CSRF; rate limiting; audit logs.
- **Tests**: Unit/component/E2E pass in CI; Lighthouse ≥ 95 (PWA/Perf/SEO/A11y).
- **Docs**: README, Admin Guide, Content Guide, Data Import, I18N, Design System, Ops Runbook.
- **Seed**: Sample data present and coherent.

---

## 25) Implementation Plan (Step-by-Step PRs)

> Each PR includes: scope, checklist, screenshots (UI), test results, bilingual acceptance notes.

**PR 0 — Bootstrap & DX**
- Next.js 14 + TS + Tailwind + shadcn/ui.
- next-intl (hi/en), locale middleware; language switcher with cookie.
- Base layout, mega-menu shell, skip links, focus styles.
- ESLint/Prettier, Husky hooks, commitlint.
- **Done when**: App runs; bilingual hero; axe no critical issues; Lighthouse PWA shell ≥ 90.

**PR 1 — DB & Auth**
- Prisma schema + migrations; seed admin.
- next-auth Email OTP + optional Google; TOTP 2FA for admins.
- RBAC helpers; session hardening.
- **Done when**: Login/2FA works; role guards tested; seeds load.

**PR 2 — PWA & SW**
- Manifest/icons; Workbox strategies; offline fallback; BG sync queue.
- Install prompt & offline indicator.
- **Done when**: Offline home + queued form sync tested.

**PR 3 — Design System**
- Tailwind tokens; shadcn themes; component library; motion primitives.
- **Done when**: Component gallery page passes a11y checks.

**PR 4 — Admin Shell**
- `/admin` protected area; sidebar; breadcrumbs; profile; 2FA mgmt.
- Audit log viewer; roles mgmt.
- **Done when**: RBAC in UI + server; audit entries on changes.

**PR 5 — Content Manager**
- Blocks editor; versioning; preview; staging/publish/rollback.
- Media library with alt/caption required; PDF viewer.
- **Done when**: Home built with blocks; publish updates reflect immediately.

**PR 6 — Form Builder & Submissions**
- Visual builder → JSON schema (Zod); conditional logic; uploads.
- Public endpoints; queues; SLA/assignment; hCaptcha; rate limiting.
- My Requests dashboard.
- **Done when**: Complaint form built via builder; full flow works.

**PR 7 — Geo & Maps**
- MapLibre base; ward boundaries; projects/issues/resources layers.
- GeoJSON CRUD; CSV/GeoJSON import; photo geotagging.
- **Done when**: Admin edits layers; public map renders quickly.

**PR 8 — Projects & Budgets**
- Public list/detail; milestones; budgets; contractors; comments (moderated).
- Budget Explorer; CSV export.
- **Done when**: Seeded projects visible; charts correct.

**PR 9 — Smart Mission**
- Carbon calculator; Solar wizard; Pledge wall; Waste mini-game; Water tracker.
- **Done when**: Calculator outputs tips; pledge moderation works.

**PR 10 — Schemes & Eligibility**
- Cards; Wizard backed by rules engine; how-to; checklists; FAQs.
- **Done when**: Test scenarios yield correct eligibility outcomes.

**PR 11 — Services & Requests**
- Full form suite; SLAs; escalations; notifications.
- **Done when**: SLA breach triggers escalation & notification.

**PR 12 — News/Notices/Events**
- CRUD + public pages; calendar; RSVP; reminders; ICS.
- **Done when**: Event reminders deliver; attendance export.

**PR 13 — Directory & Economy**
- Listings; profiles; jobs; trainings; admin approvals.
- **Done when**: Pending → approved flow; enquiries deliver.

**PR 14 — Health/Education/Social**
- Content pages; emergency CTA global component.
- **Done when**: Emergency CTA always present & accessible.

**PR 15 — Tourism & Culture**
- Attractions/treks/homestays; map; gallery; tips.
- **Done when**: Gallery a11y tested; perf acceptable.

**PR 16 — Volunteer & Donate**
- Opportunities; signup; donation intents; transparency ledger.
- **Done when**: UPI intent flow documented; ledger updates.

**PR 17 — Open Data & Reports**
- CSV/JSON datasets; dashboards; profile PDF; monthly report generator.
- **Done when**: Exports accurate; API tokens configurable.

**PR 18 — Notifications Center**
- Compose; segment; schedule; track deliveries; templates.
- **Done when**: Multi-channel delivery statuses recorded.

**PR 19 — Translations Admin**
- Keys/values manager; side-by-side; missing key linter; import/export CSV.
- **Done when**: All strings externalized; admin can translate UI.

**PR 20 — Theming & Settings**
- Logo/colors/fonts/menus/SEO/OG/PWA/cookie text.
- **Done when**: Theme changes apply site-wide without redeploy.

**PR 21 — Analytics & SEO**
- Privacy-friendly analytics; JSON-LD; sitemap; robots; canonical.
- **Done when**: Pages validate with Rich Results; analytics events fire.

**PR 22 — Seeding & Fixtures**
- Populate demo data in §18.
- **Done when**: Demo content renders across modules.

**PR 23 — Hardening & Tests**
- Security headers; upload sanitization; rate limits; CSRF; DoS guard.
- Expand unit/component/E2E; Lighthouse budgets.
- **Done when**: CI green; Lighthouse ≥ 95; penetration smoke checks pass.

**PR 24 — Docs & Release**
- README; Admin/Content/I18N/Data Import/Design/Ops Runbook.
- `.env.example`; Vercel & Docker guides; release notes.
- **Done when**: First production release tagged and deployed.

---

## 26) Copywriting (Samples)

- Slogan: **“उम्मीद से हरी—स्मार्ट, स्वच्छ और सशक्त दमदई”**
- Primary CTAs:
  - “शिकायत दर्ज करें” / “File a Complaint”
  - “पात्रता जाँचें” / “Check Eligibility”
  - “हरित प्रतिज्ञा लें” / “Take a Green Pledge”
  - “परियोजनाएँ देखें” / “View Projects”

---

## 27) Risk & Mitigation

- **Connectivity variability** → Offline-first; background sync; lite mode.
- **Staff onboarding** → In-app tours; Admin Guide; role-scoped UI; guardrails.
- **Spam/abuse** → hCaptcha; rate limits; moderation queues; audit logs.
- **Privacy** → Consent gating for public listings; data minimization; retention tooling.
- **Vendor lock-in** → Adapters for SMS/tiles/storage; infrastructure docs.

---

## 28) Definition of Done (Per Feature)

- Meets acceptance criteria; a11y checks pass; tests (unit/component/E2E) updated and pass.
- Strings externalized and translated (if public); Hindi default OK.
- RBAC enforced; audit logs present.
- Perf budgets unchanged or improved; Lighthouse CI passes.
- Docs updated (developer + admin/user) where relevant.
