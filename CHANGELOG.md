# Changelog - उम्मीद से हरी

All notable changes to this project will be documented in this file.

## [0.1.0] - 2024-09-02

### Added - PR0 Bootstrap & DX

- **Next.js 14 Application**: Bootstrapped with App Router, TypeScript, and modern tooling
- **Bilingual Support**: Implemented next-intl with Hindi (default) and English support
- **Design System Foundation**:
  - Tailwind CSS configuration with custom design tokens
  - Brand colors (primary #16a34a, accent #f59e0b)
  - Typography setup with Inter, Poppins, and Noto Sans Devanagari fonts
- **PWA Capabilities**:
  - Web manifest with Hindi/English names
  - Basic service worker for offline functionality
  - Install prompt ready icons and configuration
- **Accessibility Features**:
  - Skip links for keyboard navigation
  - Focus management and WCAG 2.2 AA foundations
  - Semantic HTML structure
  - Screen reader support
- **Development Experience**:
  - ESLint + Prettier configuration
  - Husky git hooks for code quality
  - Commitlint for conventional commits
  - TypeScript strict configuration
- **Core Components**:
  - Responsive navigation with mobile menu
  - Language switcher with cookie persistence
  - Hero section with call-to-action buttons
  - Basic layout structure

### Technical Details

- **Framework**: Next.js 14.2.32 with App Router
- **Styling**: Tailwind CSS 3.4.17 with custom design tokens
- **Internationalization**: next-intl 3.26.5
- **Fonts**: Google Fonts (Inter, Poppins, Noto Sans Devanagari)
- **Code Quality**: ESLint 8.57.1, Prettier 3.6.2, Husky 9.1.7
- **PWA**: Custom service worker with cache strategies

### Infrastructure

- Package manager: pnpm
- Node.js environment variables setup
- Git workflow with conventional commits
- Production-ready build configuration
