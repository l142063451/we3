# Design System — "उम्मीद से हरी | Ummid Se Hari"

Smart & Carbon-Free Village PWA Design System for Damday Gram Panchayat

## 🎨 Theme Overview

**Theme Name**: "Ummid Se Hari" (Hope-Green)  
**Philosophy**: Hope-driven design, transparency, inclusivity (WCAG 2.2 AA), privacy-first, low data usage, offline-first.

## 🎯 Design Principles

1. **Hope-Driven**: Design that inspires optimism and positive action
2. **Inclusive**: WCAG 2.2 AA compliant, supports Hindi and English
3. **Transparent**: Clear information hierarchy and open government data
4. **Privacy-First**: Minimal data collection, user consent-focused
5. **Offline-First**: Works without internet connectivity
6. **Sustainable**: Low data usage, efficient performance

## 🌈 Color Palette

### Primary Colors
- **Primary Green** (`#16a34a`): Main brand color representing hope and growth
- **Accent Amber** (`#f59e0b`): Call-to-action and highlight color
- **Trust Blue** (`#2563eb`): Trust, reliability, government services
- **Success Green** (`#22c55e`): Success states, positive actions
- **Warning Amber** (`#f59e0b`): Warnings, attention needed
- **Error Red** (`#dc2626`): Error states, critical alerts
- **Info Blue** (`#0ea5e9`): Information, neutral announcements

### Text & Surfaces
- **Text Primary** (`#1f2937`): Main text color
- **Background** (`#f8fafc`): Main background
- **Surface** (`#ffffff`): Cards, modals, overlays

### Color Usage Guidelines

```css
/* Primary Actions */
.btn-primary { background: #16a34a; }

/* Secondary Actions */
.btn-secondary { background: #f59e0b; }

/* Government Services */
.service-card { border-color: #2563eb; }

/* Success States */
.alert-success { background: #22c55e; }

/* Warnings */
.alert-warning { background: #f59e0b; }

/* Errors */
.alert-error { background: #dc2626; }
```

### Accessibility Requirements
- **Contrast Ratio**: Minimum 4.5:1 for normal text, 3:1 for large text
- **Color Blindness**: All information conveyed by color must also be available through other means
- **Focus States**: 2px solid outline with sufficient contrast

## 🔤 Typography

### Font Stack
```css
/* Headings */
font-family: 'Poppins', 'Noto Sans Devanagari', system-ui, sans-serif;

/* Body Text */
font-family: 'Inter', 'Noto Sans Devanagari', system-ui, sans-serif;

/* Mixed Script Content */
font-family: 'Inter', 'Noto Sans Devanagari', system-ui;
```

### Type Scale
```css
/* Display */
.text-display { font-size: 3.75rem; line-height: 1; font-weight: 800; }

/* H1 */
.text-h1 { font-size: 3rem; line-height: 1.1; font-weight: 700; }

/* H2 */
.text-h2 { font-size: 2.25rem; line-height: 1.2; font-weight: 600; }

/* H3 */
.text-h3 { font-size: 1.875rem; line-height: 1.3; font-weight: 600; }

/* H4 */
.text-h4 { font-size: 1.5rem; line-height: 1.4; font-weight: 500; }

/* H5 */
.text-h5 { font-size: 1.25rem; line-height: 1.5; font-weight: 500; }

/* H6 */
.text-h6 { font-size: 1.125rem; line-height: 1.5; font-weight: 500; }

/* Body Large */
.text-lg { font-size: 1.125rem; line-height: 1.75; }

/* Body */
.text-base { font-size: 1rem; line-height: 1.75; }

/* Body Small */
.text-sm { font-size: 0.875rem; line-height: 1.6; }

/* Caption */
.text-xs { font-size: 0.75rem; line-height: 1.5; }
```

### Bilingual Typography Guidelines
- **Line Height**: Increased to 1.75 for mixed Hindi-English content
- **Font Fallbacks**: System fonts as fallbacks for all custom fonts
- **Text Rendering**: `optimizeLegibility` for Devanagari script
- **Character Spacing**: Default tracking for optimal readability

## 📐 Layout System

### Grid System
- **Container**: Max-width 1280px (7xl), responsive padding
- **Breakpoints**: `sm: 640px`, `md: 768px`, `lg: 1024px`, `xl: 1280px`
- **Grid**: 12-column system with responsive behavior
- **Gutters**: 1rem (16px) on mobile, 2rem (32px) on desktop

### Spacing Scale (8px Grid)
```css
/* Spacing tokens */
.space-1 { margin/padding: 0.25rem; } /* 4px */
.space-2 { margin/padding: 0.5rem; }  /* 8px */
.space-3 { margin/padding: 0.75rem; } /* 12px */
.space-4 { margin/padding: 1rem; }    /* 16px */
.space-5 { margin/padding: 1.25rem; } /* 20px */
.space-6 { margin/padding: 1.5rem; }  /* 24px */
.space-8 { margin/padding: 2rem; }    /* 32px */
.space-10 { margin/padding: 2.5rem; } /* 40px */
.space-12 { margin/padding: 3rem; }   /* 48px */
.space-16 { margin/padding: 4rem; }   /* 64px */
.space-20 { margin/padding: 5rem; }   /* 80px */
.space-24 { margin/padding: 6rem; }   /* 96px */
```

### Border Radius
```css
.rounded-sm { border-radius: 0.125rem; } /* 2px */
.rounded { border-radius: 0.25rem; }     /* 4px */
.rounded-md { border-radius: 0.375rem; } /* 6px */
.rounded-lg { border-radius: 0.5rem; }   /* 8px */
.rounded-xl { border-radius: 0.75rem; }  /* 12px */
.rounded-2xl { border-radius: 1rem; }    /* 16px */
```

## 🎭 Components

### Button System

#### Primary Button
```jsx
<button className="btn-primary h-10 px-4 py-2">
  Primary Action
</button>
```

#### Secondary Button
```jsx
<button className="btn-secondary h-10 px-4 py-2">
  Secondary Action
</button>
```

#### Ghost Button
```jsx
<button className="btn-ghost h-10 px-4 py-2">
  Tertiary Action
</button>
```

### Card System

#### Basic Card
```jsx
<div className="card">
  <div className="card-header">
    <h3 className="text-h5">Card Title</h3>
    <p className="text-sm text-muted-foreground">Card description</p>
  </div>
  <div className="card-body">
    <p>Card content goes here</p>
  </div>
  <div className="card-footer">
    <button className="btn-primary">Action</button>
  </div>
</div>
```

#### Service Card
```jsx
<div className="card p-6 transition-shadow hover:shadow-md">
  <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-primary-100">
    {/* Icon */}
  </div>
  <h3 className="mb-2 text-lg font-semibold">Service Name</h3>
  <p className="text-sm text-gray-600">Service description</p>
</div>
```

### Alert System

#### Success Alert
```jsx
<div className="rounded-lg bg-success-50 border border-success-200 p-4">
  <div className="flex items-center">
    <CheckCircleIcon className="h-5 w-5 text-success-500" />
    <h4 className="ml-2 font-medium text-success-800">Success!</h4>
  </div>
  <p className="mt-2 text-sm text-success-700">Operation completed successfully.</p>
</div>
```

#### Warning Alert
```jsx
<div className="rounded-lg bg-warning-50 border border-warning-200 p-4">
  <div className="flex items-center">
    <ExclamationTriangleIcon className="h-5 w-5 text-warning-500" />
    <h4 className="ml-2 font-medium text-warning-800">Warning!</h4>
  </div>
  <p className="mt-2 text-sm text-warning-700">Please review your input.</p>
</div>
```

#### Error Alert
```jsx
<div className="rounded-lg bg-error-50 border border-error-200 p-4">
  <div className="flex items-center">
    <XCircleIcon className="h-5 w-5 text-error-500" />
    <h4 className="ml-2 font-medium text-error-800">Error!</h4>
  </div>
  <p className="mt-2 text-sm text-error-700">Something went wrong. Please try again.</p>
</div>
```

## 🎬 Motion & Animation

### Animation Principles
- **Duration**: ≤ 150ms for micro-interactions, ≤ 300ms for page transitions
- **Easing**: `ease-out` for entrances, `ease-in` for exits, `ease-in-out` for movements
- **Reduced Motion**: Respects `prefers-reduced-motion: reduce`

### Animation Classes
```css
/* Fade In */
.animate-fade-in {
  animation: fade-in 0.3s ease-out;
}

/* Slide Up */
.animate-slide-up {
  animation: slide-up 0.3s ease-out;
}

/* Accordion */
.animate-accordion-down {
  animation: accordion-down 0.2s ease-out;
}

.animate-accordion-up {
  animation: accordion-up 0.2s ease-out;
}
```

### Transition Classes
```css
.transition-colors { transition: color 0.15s ease-out, background-color 0.15s ease-out; }
.transition-shadow { transition: box-shadow 0.15s ease-out; }
.transition-transform { transition: transform 0.15s ease-out; }
```

## ♿ Accessibility Guidelines

### Focus Management
- **Focus Ring**: 2px solid primary color with offset
- **Focus Trap**: Modal dialogs and dropdowns trap focus
- **Skip Links**: Available for navigation and main content

### Screen Reader Support
- **ARIA Labels**: All interactive elements have accessible names
- **ARIA States**: Dynamic states communicated to screen readers
- **Landmarks**: Proper semantic HTML structure

### Color Accessibility
- **Contrast**: All color combinations meet WCAG AA standards
- **Color Independence**: Information not conveyed by color alone
- **Focus Indicators**: High contrast focus indicators

### Keyboard Navigation
- **Tab Order**: Logical tab sequence
- **Shortcuts**: Common keyboard shortcuts supported
- **Custom Controls**: All custom controls keyboard accessible

## 📱 Responsive Behavior

### Breakpoint Strategy
```css
/* Mobile First */
.responsive-text {
  font-size: 1rem;
}

/* Tablet */
@media (min-width: 768px) {
  .responsive-text {
    font-size: 1.125rem;
  }
}

/* Desktop */
@media (min-width: 1024px) {
  .responsive-text {
    font-size: 1.25rem;
  }
}
```

### Component Responsive Rules
- **Cards**: Single column on mobile, 2-3 columns on tablet, 4+ columns on desktop
- **Navigation**: Hamburger menu on mobile, horizontal menu on desktop
- **Typography**: Larger sizes on desktop, readable sizes on mobile
- **Spacing**: Tighter spacing on mobile, generous spacing on desktop

## 🎯 Component Library

### Available Components
1. **Navigation**: Header navigation with language switcher
2. **Hero Section**: Landing page hero with bilingual content
3. **Service Cards**: Quick service access cards
4. **PWA Components**: Install prompt, offline indicator
5. **Forms**: Accessible form controls with validation
6. **Alerts**: Success, warning, error, and info alerts
7. **Loading States**: Spinners, skeleton screens
8. **Modals**: Accessible modal dialogs

### Planned Components (PR3 Implementation)
1. **Announcement Bar**: Site-wide announcements
2. **KPI Cards**: Dashboard statistics
3. **Progress Rings**: Circular progress indicators
4. **Tabs**: Horizontal tab navigation
5. **Accordions**: Collapsible content sections
6. **Timeline**: Event and progress timelines
7. **Breadcrumbs**: Navigation breadcrumbs
8. **Share Buttons**: Social sharing components
9. **Emergency CTA**: Critical action button
10. **Feedback Button**: User feedback widget

## 🛠 Development Guidelines

### CSS Organization
- **Tailwind First**: Use Tailwind utility classes
- **Component Classes**: For reusable patterns
- **Custom Properties**: For theme customization
- **BEM Methodology**: When custom CSS is needed

### Component Development
- **TypeScript**: All components use TypeScript
- **Props Interface**: Clear prop definitions
- **Forwarded Refs**: Support ref forwarding
- **Composition**: Favor composition over inheritance

### Testing Requirements
- **Accessibility**: All components tested with axe
- **Responsive**: Tested at all breakpoints
- **Keyboard**: Full keyboard navigation tested
- **Screen Reader**: Tested with screen reader

## 📊 Performance Considerations

### Bundle Size
- **Tree Shaking**: Only used utilities included
- **Code Splitting**: Components lazy loaded when possible
- **Font Loading**: Optimized font loading strategy

### Runtime Performance
- **CSS-in-JS**: Avoided for better performance
- **Critical CSS**: Above-the-fold styles inlined
- **Animations**: Hardware accelerated when possible

---

*This design system is living documentation that evolves with the product needs while maintaining consistency and accessibility standards.*