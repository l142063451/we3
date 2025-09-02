import type { Metadata, Viewport } from 'next';
import { inter, poppins, notoSansDevanagari } from '@/lib/fonts';
import { cn } from '@/lib/utils';
import './globals.css';

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXTAUTH_URL || 'http://localhost:3000'),
  title: {
    template: '%s | उम्मीद से हरी',
    default: 'उम्मीद से हरी | Damday Gram Panchayat',
  },
  description: 'स्मार्ट और कार्बन मुक्त गांव - दमदई ग्राम पंचायत, उत्तराखंड',
  keywords: [
    'Gram Panchayat',
    'Smart Village',
    'Carbon Free',
    'Damday',
    'Uttarakhand',
    'Government',
    'PWA',
  ],
  authors: [{ name: 'Damday Gram Panchayat' }],
  creator: 'Damday Gram Panchayat',
  publisher: 'Damday Gram Panchayat',
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  openGraph: {
    type: 'website',
    locale: 'hi_IN',
    alternateLocale: 'en_IN',
    title: 'उम्मीद से हरी | Smart & Carbon-Free Village',
    description:
      'दमदई-चुआनाला को एक स्मार्ट, हरित और कार्बन-सचेत मॉडल गांव में बदलना',
    siteName: 'उम्मीद से हरी',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'उम्मीद से हरी | Smart & Carbon-Free Village',
    description:
      'दमदई-चुआनाला को एक स्मार्ट, हरित और कार्बन-सचेत मॉडल गांव में बदलना',
  },
  manifest: '/manifest.json',
  icons: {
    icon: '/favicon.svg',
    shortcut: '/favicon.svg',
    apple: '/icon-192x192.png',
  },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 5,
  userScalable: true,
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#16a34a' },
    { media: '(prefers-color-scheme: dark)', color: '#22c55e' },
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="hi"
      dir="ltr"
      className={cn(
        inter.variable,
        poppins.variable,
        notoSansDevanagari.variable,
        'scroll-smooth'
      )}
    >
      <head>
        <meta charSet="utf-8" />
        <meta name="format-detection" content="telephone=no" />
      </head>
      <body
        className={cn(
          'mixed-script min-h-screen bg-background font-sans antialiased'
        )}
      >
        {/* Skip Links for Accessibility */}
        <a href="#main-content" className="skip-link">
          Skip to main content
        </a>
        <a href="#navigation" className="skip-link">
          Skip to navigation
        </a>

        {/* Main App Structure */}
        <div className="flex min-h-screen flex-col">{children}</div>

        {/* Service Worker Registration */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              if ('serviceWorker' in navigator) {
                window.addEventListener('load', function() {
                  navigator.serviceWorker.register('/sw.js')
                    .then(function(registration) {
                      console.log('SW registered: ', registration);
                    })
                    .catch(function(registrationError) {
                      console.log('SW registration failed: ', registrationError);
                    });
                });
              }
            `,
          }}
        />
      </body>
    </html>
  );
}
