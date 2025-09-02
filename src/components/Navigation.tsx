'use client';

import { useState } from 'react';
import Link from 'next/link';
import { cn } from '@/lib/utils';
import { LanguageSwitcher } from './LanguageSwitcher';

interface NavigationProps {
  className?: string;
}

// Temporary static navigation translations
const navigationTranslations = {
  home: 'होम',
  about: 'गांव के बारे में',
  governance: 'ग्राम पंचायत और शासन',
  mission: 'स्मार्ट और कार्बन मुक्त मिशन',
  schemes: 'योजनाएं और लाभ',
  services: 'सेवाएं और अनुरोध',
  projects: 'परियोजनाएं और बजट',
  news: 'समाचार, नोटिस और इवेंट',
  directory: 'निर्देशिका और अर्थव्यवस्था',
  health: 'स्वास्थ्य, शिक्षा और सामाजिक',
  tourism: 'पर्यटन और संस्कृति',
  volunteer: 'स्वयंसेवा और दान',
  data: 'ओपन डेटा और रिपोर्ट',
  contact: 'संपर्क और सहायता',
};

export function Navigation({ className }: NavigationProps) {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const t = navigationTranslations;

  const navigationItems = [
    { href: '/', label: t.home },
    { href: '/about', label: t.about },
    { href: '/governance', label: t.governance },
    { href: '/mission', label: t.mission },
    { href: '/schemes', label: t.schemes },
    { href: '/services', label: t.services },
    { href: '/projects', label: t.projects },
    { href: '/news', label: t.news },
    { href: '/directory', label: t.directory },
    { href: '/health', label: t.health },
    { href: '/tourism', label: t.tourism },
    { href: '/volunteer', label: t.volunteer },
    { href: '/data', label: t.data },
    { href: '/contact', label: t.contact },
  ];

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  return (
    <nav className={cn('bg-white shadow-lg', className)} role="navigation">
      <div className="container">
        <div className="flex h-16 items-center justify-between">
          {/* Logo */}
          <div className="flex-shrink-0">
            <Link
              href="/"
              className="flex items-center space-x-2 rounded-md px-2 py-1 text-xl font-bold text-primary focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2"
            >
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary">
                <div className="h-3 w-3 rounded-full bg-accent" />
              </div>
              <span className="hidden sm:block">उम्मीद से हरी</span>
            </Link>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden lg:block">
            <div className="flex items-baseline space-x-4">
              {navigationItems.slice(0, 6).map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="rounded-md px-3 py-2 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-50 hover:text-primary focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2"
                >
                  {item.label}
                </Link>
              ))}
              {/* More dropdown can be added here */}
            </div>
          </div>

          {/* Language Switcher and Mobile Menu Button */}
          <div className="flex items-center space-x-4">
            <LanguageSwitcher className="hidden sm:flex" />

            {/* Mobile menu button */}
            <button
              onClick={toggleMobileMenu}
              className="rounded-md p-2 text-gray-700 hover:bg-gray-50 hover:text-primary focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 lg:hidden"
              aria-expanded={isMobileMenuOpen}
              aria-label={isMobileMenuOpen ? 'Close menu' : 'Open menu'}
            >
              <svg
                className="h-6 w-6"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                {isMobileMenuOpen ? (
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                ) : (
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 6h16M4 12h16M4 18h16"
                  />
                )}
              </svg>
            </button>
          </div>
        </div>

        {/* Mobile Navigation Menu */}
        {isMobileMenuOpen && (
          <div className="lg:hidden">
            <div className="mt-2 space-y-1 rounded-md bg-gray-50 px-2 pb-3 pt-2">
              {navigationItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="block rounded-md px-3 py-2 text-base font-medium text-gray-700 transition-colors hover:bg-gray-100 hover:text-primary focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2"
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  {item.label}
                </Link>
              ))}
              <div className="border-t border-gray-200 pt-4">
                <LanguageSwitcher />
              </div>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
}
