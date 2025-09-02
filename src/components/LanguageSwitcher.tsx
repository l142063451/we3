'use client';

import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { cn } from '@/lib/utils';

interface LanguageSwitcherProps {
  className?: string;
}

// Temporary static translations
const translations = {
  hi: { switchTo: 'भाषा बदलें:', hindi: 'हिंदी', english: 'English' },
  en: { switchTo: 'Switch language to:', hindi: 'हिंदी', english: 'English' },
};

export function LanguageSwitcher({ className }: LanguageSwitcherProps) {
  const router = useRouter();
  const [locale, setLocale] = useState<'hi' | 'en'>('hi'); // Manage locale state
  const t = translations[locale];

  const handleLanguageChange = (newLocale: 'hi' | 'en') => {
    // Set cookie and refresh
    document.cookie = `NEXT_LOCALE=${newLocale}; path=/; max-age=${
      60 * 60 * 24 * 365
    }`;
    setLocale(newLocale);
    router.refresh();
  };

  return (
    <div className={cn('flex items-center space-x-2', className)}>
      <span className="text-sm text-muted-foreground">{t.switchTo}</span>
      <div className="flex rounded-md border">
        <button
          onClick={() => handleLanguageChange('hi')}
          className={cn(
            'rounded-l-md px-3 py-1 text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2',
            locale === 'hi'
              ? 'bg-primary text-primary-foreground'
              : 'bg-background text-foreground hover:bg-muted'
          )}
          aria-label={t.switchTo + ' ' + t.hindi}
          lang="hi"
        >
          {t.hindi}
        </button>
        <button
          onClick={() => handleLanguageChange('en')}
          className={cn(
            'rounded-r-md px-3 py-1 text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2',
            locale === 'en'
              ? 'bg-primary text-primary-foreground'
              : 'bg-background text-foreground hover:bg-muted'
          )}
          aria-label={t.switchTo + ' ' + t.english}
          lang="en"
        >
          {t.english}
        </button>
      </div>
    </div>
  );
}
