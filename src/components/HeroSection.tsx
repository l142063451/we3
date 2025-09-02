import Link from 'next/link';
import { cn } from '@/lib/utils';

interface HeroSectionProps {
  className?: string;
}

// Temporary static hero translations
const heroTranslations = {
  title: 'उम्मीद से हरी',
  subtitle: 'स्मार्ट, स्वच्छ और सशक्त दमदई',
  description:
    'दमदई-चुआनाला को एक स्मार्ट, हरित और कार्बन-सचेत मॉडल गांव में बदलना, उम्मीद से प्रेरित होकर और पारदर्शी, सहभागी शासन को सक्षम बनाना।',
  cta: {
    complaint: 'शिकायत दर्ज करें',
    eligibility: 'पात्रता जांचें',
    pledge: 'हरित प्रतिज्ञा लें',
    projects: 'परियोजनाएं देखें',
  },
};

export function HeroSection({ className }: HeroSectionProps) {
  const t = heroTranslations;

  return (
    <section
      className={cn(
        'bg-gradient-to-br from-primary-50 to-accent-50 py-16 lg:py-24',
        className
      )}
    >
      <div className="container">
        <div className="mx-auto max-w-4xl text-center">
          {/* Main Title */}
          <h1 className="mixed-script mb-4 text-4xl font-bold text-primary md:text-5xl lg:text-6xl">
            {t.title}
          </h1>

          {/* Subtitle */}
          <h2 className="mixed-script mb-6 text-xl font-semibold text-gray-700 md:text-2xl lg:text-3xl">
            {t.subtitle}
          </h2>

          {/* Description */}
          <p className="mixed-script mx-auto mb-8 max-w-3xl text-lg leading-relaxed text-gray-600 md:text-xl">
            {t.description}
          </p>

          {/* Call to Action Buttons */}
          <div className="mb-12 flex flex-col items-center justify-center gap-4 sm:flex-row">
            <Link
              href="/services/complaints"
              className="btn btn-primary mixed-script min-w-[200px] px-8 py-3 text-lg font-semibold"
            >
              {t.cta.complaint}
            </Link>
            <Link
              href="/schemes/eligibility"
              className="btn btn-secondary mixed-script min-w-[200px] px-8 py-3 text-lg font-semibold"
            >
              {t.cta.eligibility}
            </Link>
          </div>

          {/* Secondary CTAs */}
          <div className="flex flex-col items-center justify-center gap-4 sm:flex-row">
            <Link
              href="/mission/pledge"
              className="btn btn-ghost mixed-script px-6 py-2 text-base font-medium"
            >
              {t.cta.pledge}
            </Link>
            <Link
              href="/projects"
              className="btn btn-ghost mixed-script px-6 py-2 text-base font-medium"
            >
              {t.cta.projects}
            </Link>
          </div>
        </div>

        {/* Visual Elements */}
        <div className="mt-16 flex items-center justify-center space-x-8">
          <div className="flex items-center space-x-2 text-success-600">
            <div className="h-4 w-4 animate-pulse rounded-full bg-success-500" />
            <span className="text-sm font-medium">स्मार्ट गांव</span>
          </div>
          <div className="flex items-center space-x-2 text-primary-600">
            <div className="h-4 w-4 animate-pulse rounded-full bg-primary-500" />
            <span className="text-sm font-medium">हरित भविष्य</span>
          </div>
          <div className="flex items-center space-x-2 text-accent-600">
            <div className="h-4 w-4 animate-pulse rounded-full bg-accent-500" />
            <span className="text-sm font-medium">पारदर्शिता</span>
          </div>
        </div>
      </div>
    </section>
  );
}
