// Temporary simple middleware without next-intl
import { NextRequest, NextResponse } from 'next/server';

export default function middleware(request: NextRequest) {
  // Set default locale cookie
  const response = NextResponse.next();

  if (!request.cookies.has('NEXT_LOCALE')) {
    response.cookies.set('NEXT_LOCALE', 'hi', {
      maxAge: 60 * 60 * 24 * 365, // 1 year
      path: '/',
    });
  }

  return response;
}

export const config = {
  matcher: ['/((?!api|_next|_vercel|.*\\..*).*)'],
};
