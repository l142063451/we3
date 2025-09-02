import { Navigation } from '@/components/Navigation';
import { HeroSection } from '@/components/HeroSection';

export default function HomePage() {
  return (
    <>
      <div id="navigation">
        <Navigation />
      </div>
      <main id="main-content" className="flex-1">
        <HeroSection />

        {/* Quick Stats Section */}
        <section className="bg-white py-12">
          <div className="container">
            <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
              <div className="text-center">
                <div className="mb-2 text-3xl font-bold text-primary">
                  1,247
                </div>
                <div className="text-gray-600">परिवार</div>
              </div>
              <div className="text-center">
                <div className="mb-2 text-3xl font-bold text-primary">12</div>
                <div className="text-gray-600">चालू परियोजनाएं</div>
              </div>
              <div className="text-center">
                <div className="mb-2 text-3xl font-bold text-primary">95%</div>
                <div className="text-gray-600">संतुष्टि दर</div>
              </div>
            </div>
          </div>
        </section>

        {/* Quick Links Section */}
        <section className="bg-gray-50 py-12">
          <div className="container">
            <h2 className="mb-12 text-center text-3xl font-bold">
              त्वरित सेवाएं
            </h2>
            <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
              <div className="card p-6 transition-shadow hover:shadow-md">
                <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-primary-100">
                  <svg
                    className="h-6 w-6 text-primary"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                </div>
                <h3 className="mb-2 text-lg font-semibold">शिकायत दर्ज करें</h3>
                <p className="text-sm text-gray-600">
                  अपनी समस्याओं को दर्ज करें और ट्रैक करें
                </p>
              </div>

              <div className="card p-6 transition-shadow hover:shadow-md">
                <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-accent-100">
                  <svg
                    className="h-6 w-6 text-accent"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                </div>
                <h3 className="mb-2 text-lg font-semibold">योजना पात्रता</h3>
                <p className="text-sm text-gray-600">
                  सरकारी योजनाओं के लिए अपनी पात्रता जांचें
                </p>
              </div>

              <div className="card p-6 transition-shadow hover:shadow-md">
                <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-success-100">
                  <svg
                    className="h-6 w-6 text-success"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z"
                    />
                  </svg>
                </div>
                <h3 className="mb-2 text-lg font-semibold">हरित प्रतिज्ञा</h3>
                <p className="text-sm text-gray-600">
                  पर्यावरण संरक्षण के लिए अपना योगदान दें
                </p>
              </div>

              <div className="card p-6 transition-shadow hover:shadow-md">
                <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-trust-100">
                  <svg
                    className="h-6 w-6 text-trust"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4"
                    />
                  </svg>
                </div>
                <h3 className="mb-2 text-lg font-semibold">परियोजनाएं</h3>
                <p className="text-sm text-gray-600">
                  विकास कार्यों की प्रगति देखें
                </p>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 py-8 text-white">
        <div className="container">
          <div className="flex flex-col items-center justify-between md:flex-row">
            <div className="mb-4 md:mb-0">
              <p className="text-sm">
                © 2024 दमदई ग्राम पंचायत। सभी अधिकार सुरक्षित।
              </p>
            </div>
            <div className="flex space-x-4 text-sm">
              <a
                href="/privacy"
                className="transition-colors hover:text-primary-300"
              >
                गोपनीयता नीति
              </a>
              <a
                href="/terms"
                className="transition-colors hover:text-primary-300"
              >
                उपयोग की शर्तें
              </a>
              <a
                href="/contact"
                className="transition-colors hover:text-primary-300"
              >
                संपर्क
              </a>
            </div>
          </div>
        </div>
      </footer>
    </>
  );
}
