'use client';

export default function OfflinePage() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50">
      <div className="text-center">
        <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-primary">
          <svg
            className="h-8 w-8 text-white"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M18.364 5.636l-3.536 3.536m0 5.656l3.536 3.536M9.172 9.172L5.636 5.636m3.536 9.192L5.636 18.364M12 2v20M2 12h20"
            />
          </svg>
        </div>
        <h1 className="mb-2 text-2xl font-bold text-gray-900">आप ऑफलाइन हैं</h1>
        <p className="mb-4 text-gray-600">
          इंटरनेट कनेक्शन की जांच करें और फिर से कोशिश करें
        </p>
        <button
          onClick={() => window.location.reload()}
          className="btn btn-primary"
        >
          फिर से कोशिश करें
        </button>
      </div>
    </div>
  );
}
