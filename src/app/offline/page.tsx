'use client'

export default function OfflinePage() {
  const handleReload = () => {
    if (typeof window !== 'undefined') {
      window.location.reload()
    }
  }

  const handleGoBack = () => {
    if (typeof window !== 'undefined') {
      window.history.back()
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 px-4">
      <div className="max-w-md w-full text-center">
        <div className="mb-8">
          <div className="mx-auto h-24 w-24 flex items-center justify-center rounded-full bg-primary-100">
            <svg className="h-12 w-12 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 5.636l-12.728 12.728m0-12.728l12.728 12.728" />
            </svg>
          </div>
        </div>
        
        <h1 className="text-2xl font-bold text-gray-900 mb-2">
          आप ऑफलाइन हैं
        </h1>
        <h2 className="text-lg font-medium text-gray-600 mb-4">
          You&apos;re Offline
        </h2>
        
        <p className="text-gray-600 mb-8">
          इंटरनेत कनेक्शन की जांच करें। ऑफलाइन रहते हुए कुछ सुविधाएं उपलब्ध नहीं हो सकती हैं, लेकिन आप कैश की गई सामग्री देख सकते हैं।
        </p>
        
        <div className="space-y-4">
          <button
            onClick={handleReload}
            className="w-full bg-primary-600 text-white py-3 px-4 rounded-lg font-medium hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 transition-colors"
          >
            फिर से कोशिश करें | Try Again
          </button>
          
          <button
            onClick={handleGoBack}
            className="w-full bg-gray-100 text-gray-700 py-3 px-4 rounded-lg font-medium hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 transition-colors"
          >
            वापस जाएं | Go Back
          </button>
        </div>
        
        <div className="mt-8 p-4 bg-blue-50 border border-blue-200 rounded-lg text-left">
          <div className="flex items-start">
            <svg className="h-5 w-5 text-blue-400 mt-0.5 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div className="text-sm">
              <p className="font-medium text-blue-800 mb-1">
                ऑफलाइन उपलब्ध सुविधाएं:
              </p>
              <ul className="text-blue-700 space-y-1">
                <li>• कैश किए गए पृष्ठ देखें</li>
                <li>• डाउनलोड की गई सामग्री देखें</li>
                <li>• फॉर्म भरें (ऑनलाइन होने पर सिंक होगा)</li>
              </ul>
            </div>
          </div>
        </div>
        
        <div className="mt-6">
          <a
            href="/"
            className="text-primary-600 hover:text-primary-700 font-medium text-sm"
          >
            मुख्य पृष्ठ पर वापस जाएं | Return to Homepage
          </a>
        </div>
      </div>
    </div>
  )
}
