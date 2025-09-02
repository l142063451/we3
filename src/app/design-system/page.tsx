'use client'

import React, { useState } from 'react'

// Component Gallery Page to showcase design system
export default function ComponentGallery() {
  const [activeTab, setActiveTab] = useState('buttons')

  const tabs = [
    { id: 'buttons', label: 'Buttons' },
    { id: 'cards', label: 'Cards' },
    { id: 'alerts', label: 'Alerts' },
    { id: 'components', label: 'Components' },
    { id: 'typography', label: 'Typography' },
    { id: 'colors', label: 'Colors' },
    { id: 'spacing', label: 'Spacing' },
  ]

  const colors = {
    primary: { name: 'Primary', value: '#16a34a', description: 'Main brand color' },
    accent: { name: 'Accent', value: '#f59e0b', description: 'Call-to-action color' },
    trust: { name: 'Trust', value: '#2563eb', description: 'Government services' },
    success: { name: 'Success', value: '#22c55e', description: 'Success states' },
    warning: { name: 'Warning', value: '#f59e0b', description: 'Warnings' },
    error: { name: 'Error', value: '#dc2626', description: 'Error states' },
    info: { name: 'Info', value: '#0ea5e9', description: 'Information' },
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b bg-card">
        <div className="container py-8">
          <h1 className="text-4xl font-bold tracking-tight">Design System Gallery</h1>
          <p className="mt-2 text-lg text-muted-foreground">
            &ldquo;उम्मीद से हरी | Ummid Se Hari&rdquo; Component Library
          </p>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="border-b">
        <div className="container">
          <nav className="flex space-x-8" aria-label="Gallery sections">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`border-b-2 py-4 px-1 text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'border-primary text-primary'
                    : 'border-transparent text-muted-foreground hover:text-foreground hover:border-gray-300'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Content */}
      <div className="container py-12">
        {activeTab === 'buttons' && (
          <div className="space-y-12">
            <section>
              <h2 className="text-2xl font-semibold mb-6">Button Variants</h2>
              <div className="grid gap-8">
                <div>
                  <h3 className="text-lg font-medium mb-4">Primary Buttons</h3>
                  <div className="flex flex-wrap gap-4">
                    <button className="btn-primary h-10 px-4 py-2">Default</button>
                    <button className="btn-primary h-10 px-4 py-2" disabled>
                      Disabled
                    </button>
                    <button className="btn-primary h-8 px-3 py-1 text-sm">Small</button>
                    <button className="btn-primary h-12 px-6 py-3 text-lg">Large</button>
                  </div>
                </div>
                
                <div>
                  <h3 className="text-lg font-medium mb-4">Secondary Buttons</h3>
                  <div className="flex flex-wrap gap-4">
                    <button className="btn-secondary h-10 px-4 py-2">Default</button>
                    <button className="btn-secondary h-10 px-4 py-2" disabled>
                      Disabled
                    </button>
                    <button className="btn-secondary h-8 px-3 py-1 text-sm">Small</button>
                    <button className="btn-secondary h-12 px-6 py-3 text-lg">Large</button>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-medium mb-4">Ghost Buttons</h3>
                  <div className="flex flex-wrap gap-4">
                    <button className="btn-ghost h-10 px-4 py-2">Default</button>
                    <button className="btn-ghost h-10 px-4 py-2" disabled>
                      Disabled
                    </button>
                    <button className="btn-ghost h-8 px-3 py-1 text-sm">Small</button>
                    <button className="btn-ghost h-12 px-6 py-3 text-lg">Large</button>
                  </div>
                </div>
              </div>
            </section>
          </div>
        )}

        {activeTab === 'cards' && (
          <div className="space-y-12">
            <section>
              <h2 className="text-2xl font-semibold mb-6">Card Components</h2>
              <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
                <div className="card">
                  <div className="card-header">
                    <h3 className="text-lg font-semibold">Basic Card</h3>
                    <p className="text-sm text-muted-foreground">Simple card with header and body</p>
                  </div>
                  <div className="card-body">
                    <p>This is the card content area. You can put any content here.</p>
                  </div>
                  <div className="card-footer">
                    <button className="btn-primary">Action</button>
                  </div>
                </div>

                <div className="card p-6 transition-shadow hover:shadow-md">
                  <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-primary-100">
                    <svg className="h-6 w-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <h3 className="mb-2 text-lg font-semibold">Service Card</h3>
                  <p className="text-sm text-gray-600">Card with icon for services</p>
                </div>

                <div className="card p-6 border-trust-200 bg-trust-50">
                  <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-trust-100">
                    <svg className="h-6 w-6 text-trust" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <h3 className="mb-2 text-lg font-semibold">Government Service</h3>
                  <p className="text-sm text-trust-700">Official government service card</p>
                </div>
              </div>
            </section>
          </div>
        )}

        {activeTab === 'alerts' && (
          <div className="space-y-8">
            <section>
              <h2 className="text-2xl font-semibold mb-6">Alert Components</h2>
              <div className="space-y-4">
                <div className="rounded-lg bg-success-50 border border-success-200 p-4">
                  <div className="flex items-center">
                    <svg className="h-5 w-5 text-success-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <h4 className="ml-2 font-medium text-success-800">Success!</h4>
                  </div>
                  <p className="mt-2 text-sm text-success-700">Operation completed successfully.</p>
                </div>

                <div className="rounded-lg bg-warning-50 border border-warning-200 p-4">
                  <div className="flex items-center">
                    <svg className="h-5 w-5 text-warning-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.464 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z" />
                    </svg>
                    <h4 className="ml-2 font-medium text-warning-800">Warning!</h4>
                  </div>
                  <p className="mt-2 text-sm text-warning-700">Please review your input.</p>
                </div>

                <div className="rounded-lg bg-error-50 border border-error-200 p-4">
                  <div className="flex items-center">
                    <svg className="h-5 w-5 text-error-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <h4 className="ml-2 font-medium text-error-800">Error!</h4>
                  </div>
                  <p className="mt-2 text-sm text-error-700">Something went wrong. Please try again.</p>
                </div>

                <div className="rounded-lg bg-info-50 border border-info-200 p-4">
                  <div className="flex items-center">
                    <svg className="h-5 w-5 text-info-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <h4 className="ml-2 font-medium text-info-800">Information</h4>
                  </div>
                  <p className="mt-2 text-sm text-info-700">Here&apos;s some helpful information for you.</p>
                </div>
              </div>
            </section>
          </div>
        )}

        {activeTab === 'components' && (
          <div className="space-y-12">
            <section>
              <h2 className="text-2xl font-semibold mb-6">Interactive Components</h2>
              
              <div className="space-y-8">
                {/* Announcement Bar */}
                <div>
                  <h3 className="text-lg font-medium mb-4">Announcement Bar</h3>
                  <div className="space-y-2">
                    <div className="rounded-lg bg-info-50 border border-info-200 text-info-800 border-b px-4 py-3">
                      <div className="flex items-center justify-center max-w-7xl mx-auto">
                        <div className="flex items-center space-x-2">
                          <div className="flex-shrink-0">
                            <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                          </div>
                          <div className="text-sm font-medium text-center">
                            गांव सभा बैठक कल 10 बजे | Village meeting tomorrow at 10 AM
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Progress Ring */}
                <div>
                  <h3 className="text-lg font-medium mb-4">Progress Ring</h3>
                  <div className="flex flex-wrap gap-8 items-center">
                    <div className="text-center">
                      <div className="relative mb-2" style={{ width: 64, height: 64 }}>
                        <svg
                          className="transform -rotate-90 w-full h-full"
                          viewBox="0 0 64 64"
                        >
                          <circle
                            cx={32}
                            cy={32}
                            r={28}
                            fill="transparent"
                            className="stroke-muted-foreground/20"
                            strokeWidth={4}
                          />
                          <circle
                            cx={32}
                            cy={32}
                            r={28}
                            fill="transparent"
                            className="stroke-primary-500"
                            strokeWidth={4}
                            strokeDasharray={176}
                            strokeDashoffset={44}
                            strokeLinecap="round"
                          />
                        </svg>
                        <div className="absolute inset-0 flex items-center justify-center">
                          <span className="text-sm font-semibold">75%</span>
                        </div>
                      </div>
                      <p className="text-xs text-muted-foreground">Budget Used</p>
                    </div>

                    <div className="text-center">
                      <div className="relative mb-2" style={{ width: 96, height: 96 }}>
                        <svg
                          className="transform -rotate-90 w-full h-full"
                          viewBox="0 0 96 96"
                        >
                          <circle
                            cx={48}
                            cy={48}
                            r={44}
                            fill="transparent"
                            className="stroke-muted-foreground/20"
                            strokeWidth={4}
                          />
                          <circle
                            cx={48}
                            cy={48}
                            r={44}
                            fill="transparent"
                            className="stroke-success-500"
                            strokeWidth={4}
                            strokeDasharray={276}
                            strokeDashoffset={55}
                            strokeLinecap="round"
                          />
                        </svg>
                        <div className="absolute inset-0 flex items-center justify-center">
                          <span className="text-lg font-semibold">80%</span>
                        </div>
                      </div>
                      <p className="text-sm text-muted-foreground">Project Progress</p>
                    </div>
                  </div>
                </div>

                {/* Tab Component */}
                <div>
                  <h3 className="text-lg font-medium mb-4">Tabs</h3>
                  <div className="card p-6">
                    <div className="inline-flex h-10 items-center justify-center rounded-md bg-muted p-1 text-muted-foreground mb-4">
                      <button className="inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium bg-background text-foreground shadow-sm">
                        Services
                      </button>
                      <button className="inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium hover:bg-background/50 hover:text-foreground">
                        Projects
                      </button>
                      <button className="inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium hover:bg-background/50 hover:text-foreground">
                        Reports
                      </button>
                    </div>
                    <div>
                      <p>Service content would appear here based on the selected tab.</p>
                    </div>
                  </div>
                </div>

                {/* Accordion */}
                <div>
                  <h3 className="text-lg font-medium mb-4">Accordion</h3>
                  <div className="space-y-2">
                    <div className="border rounded-lg">
                      <button className="flex w-full items-center justify-between py-4 px-6 text-left font-medium">
                        शिकायत कैसे दर्ज करें? | How to file a complaint?
                        <svg className="h-4 w-4 shrink-0 transition-transform duration-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="m6 9 6 6 6-6" />
                        </svg>
                      </button>
                    </div>
                    <div className="border rounded-lg">
                      <button className="flex w-full items-center justify-between py-4 px-6 text-left font-medium">
                        योजना की स्थिति कैसे जांचें? | How to check scheme status?
                        <svg className="h-4 w-4 shrink-0 transition-transform duration-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="m6 9 6 6 6-6" />
                        </svg>
                      </button>
                    </div>
                  </div>
                </div>

                {/* Emergency CTA */}
                <div>
                  <h3 className="text-lg font-medium mb-4">Emergency CTA</h3>
                  <div className="flex flex-wrap gap-4 items-center">
                    <button className="inline-flex items-center justify-center gap-2 rounded-lg bg-error px-6 py-3 text-white font-medium shadow-md">
                      <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                      </svg>
                      <span>आपातकाल | Emergency</span>
                      <span className="text-sm opacity-75">(112)</span>
                    </button>
                    
                    <div className="h-14 w-14 flex items-center justify-center rounded-full bg-error text-white shadow-lg animate-pulse">
                      <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                      </svg>
                    </div>
                  </div>
                </div>
              </div>
            </section>
          </div>
        )}

        {activeTab === 'typography' && (
          <div className="space-y-8">
            <section>
              <h2 className="text-2xl font-semibold mb-6">Typography Scale</h2>
              <div className="space-y-6">
                <div>
                  <h1 className="text-4xl font-bold mb-2">Display Heading</h1>
                  <p className="text-sm text-muted-foreground">text-4xl font-bold</p>
                </div>
                <div>
                  <h1 className="text-3xl font-bold mb-2">H1 Heading</h1>
                  <p className="text-sm text-muted-foreground">text-3xl font-bold</p>
                </div>
                <div>
                  <h2 className="text-2xl font-semibold mb-2">H2 Heading</h2>
                  <p className="text-sm text-muted-foreground">text-2xl font-semibold</p>
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-2">H3 Heading</h3>
                  <p className="text-sm text-muted-foreground">text-xl font-semibold</p>
                </div>
                <div>
                  <h4 className="text-lg font-medium mb-2">H4 Heading</h4>
                  <p className="text-sm text-muted-foreground">text-lg font-medium</p>
                </div>
                <div>
                  <h5 className="text-base font-medium mb-2">H5 Heading</h5>
                  <p className="text-sm text-muted-foreground">text-base font-medium</p>
                </div>
                <div>
                  <h6 className="text-sm font-medium mb-2">H6 Heading</h6>
                  <p className="text-sm text-muted-foreground">text-sm font-medium</p>
                </div>
                <div>
                  <p className="text-base mb-2">Body text (default)</p>
                  <p className="text-sm text-muted-foreground">text-base</p>
                </div>
                <div>
                  <p className="text-sm mb-2">Small text</p>
                  <p className="text-sm text-muted-foreground">text-sm</p>
                </div>
                <div>
                  <p className="text-xs mb-2">Extra small text</p>
                  <p className="text-sm text-muted-foreground">text-xs</p>
                </div>
              </div>
            </section>

            <section>
              <h2 className="text-2xl font-semibold mb-6">Bilingual Text</h2>
              <div className="space-y-4">
                <div className="card p-6">
                  <h3 className="text-xl font-semibold mb-4">शिकायत दर्ज करें | File a Complaint</h3>
                  <p className="mb-4">
                    यह एक द्विभाषी पाठ का उदाहरण है जहाम हिंदी और अंग्रेजी दोनों भाषाओं का उपयोग किया गया है।
                  </p>
                  <p>
                    This is an example of bilingual text where both Hindi and English languages are used together.
                  </p>
                </div>
              </div>
            </section>
          </div>
        )}

        {activeTab === 'colors' && (
          <div className="space-y-8">
            <section>
              <h2 className="text-2xl font-semibold mb-6">Color Palette</h2>
              <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                {Object.entries(colors).map(([key, color]) => (
                  <div key={key} className="card p-6">
                    <div 
                      className="w-full h-20 rounded-lg mb-4"
                      style={{ backgroundColor: color.value }}
                    />
                    <h3 className="text-lg font-semibold">{color.name}</h3>
                    <p className="text-sm text-muted-foreground mb-2">{color.description}</p>
                    <code className="text-xs bg-muted px-2 py-1 rounded">{color.value}</code>
                  </div>
                ))}
              </div>
            </section>

            <section>
              <h2 className="text-2xl font-semibold mb-6">Color Variations</h2>
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-medium mb-4">Primary Green Scale</h3>
                  <div className="flex flex-wrap gap-2">
                    {[50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950].map((shade) => (
                      <div key={shade} className="flex flex-col items-center">
                        <div 
                          className={`w-16 h-16 rounded-lg border bg-primary-${shade}`}
                        />
                        <span className="text-xs mt-2">{shade}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </section>
          </div>
        )}

        {activeTab === 'spacing' && (
          <div className="space-y-8">
            <section>
              <h2 className="text-2xl font-semibold mb-6">Spacing Scale</h2>
              <div className="space-y-4">
                <div className="card p-6">
                  <h3 className="text-lg font-medium mb-4">8px Grid System</h3>
                  <div className="space-y-3">
                    {[1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24].map((space) => (
                      <div key={space} className="flex items-center space-x-4">
                        <div className="w-16 text-sm">{space * 4}px</div>
                        <div 
                          className="bg-primary h-4"
                          style={{ width: `${space * 4}px` }}
                        />
                        <code className="text-xs">space-{space}</code>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </section>

            <section>
              <h2 className="text-2xl font-semibold mb-6">Border Radius</h2>
              <div className="grid gap-4 md:grid-cols-3">
                {[
                  { name: 'Small', class: 'rounded-sm', value: '2px' },
                  { name: 'Default', class: 'rounded', value: '4px' },
                  { name: 'Medium', class: 'rounded-md', value: '6px' },
                  { name: 'Large', class: 'rounded-lg', value: '8px' },
                  { name: 'XL', class: 'rounded-xl', value: '12px' },
                  { name: '2XL', class: 'rounded-2xl', value: '16px' },
                ].map((radius) => (
                  <div key={radius.name} className="card p-4 text-center">
                    <div className={`w-16 h-16 bg-primary mx-auto mb-3 ${radius.class}`} />
                    <h4 className="font-medium">{radius.name}</h4>
                    <p className="text-sm text-muted-foreground">{radius.value}</p>
                    <code className="text-xs">{radius.class}</code>
                  </div>
                ))}
              </div>
            </section>
          </div>
        )}
      </div>
    </div>
  )
}