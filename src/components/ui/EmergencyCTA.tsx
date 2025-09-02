import React from 'react'
import { cn } from '@/lib/utils'

export interface EmergencyCTAProps {
  phone?: string
  message?: string
  variant?: 'floating' | 'inline'
  position?: 'bottom-left' | 'bottom-right' | 'top-left' | 'top-right'
  size?: 'sm' | 'md' | 'lg'
  className?: string
  onClick?: () => void
}

export function EmergencyCTA({
  phone = '112', // Default emergency number in India
  message = 'आपातकाल | Emergency',
  variant = 'floating',
  position = 'bottom-right',
  size = 'md',
  className,
  onClick
}: EmergencyCTAProps) {
  const handleClick = () => {
    if (onClick) {
      onClick()
    } else if (phone) {
      window.location.href = `tel:${phone}`
    }
  }

  const sizeClasses = {
    sm: 'h-12 w-12 text-sm',
    md: 'h-14 w-14 text-base',
    lg: 'h-16 w-16 text-lg',
  }

  const positionClasses = {
    'bottom-left': 'bottom-6 left-6',
    'bottom-right': 'bottom-6 right-6',
    'top-left': 'top-6 left-6',
    'top-right': 'top-6 right-6',
  }

  if (variant === 'floating') {
    return (
      <button
        onClick={handleClick}
        className={cn(
          'fixed z-50 flex items-center justify-center rounded-full bg-error text-white shadow-lg transition-all duration-200',
          'hover:bg-error/90 hover:scale-105 focus:outline-none focus:ring-4 focus:ring-error/25',
          'animate-pulse hover:animate-none',
          sizeClasses[size],
          positionClasses[position],
          className
        )}
        aria-label={`Emergency call ${phone}`}
        title={message}
      >
        <svg
          className={cn('h-6 w-6', size === 'sm' && 'h-5 w-5', size === 'lg' && 'h-7 w-7')}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"
          />
        </svg>
      </button>
    )
  }

  return (
    <button
      onClick={handleClick}
      className={cn(
        'inline-flex items-center justify-center gap-2 rounded-lg bg-error px-6 py-3 text-white font-medium shadow-md transition-all duration-200',
        'hover:bg-error/90 hover:shadow-lg focus:outline-none focus:ring-4 focus:ring-error/25',
        'disabled:pointer-events-none disabled:opacity-50',
        className
      )}
      aria-label={`Emergency call ${phone}`}
    >
      <svg
        className="h-5 w-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
        aria-hidden="true"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"
        />
      </svg>
      <span>{message}</span>
      <span className="text-sm opacity-75">({phone})</span>
    </button>
  )
}

// Emergency numbers data for India
export const emergencyNumbers = {
  police: '100',
  fire: '101',
  ambulance: '102',
  disaster: '108',
  general: '112', // Unified emergency number
  women: '1091',
  child: '1098',
  senior: '14567',
}

export interface EmergencyContactsProps {
  className?: string
  title?: string
  showAll?: boolean
}

export function EmergencyContacts({
  className,
  title = 'आपातकालीन संपर्क | Emergency Contacts',
  showAll = false
}: EmergencyContactsProps) {
  const contacts = showAll
    ? Object.entries(emergencyNumbers)
    : Object.entries(emergencyNumbers).slice(0, 4)

  const contactLabels: Record<string, { hi: string; en: string }> = {
    police: { hi: 'पुलिस', en: 'Police' },
    fire: { hi: 'अग्निशमन', en: 'Fire' },
    ambulance: { hi: 'एम्बुलेंस', en: 'Ambulance' },
    disaster: { hi: 'आपदा प्रबंधन', en: 'Disaster' },
    general: { hi: 'सामान्य आपातकाल', en: 'General Emergency' },
    women: { hi: 'महिला हेल्पलाइन', en: 'Women Helpline' },
    child: { hi: 'चाइल्ड हेल्पलाइन', en: 'Child Helpline' },
    senior: { hi: 'वरिष्ठ नागरिक', en: 'Senior Citizen' },
  }

  return (
    <div className={cn('card p-6', className)}>
      <h3 className="text-lg font-semibold mb-4">{title}</h3>
      <div className="space-y-3">
        {contacts.map(([key, number]) => (
          <div key={key} className="flex items-center justify-between py-2 border-b border-border last:border-0">
            <div>
              <div className="font-medium text-sm">
                {contactLabels[key]?.hi} | {contactLabels[key]?.en}
              </div>
            </div>
            <a
              href={`tel:${number}`}
              className="inline-flex items-center gap-2 px-3 py-1 text-sm font-medium text-error hover:text-error/80 transition-colors"
              aria-label={`Call ${contactLabels[key]?.en} at ${number}`}
            >
              <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"
                />
              </svg>
              {number}
            </a>
          </div>
        ))}
      </div>
    </div>
  )
}