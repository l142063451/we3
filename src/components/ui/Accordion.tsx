import React from 'react'
import { cn } from '@/lib/utils'

export interface AccordionProps {
  children: React.ReactNode
  type?: 'single' | 'multiple'
  defaultValue?: string | string[]
  value?: string | string[]
  onValueChange?: (value: string | string[]) => void
  className?: string
  collapsible?: boolean
}

export interface AccordionItemProps {
  children: React.ReactNode
  value: string
  className?: string
  disabled?: boolean
}

export interface AccordionTriggerProps {
  children: React.ReactNode
  className?: string
}

export interface AccordionContentProps {
  children: React.ReactNode
  className?: string
}

const AccordionContext = React.createContext<{
  expandedItems: Set<string>
  toggleItem: (value: string) => void
  type: 'single' | 'multiple'
  collapsible: boolean
} | null>(null)

const AccordionItemContext = React.createContext<{
  value: string
  isExpanded: boolean
  disabled: boolean
} | null>(null)

export function Accordion({
  children,
  type = 'single',
  defaultValue,
  value: controlledValue,
  onValueChange,
  collapsible = true,
  className
}: AccordionProps) {
  const [internalValue, setInternalValue] = React.useState<Set<string>>(() => {
    if (controlledValue) {
      return new Set(Array.isArray(controlledValue) ? controlledValue : [controlledValue])
    }
    if (defaultValue) {
      return new Set(Array.isArray(defaultValue) ? defaultValue : [defaultValue])
    }
    return new Set()
  })

  const expandedItems = React.useMemo(() => {
    return controlledValue
      ? new Set(Array.isArray(controlledValue) ? controlledValue : [controlledValue])
      : internalValue
  }, [controlledValue, internalValue])

  const toggleItem = React.useCallback(
    (value: string) => {
      const newExpanded = new Set(expandedItems)
      
      if (expandedItems.has(value)) {
        if (collapsible || type === 'multiple') {
          newExpanded.delete(value)
        }
      } else {
        if (type === 'single') {
          newExpanded.clear()
        }
        newExpanded.add(value)
      }

      const newValue = type === 'single' 
        ? Array.from(newExpanded)[0] || ''
        : Array.from(newExpanded)

      if (!controlledValue) {
        setInternalValue(newExpanded)
      }
      
      onValueChange?.(newValue)
    },
    [expandedItems, type, collapsible, onValueChange, controlledValue]
  )

  return (
    <AccordionContext.Provider value={{ expandedItems, toggleItem, type, collapsible }}>
      <div className={cn('space-y-2', className)}>
        {children}
      </div>
    </AccordionContext.Provider>
  )
}

export function AccordionItem({ children, value, className, disabled = false }: AccordionItemProps) {
  const context = React.useContext(AccordionContext)
  if (!context) {
    throw new Error('AccordionItem must be used within Accordion')
  }

  const { expandedItems } = context
  const isExpanded = expandedItems.has(value)

  return (
    <AccordionItemContext.Provider value={{ value, isExpanded, disabled }}>
      <div className={cn('border rounded-lg', className)}>
        {children}
      </div>
    </AccordionItemContext.Provider>
  )
}

export function AccordionTrigger({ children, className }: AccordionTriggerProps) {
  const accordionContext = React.useContext(AccordionContext)
  const itemContext = React.useContext(AccordionItemContext)
  
  if (!accordionContext || !itemContext) {
    throw new Error('AccordionTrigger must be used within AccordionItem')
  }

  const { toggleItem } = accordionContext
  const { value, isExpanded, disabled } = itemContext

  return (
    <button
      className={cn(
        'flex w-full items-center justify-between py-4 px-6 text-left font-medium transition-all',
        'hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
        'disabled:pointer-events-none disabled:opacity-50',
        '[&[data-state=open]>svg]:rotate-180',
        className
      )}
      data-state={isExpanded ? 'open' : 'closed'}
      aria-expanded={isExpanded}
      aria-controls={`content-${value}`}
      disabled={disabled}
      onClick={() => !disabled && toggleItem(value)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          !disabled && toggleItem(value)
        }
      }}
    >
      {children}
      <svg
        className="h-4 w-4 shrink-0 transition-transform duration-200"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
        aria-hidden="true"
      >
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="m6 9 6 6 6-6" />
      </svg>
    </button>
  )
}

export function AccordionContent({ children, className }: AccordionContentProps) {
  const itemContext = React.useContext(AccordionItemContext)
  
  if (!itemContext) {
    throw new Error('AccordionContent must be used within AccordionItem')
  }

  const { value, isExpanded } = itemContext

  return (
    <div
      id={`content-${value}`}
      className={cn(
        'overflow-hidden transition-all duration-200 ease-out',
        isExpanded ? 'animate-accordion-down' : 'animate-accordion-up'
      )}
      style={{
        display: isExpanded ? 'block' : 'none'
      }}
    >
      <div className={cn('pb-4 px-6', className)}>
        {children}
      </div>
    </div>
  )
}