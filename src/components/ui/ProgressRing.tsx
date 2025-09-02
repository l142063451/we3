import React from 'react'
import { cn } from '@/lib/utils'

export interface ProgressRingProps {
  progress: number // 0-100
  size?: 'sm' | 'md' | 'lg' | 'xl'
  strokeWidth?: number
  className?: string
  children?: React.ReactNode
  showPercentage?: boolean
  color?: 'primary' | 'success' | 'warning' | 'error' | 'info'
}

export function ProgressRing({
  progress,
  size = 'md',
  strokeWidth = 4,
  className,
  children,
  showPercentage = true,
  color = 'primary'
}: ProgressRingProps) {
  const sizeMap = {
    sm: { diameter: 40, fontSize: 'text-xs' },
    md: { diameter: 64, fontSize: 'text-sm' },
    lg: { diameter: 96, fontSize: 'text-lg' },
    xl: { diameter: 128, fontSize: 'text-xl' },
  }

  const colorMap = {
    primary: 'stroke-primary-500',
    success: 'stroke-success-500',
    warning: 'stroke-warning-500',
    error: 'stroke-error-500',
    info: 'stroke-info-500',
  }

  const { diameter, fontSize } = sizeMap[size]
  const radius = (diameter - strokeWidth) / 2
  const circumference = radius * 2 * Math.PI
  const offset = circumference - (progress / 100) * circumference

  return (
    <div className={cn('relative', className)} style={{ width: diameter, height: diameter }}>
      <svg
        className="transform -rotate-90 w-full h-full"
        viewBox={`0 0 ${diameter} ${diameter}`}
        aria-hidden="true"
      >
        {/* Background circle */}
        <circle
          cx={diameter / 2}
          cy={diameter / 2}
          r={radius}
          fill="transparent"
          className="stroke-muted-foreground/20"
          strokeWidth={strokeWidth}
        />
        {/* Progress circle */}
        <circle
          cx={diameter / 2}
          cy={diameter / 2}
          r={radius}
          fill="transparent"
          className={cn('transition-all duration-300 ease-out', colorMap[color])}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
        />
      </svg>
      
      {/* Content overlay */}
      <div className="absolute inset-0 flex items-center justify-center">
        {children ? (
          children
        ) : showPercentage ? (
          <span className={cn('font-semibold tabular-nums', fontSize)}>
            {Math.round(progress)}%
          </span>
        ) : null}
      </div>
    </div>
  )
}

// Animated progress ring that counts up
export interface AnimatedProgressRingProps extends ProgressRingProps {
  duration?: number // Animation duration in milliseconds
}

export function AnimatedProgressRing({
  progress,
  duration = 1500,
  ...props
}: AnimatedProgressRingProps) {
  const [currentProgress, setCurrentProgress] = React.useState(0)

  React.useEffect(() => {
    const startTime = Date.now()
    const startProgress = currentProgress

    const animate = () => {
      const elapsed = Date.now() - startTime
      const ratio = Math.min(elapsed / duration, 1)
      
      // Ease-out curve
      const easeOut = 1 - Math.pow(1 - ratio, 3)
      const newProgress = startProgress + (progress - startProgress) * easeOut

      setCurrentProgress(newProgress)

      if (ratio < 1) {
        requestAnimationFrame(animate)
      }
    }

    requestAnimationFrame(animate)
  }, [progress, duration, currentProgress])

  return <ProgressRing {...props} progress={currentProgress} />
}