import { useState } from 'react'
import { postInteraction } from '../api/client'

interface FeedbackControlsProps {
  animeId: number
}

export function FeedbackControls({ animeId }: FeedbackControlsProps) {
  const [rating, setRating] = useState<number | null>(null)
  const [status, setStatus] = useState<'idle' | 'watched' | 'dropped'>('idle')

  async function handleRate(nextRating: number) {
    setRating(nextRating)
    await postInteraction({
      animeId,
      signal: 'rating',
      value: nextRating,
      timestamp: new Date().toISOString(),
    })
  }

  async function handleStatus(nextStatus: 'watched' | 'dropped') {
    setStatus(nextStatus)
    await postInteraction({
      animeId,
      signal: nextStatus,
      value: true,
      timestamp: new Date().toISOString(),
    })
  }

  return (
    <section className="space-y-3 rounded-xl border border-slate-800 bg-slate-900 p-4">
      <h3 className="text-sm font-semibold text-slate-200">Feedback</h3>
      <div className="flex items-center gap-2">
        {[1, 2, 3, 4, 5].map((star) => (
          <button
            key={star}
            type="button"
            aria-label={`Rate ${star} stars`}
            onClick={() => void handleRate(star)}
            className={`h-8 w-8 rounded-md border text-sm ${
              rating !== null && star <= rating
                ? 'border-amber-400 bg-amber-400/15 text-amber-300'
                : 'border-slate-700 text-slate-400 hover:border-indigo-400 hover:text-indigo-300'
            }`}
          >
            ★
          </button>
        ))}
      </div>
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => void handleStatus('watched')}
          className={`rounded-md border px-3 py-1.5 text-sm ${
            status === 'watched'
              ? 'border-emerald-400 bg-emerald-400/15 text-emerald-300'
              : 'border-slate-700 text-slate-300 hover:border-emerald-400'
          }`}
        >
          Watched
        </button>
        <button
          type="button"
          onClick={() => void handleStatus('dropped')}
          className={`rounded-md border px-3 py-1.5 text-sm ${
            status === 'dropped'
              ? 'border-rose-400 bg-rose-400/15 text-rose-300'
              : 'border-slate-700 text-slate-300 hover:border-rose-400'
          }`}
        >
          Dropped
        </button>
      </div>
    </section>
  )
}
