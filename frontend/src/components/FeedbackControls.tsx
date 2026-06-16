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
    <section className="overflow-hidden rounded border border-[#bbbbbb] bg-white">
      <h3 className="bg-[#2e51a2] px-3 py-1.5 text-[12px] font-bold uppercase tracking-wide text-white">
        Add to List
      </h3>
      <div className="space-y-4 p-3">
        <div>
          <p className="mb-1 text-[11px] font-bold text-[#666]">Your Score</p>
          <div className="flex items-center gap-1">
            {[1, 2, 3, 4, 5].map((star) => (
              <button
                key={star}
                type="button"
                aria-label={`Rate ${star} stars`}
                onClick={() => void handleRate(star)}
                className={`text-lg leading-none transition-colors ${
                  rating !== null && star <= rating
                    ? 'text-[#f9a825]'
                    : 'text-[#cccccc] hover:text-[#f9a825]'
                }`}
              >
                ★
              </button>
            ))}
          </div>
        </div>

        <div>
          <p className="mb-1 text-[11px] font-bold text-[#666]">Status</p>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => void handleStatus('watched')}
              className={`rounded border px-3 py-1 text-[12px] font-bold ${
                status === 'watched'
                  ? 'border-[#2db039] bg-[#2db039] text-white'
                  : 'border-[#bbbbbb] text-[#1f4392] hover:border-[#2db039]'
              }`}
            >
              Completed
            </button>
            <button
              type="button"
              onClick={() => void handleStatus('dropped')}
              className={`rounded border px-3 py-1 text-[12px] font-bold ${
                status === 'dropped'
                  ? 'border-[#a12f31] bg-[#a12f31] text-white'
                  : 'border-[#bbbbbb] text-[#1f4392] hover:border-[#a12f31]'
              }`}
            >
              Dropped
            </button>
          </div>
        </div>
      </div>
    </section>
  )
}
