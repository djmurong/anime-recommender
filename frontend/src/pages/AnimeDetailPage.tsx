import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { fetchAnimeById, postInteraction } from '../api/client'
import { FeedbackControls } from '../components/FeedbackControls'
import { EmptyState } from '../components/EmptyState'
import type { Anime } from '../types'

export function AnimeDetailPage() {
  const { id } = useParams()
  const [anime, setAnime] = useState<Anime | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function load() {
      const animeId = Number(id)
      if (Number.isNaN(animeId)) {
        setAnime(null)
        setLoading(false)
        return
      }

      const found = await fetchAnimeById(animeId)
      if (found) {
        void postInteraction({
          animeId: found.id,
          signal: 'click',
          value: true,
          timestamp: new Date().toISOString(),
        })
      }
      setAnime(found)
      setLoading(false)
    }
    void load()
  }, [id])

  if (loading) {
    return <p className="text-sm text-slate-400">Loading anime details...</p>
  }

  if (!anime) {
    return (
      <EmptyState
        title="Anime not found"
        description="This placeholder detail route could not find the requested anime."
      />
    )
  }

  return (
    <section className="space-y-6">
      <Link to="/" className="inline-flex text-sm text-indigo-300 hover:text-indigo-200">
        ← Back to recommendations
      </Link>
      <div className="grid gap-6 md:grid-cols-[280px_1fr]">
        <img
          src={anime.imageUrl}
          alt={anime.title}
          className="h-[400px] w-full rounded-xl object-cover"
        />
        <div className="space-y-4">
          <h1 className="text-3xl font-semibold text-slate-100">{anime.title}</h1>
          <p className="text-sm text-slate-300">{anime.synopsis}</p>
          <div className="flex flex-wrap gap-2 text-xs text-slate-300">
            {anime.genres.map((genre) => (
              <span key={genre} className="rounded-full border border-slate-700 px-2 py-0.5">
                {genre}
              </span>
            ))}
          </div>
          <dl className="grid grid-cols-2 gap-2 text-sm text-slate-300">
            <div>
              <dt className="text-slate-400">Studio</dt>
              <dd>{anime.studio ?? 'Unknown'}</dd>
            </div>
            <div>
              <dt className="text-slate-400">Year</dt>
              <dd>{anime.year ?? 'Unknown'}</dd>
            </div>
            <div>
              <dt className="text-slate-400">Episodes</dt>
              <dd>{anime.episodes ?? 'Unknown'}</dd>
            </div>
            <div>
              <dt className="text-slate-400">Popularity</dt>
              <dd>{anime.malPopularity ?? 'N/A'}</dd>
            </div>
          </dl>
          <FeedbackControls animeId={anime.id} />
        </div>
      </div>
    </section>
  )
}
