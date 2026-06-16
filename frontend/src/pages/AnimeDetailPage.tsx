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
    return (
      <div className="rounded border border-[#bbbbbb] bg-white p-8 text-center text-[12px] text-slate-500">
        Loading anime details...
      </div>
    )
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
    <div className="space-y-3">
      <Link to="/" className="inline-flex text-[12px] font-bold text-[#1f4392] hover:underline">
        &laquo; Back to list
      </Link>

      <div className="grid gap-3 md:grid-cols-[225px_1fr]">
        {/* Left info column */}
        <aside className="space-y-3">
          <img
            src={anime.imageUrl}
            alt={anime.title}
            className="w-full rounded border border-[#bbbbbb] object-cover"
          />
          <FeedbackControls animeId={anime.id} />
          <section className="overflow-hidden rounded border border-[#bbbbbb] bg-white">
            <h3 className="bg-[#2e51a2] px-3 py-1.5 text-[12px] font-bold uppercase tracking-wide text-white">
              Information
            </h3>
            <dl className="space-y-1.5 p-3 text-[12px] leading-relaxed">
              <div>
                <dt className="inline font-bold text-[#1c1c1c]">Type: </dt>
                <dd className="inline text-[#1f4392]">{anime.studio ? 'TV' : 'Unknown'}</dd>
              </div>
              <div>
                <dt className="inline font-bold text-[#1c1c1c]">Episodes: </dt>
                <dd className="inline text-[#666]">{anime.episodes ?? 'Unknown'}</dd>
              </div>
              <div>
                <dt className="inline font-bold text-[#1c1c1c]">Aired: </dt>
                <dd className="inline text-[#666]">{anime.year ?? 'Unknown'}</dd>
              </div>
              <div>
                <dt className="inline font-bold text-[#1c1c1c]">Studio: </dt>
                <dd className="inline text-[#1f4392]">{anime.studio ?? 'Unknown'}</dd>
              </div>
              <div>
                <dt className="inline font-bold text-[#1c1c1c]">Popularity: </dt>
                <dd className="inline text-[#666]">#{anime.malPopularity ?? 'N/A'}</dd>
              </div>
              <div>
                <dt className="inline font-bold text-[#1c1c1c]">Genres: </dt>
                <dd className="inline text-[#1f4392]">{anime.genres.join(', ')}</dd>
              </div>
            </dl>
          </section>
        </aside>

        {/* Main content */}
        <main className="space-y-3">
          <div className="overflow-hidden rounded border border-[#bbbbbb] bg-white">
            <h1 className="bg-[#2e51a2] px-4 py-2 text-[18px] font-bold text-white">
              {anime.title}
            </h1>
            <div className="p-4">
              <h2 className="mb-2 border-b border-[#e5e5e5] pb-1 text-[14px] font-bold text-[#2e51a2]">
                Synopsis
              </h2>
              <p className="text-[13px] leading-relaxed text-[#1c1c1c]">{anime.synopsis}</p>

              <div className="mt-4 flex flex-wrap gap-2">
                {anime.genres.map((genre) => (
                  <span
                    key={genre}
                    className="rounded border border-[#2e51a2] px-2 py-0.5 text-[11px] font-bold text-[#2e51a2]"
                  >
                    {genre}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
