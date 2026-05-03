import { Link } from 'react-router-dom'
import type { Recommendation } from '../types'

interface AnimeCardProps {
  recommendation: Recommendation
}

export function AnimeCard({ recommendation }: AnimeCardProps) {
  const { anime } = recommendation

  return (
    <article className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900/80 shadow-md shadow-slate-950/30">
      <Link to={`/anime/${anime.id}`} className="block focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400">
        <img src={anime.imageUrl} alt={anime.title} className="h-64 w-full object-cover" />
      </Link>
      <div className="space-y-3 p-4">
        <div className="flex items-start justify-between gap-2">
          <h3 className="line-clamp-1 text-base font-semibold">{anime.title}</h3>
          {recommendation.isExploratory ? (
            <span className="shrink-0 rounded-full bg-amber-500/20 px-2 py-0.5 text-xs text-amber-300">
              Discovery
            </span>
          ) : null}
        </div>
        <p className="line-clamp-2 text-sm text-slate-300">{anime.synopsis}</p>
        <div className="flex flex-wrap gap-2">
          {anime.genres.slice(0, 3).map((genre) => (
            <span key={genre} className="rounded-full border border-slate-700 px-2 py-0.5 text-xs text-slate-300">
              {genre}
            </span>
          ))}
        </div>
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span>Rank #{recommendation.rank}</span>
          <span>Score {(recommendation.score ?? 0).toFixed(3)}</span>
        </div>
      </div>
    </article>
  )
}
