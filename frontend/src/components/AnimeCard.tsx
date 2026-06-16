import { Link } from 'react-router-dom'
import type { Recommendation } from '../types'

interface AnimeCardProps {
  recommendation: Recommendation
}

export function AnimeCard({ recommendation }: AnimeCardProps) {
  const { anime } = recommendation

  return (
    <article className="overflow-hidden rounded border border-[#bbbbbb] bg-white">
      <Link to={`/anime/${anime.id}`} className="block focus:outline-none focus-visible:ring-2 focus-visible:ring-[#2e51a2]">
        <img src={anime.imageUrl} alt={anime.title} className="h-56 w-full object-cover" />
      </Link>
      <div className="space-y-2 p-3">
        <div className="flex items-start justify-between gap-2">
          <h3 className="line-clamp-1 text-[13px] font-bold text-[#1f4392]">{anime.title}</h3>
          {recommendation.isExploratory ? (
            <span className="shrink-0 rounded-sm bg-[#f9d457] px-1 py-0.5 text-[10px] font-bold text-[#5a4a00]">
              Recommended
            </span>
          ) : null}
        </div>
        <p className="line-clamp-2 text-[12px] text-[#666]">{anime.synopsis}</p>
        <div className="flex flex-wrap gap-1.5">
          {anime.genres.slice(0, 3).map((genre) => (
            <span key={genre} className="rounded border border-[#2e51a2] px-1.5 py-0.5 text-[10px] font-bold text-[#2e51a2]">
              {genre}
            </span>
          ))}
        </div>
        <div className="flex items-center justify-between text-[11px] text-[#999]">
          <span>Rank #{recommendation.rank}</span>
          <span>Score {(recommendation.score ?? 0).toFixed(3)}</span>
        </div>
      </div>
    </article>
  )
}
