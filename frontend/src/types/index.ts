export type FallbackReason = 'onboarding' | 'popularity' | 'content_similarity'

export interface Anime {
  id: number
  title: string
  titleEnglish?: string
  synopsis: string
  imageUrl: string
  genres: string[]
  studio?: string
  year?: number
  episodes?: number
  malPopularity?: number
}

export interface Recommendation {
  anime: Anime
  score?: number
  rank: number
  isExploratory?: boolean
  fallbackReason?: FallbackReason
}

export interface UserOnboarding {
  favoriteGenreIds: string[]
  favoriteAnimeIds: number[]
  completedAt?: string
}

export type FeedbackSignal = 'click' | 'rating' | 'watched' | 'dropped'

export interface InteractionEvent {
  userId: string
  animeId: number
  signal: FeedbackSignal
  value?: number | boolean
  timestamp: string
}
