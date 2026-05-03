import type { Anime, Recommendation, UserOnboarding } from '../../types'

const PLACEHOLDER_SYNOPSIS =
  'A placeholder synopsis for the frontend prototype. Real metadata and embeddings will be wired from the backend later.'

export const genreOptions = [
  'Action',
  'Adventure',
  'Comedy',
  'Drama',
  'Fantasy',
  'Mystery',
  'Romance',
  'Sci-Fi',
  'Slice of Life',
  'Sports',
  'Thriller',
]

export const animeCatalog: Anime[] = [
  { id: 1, title: 'Skyblade Academy', synopsis: PLACEHOLDER_SYNOPSIS, imageUrl: 'https://picsum.photos/seed/anime-1/360/540', genres: ['Action', 'Fantasy'], studio: 'Studio Nova', year: 2018, episodes: 24, malPopularity: 88 },
  { id: 2, title: 'Neon Harbor', synopsis: PLACEHOLDER_SYNOPSIS, imageUrl: 'https://picsum.photos/seed/anime-2/360/540', genres: ['Sci-Fi', 'Thriller'], studio: 'Pulse Works', year: 2021, episodes: 12, malPopularity: 71 },
  { id: 3, title: 'Garden of Kites', synopsis: PLACEHOLDER_SYNOPSIS, imageUrl: 'https://picsum.photos/seed/anime-3/360/540', genres: ['Slice of Life', 'Romance'], studio: 'Mori House', year: 2019, episodes: 13, malPopularity: 63 },
  { id: 4, title: 'Iron Whistle FC', synopsis: PLACEHOLDER_SYNOPSIS, imageUrl: 'https://picsum.photos/seed/anime-4/360/540', genres: ['Sports', 'Drama'], studio: 'Quarterline', year: 2017, episodes: 26, malPopularity: 75 },
  { id: 5, title: 'Binary Witch', synopsis: PLACEHOLDER_SYNOPSIS, imageUrl: 'https://picsum.photos/seed/anime-5/360/540', genres: ['Fantasy', 'Comedy'], studio: 'Hex Frame', year: 2022, episodes: 12, malPopularity: 59 },
  { id: 6, title: 'Metro Eclipse', synopsis: PLACEHOLDER_SYNOPSIS, imageUrl: 'https://picsum.photos/seed/anime-6/360/540', genres: ['Mystery', 'Thriller'], studio: 'Darkwater', year: 2016, episodes: 24, malPopularity: 82 },
  { id: 7, title: 'Starlight Bento Club', synopsis: PLACEHOLDER_SYNOPSIS, imageUrl: 'https://picsum.photos/seed/anime-7/360/540', genres: ['Comedy', 'Slice of Life'], studio: 'Sumi Art', year: 2020, episodes: 12, malPopularity: 54 },
  { id: 8, title: 'Crimson Archive', synopsis: PLACEHOLDER_SYNOPSIS, imageUrl: 'https://picsum.photos/seed/anime-8/360/540', genres: ['Action', 'Mystery'], studio: 'Delta Screen', year: 2015, episodes: 25, malPopularity: 84 },
  { id: 9, title: 'Moonlight Relay', synopsis: PLACEHOLDER_SYNOPSIS, imageUrl: 'https://picsum.photos/seed/anime-9/360/540', genres: ['Sports', 'Romance'], studio: 'Blue Arc', year: 2023, episodes: 12, malPopularity: 44 },
  { id: 10, title: 'Citadel 404', synopsis: PLACEHOLDER_SYNOPSIS, imageUrl: 'https://picsum.photos/seed/anime-10/360/540', genres: ['Sci-Fi', 'Action'], studio: 'Coreline', year: 2024, episodes: 12, malPopularity: 67 },
  { id: 11, title: 'Lotus in Winter', synopsis: PLACEHOLDER_SYNOPSIS, imageUrl: 'https://picsum.photos/seed/anime-11/360/540', genres: ['Drama', 'Romance'], studio: 'Mori House', year: 2014, episodes: 24, malPopularity: 77 },
  { id: 12, title: 'Parallel Street', synopsis: PLACEHOLDER_SYNOPSIS, imageUrl: 'https://picsum.photos/seed/anime-12/360/540', genres: ['Comedy', 'Sci-Fi'], studio: 'Pulse Works', year: 2022, episodes: 13, malPopularity: 48 },
  { id: 13, title: 'Silent Reef', synopsis: PLACEHOLDER_SYNOPSIS, imageUrl: 'https://picsum.photos/seed/anime-13/360/540', genres: ['Adventure', 'Drama'], studio: 'Wavemark', year: 2018, episodes: 24, malPopularity: 69 },
  { id: 14, title: 'Dragon Hour', synopsis: PLACEHOLDER_SYNOPSIS, imageUrl: 'https://picsum.photos/seed/anime-14/360/540', genres: ['Action', 'Adventure'], studio: 'Studio Nova', year: 2019, episodes: 25, malPopularity: 86 },
  { id: 15, title: 'Afterimage District', synopsis: PLACEHOLDER_SYNOPSIS, imageUrl: 'https://picsum.photos/seed/anime-15/360/540', genres: ['Mystery', 'Sci-Fi'], studio: 'Darkwater', year: 2021, episodes: 12, malPopularity: 62 },
  { id: 16, title: 'Petal Parade', synopsis: PLACEHOLDER_SYNOPSIS, imageUrl: 'https://picsum.photos/seed/anime-16/360/540', genres: ['Slice of Life', 'Comedy'], studio: 'Sumi Art', year: 2020, episodes: 12, malPopularity: 57 },
  { id: 17, title: 'Velocity Crown', synopsis: PLACEHOLDER_SYNOPSIS, imageUrl: 'https://picsum.photos/seed/anime-17/360/540', genres: ['Sports', 'Action'], studio: 'Quarterline', year: 2023, episodes: 13, malPopularity: 51 },
  { id: 18, title: 'Lanterns of Orion', synopsis: PLACEHOLDER_SYNOPSIS, imageUrl: 'https://picsum.photos/seed/anime-18/360/540', genres: ['Fantasy', 'Adventure'], studio: 'Hex Frame', year: 2017, episodes: 24, malPopularity: 73 },
  { id: 19, title: 'Hollow Circuit', synopsis: PLACEHOLDER_SYNOPSIS, imageUrl: 'https://picsum.photos/seed/anime-19/360/540', genres: ['Thriller', 'Sci-Fi'], studio: 'Coreline', year: 2024, episodes: 12, malPopularity: 65 },
  { id: 20, title: 'Amber Notebook', synopsis: PLACEHOLDER_SYNOPSIS, imageUrl: 'https://picsum.photos/seed/anime-20/360/540', genres: ['Drama', 'Slice of Life'], studio: 'Blue Arc', year: 2016, episodes: 12, malPopularity: 58 },
]

function scoreAnimeForGenres(anime: Anime, genres: string[]) {
  if (!genres.length) return 0
  return anime.genres.filter((genre) => genres.includes(genre)).length
}

export function searchAnime(query: string) {
  const normalized = query.trim().toLowerCase()
  if (!normalized) {
    return animeCatalog.slice(0, 12)
  }

  return animeCatalog.filter((anime) =>
    anime.title.toLowerCase().includes(normalized),
  )
}

export function getRecommendations(onboarding: UserOnboarding | null): Recommendation[] {
  const baseList = [...animeCatalog]
  const preferredGenres = onboarding?.favoriteGenreIds ?? []

  baseList.sort((a, b) => {
    const genreDiff =
      scoreAnimeForGenres(b, preferredGenres) - scoreAnimeForGenres(a, preferredGenres)
    if (genreDiff !== 0) return genreDiff
    return (b.malPopularity ?? 0) - (a.malPopularity ?? 0)
  })

  const fallbackReason = onboarding
    ? ('onboarding' as const)
    : ('popularity' as const)

  return baseList.slice(0, 16).map((anime, index) => ({
    anime,
    rank: index + 1,
    score: Number((1 - index * 0.038).toFixed(3)),
    isExploratory: (index + 1) % 10 === 0,
    fallbackReason,
  }))
}
