import { animeCatalog, getRecommendations, searchAnime } from '../lib/mock/anime'
import { saveInteraction } from '../lib/storage'
import type { InteractionEvent, Recommendation, UserOnboarding } from '../types'

const USE_MOCK = true
const MOCK_USER_ID = 'guest-user'

function delay<T>(value: T, ms = 220): Promise<T> {
  return new Promise((resolve) => {
    window.setTimeout(() => resolve(value), ms)
  })
}

export async function fetchRecommendations(
  onboarding: UserOnboarding | null,
): Promise<Recommendation[]> {
  if (USE_MOCK) {
    return delay(getRecommendations(onboarding))
  }

  return delay([])
}

export async function fetchAnimeById(id: number) {
  if (USE_MOCK) {
    return delay(animeCatalog.find((anime) => anime.id === id) ?? null)
  }

  return delay(null)
}

export async function searchAnimeCatalog(query: string) {
  if (USE_MOCK) {
    return delay(searchAnime(query))
  }

  return delay([])
}

export async function postInteraction(event: Omit<InteractionEvent, 'userId'>) {
  const payload: InteractionEvent = {
    ...event,
    userId: MOCK_USER_ID,
  }

  if (USE_MOCK) {
    saveInteraction(payload)
    console.log('placeholder POST /interactions', payload)
    return delay({ ok: true })
  }

  return delay({ ok: false })
}
