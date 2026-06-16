import type { InteractionEvent, UserOnboarding } from '../types'

const ONBOARDING_KEY = 'anime-recommender:onboarding'
const INTERACTIONS_KEY = 'anime-recommender:interactions'
const AUTH_KEY = 'anime-recommender:auth'

export function saveAuth(isAuthenticated: boolean) {
  window.localStorage.setItem(AUTH_KEY, JSON.stringify(isAuthenticated))
}

export function readAuth(): boolean {
  const raw = window.localStorage.getItem(AUTH_KEY)
  if (!raw) return false
  try {
    return JSON.parse(raw) as boolean
  } catch {
    return false
  }
}

export function clearAuth() {
  window.localStorage.removeItem(AUTH_KEY)
}

export function saveOnboarding(data: UserOnboarding) {
  window.localStorage.setItem(ONBOARDING_KEY, JSON.stringify(data))
}

export function readOnboarding(): UserOnboarding | null {
  const raw = window.localStorage.getItem(ONBOARDING_KEY)
  if (!raw) return null
  try {
    return JSON.parse(raw) as UserOnboarding
  } catch {
    return null
  }
}

export function clearOnboarding() {
  window.localStorage.removeItem(ONBOARDING_KEY)
}

export function saveInteraction(event: InteractionEvent) {
  const all = readInteractions()
  all.push(event)
  window.localStorage.setItem(INTERACTIONS_KEY, JSON.stringify(all))
}

export function readInteractions(): InteractionEvent[] {
  const raw = window.localStorage.getItem(INTERACTIONS_KEY)
  if (!raw) return []
  try {
    return JSON.parse(raw) as InteractionEvent[]
  } catch {
    return []
  }
}
