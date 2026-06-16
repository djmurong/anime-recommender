import { EmptyState } from '../components/EmptyState'

export function ImportPage() {
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-bold text-[#2e51a2]">Import Watch History</h1>
      <EmptyState
        title="Sync external accounts"
        description="This is a placeholder page for syncing your history from MyAnimeList or Anilist."
      />
    </div>
  )
}
