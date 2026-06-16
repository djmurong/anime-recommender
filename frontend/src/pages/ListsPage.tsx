import { EmptyState } from '../components/EmptyState'

export function ListsPage() {
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-bold text-[#2e51a2]">My Anime List</h1>
      <EmptyState
        title="Your list is empty"
        description="This is a placeholder page for managing your watch history, plan-to-watch, etc."
      />
    </div>
  )
}
