import { EmptyState } from '../components/EmptyState'

export function SearchPage() {
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-bold text-[#2e51a2]">Search Catalog</h1>
      <EmptyState
        title="Search functionality coming soon"
        description="This is a placeholder page for searching the full anime catalog."
      />
    </div>
  )
}
