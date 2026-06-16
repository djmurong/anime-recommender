import { EmptyState } from '../components/EmptyState'

export function ProfilePage() {
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-bold text-[#2e51a2]">Profile & Settings</h1>
      <EmptyState
        title="Profile settings"
        description="This is a placeholder page for managing your account, avatar, and viewing your genre affinities."
      />
    </div>
  )
}
