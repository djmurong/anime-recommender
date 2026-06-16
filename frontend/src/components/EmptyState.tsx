interface EmptyStateProps {
  title: string
  description: string
}

export function EmptyState({ title, description }: EmptyStateProps) {
  return (
    <div className="rounded border border-[#bbbbbb] bg-white p-8 text-center">
      <h2 className="text-[14px] font-bold text-[#2e51a2]">{title}</h2>
      <p className="mt-2 text-[12px] text-[#666]">{description}</p>
    </div>
  )
}
