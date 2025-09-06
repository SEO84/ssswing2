"use client"
import { useEffect } from 'react'
import { useParams, useRouter } from 'next/navigation'

export default function ProComparisonRedirect() {
  const params = useParams<{ id: string }>()
  const router = useRouter()
  useEffect(() => {
    if (params?.id) router.replace(`/result/${params.id}`)
  }, [params?.id, router])
  return null
}


