"use client"
import { useRef, useState } from 'react'

type Props = { videoUrl: string; filename: string }

export function MergeDownload({ videoUrl, filename }: Props) {
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState<string>("")
  const linkRef = useRef<HTMLAnchorElement | null>(null)

  async function download() {
    setBusy(true)
    setStatus('다운로드 준비 중...')
    
    try {
      // 교차 출처에서 a[download]가 무시되는 경우가 있어 Blob으로 강제 다운로드 처리
      const bust = videoUrl.includes('?') ? `${videoUrl}&download=1` : `${videoUrl}?download=1`
      const res = await fetch(bust, { mode: 'cors', cache: 'no-store' })
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }

      const blob = await res.blob()
      const blobUrl = URL.createObjectURL(blob)

      if (!linkRef.current) {
        linkRef.current = document.createElement('a')
        document.body.appendChild(linkRef.current)
      }

      // 응답 Content-Type 기반으로 확장자 보정
      const ct = res.headers.get('content-type') || ''
      const ext = ct.includes('webm') ? 'webm' : ct.includes('mp4') ? 'mp4' : ''
      const finalName = filename && filename.includes('.') ? filename : (ext ? `${filename}.${ext}` : filename)

      linkRef.current.href = blobUrl
      linkRef.current.download = finalName
      linkRef.current.click()

      setStatus('다운로드 완료')
      // 메모리 해제
      setTimeout(() => URL.revokeObjectURL(blobUrl), 1000)
    } catch (error) {
      setStatus('다운로드 실패')
      console.error('다운로드 오류:', error)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-2">
      <button className="w-full bg-black text-white rounded py-2 disabled:opacity-50" disabled={busy} onClick={download}>
        {busy ? '다운로드 중...' : '합성 영상 다운로드'}
      </button>
      {status && <div className="text-xs text-gray-500">{status}</div>}
    </div>
  )
}


