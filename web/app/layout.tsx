import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'sssw | same same swing',
  description: '프로 vs 사용자 스윙 비교/분석 MVP',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <head>
        {/* Google AdSense placeholder */}
        {/* 실제 배포 시 아래 client ID를 환경 변수로 주입 후 스크립트 활성화 */}
        {/* <script async src={`https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=${process.env.NEXT_PUBLIC_ADSENSE_CLIENT}`} crossOrigin="anonymous"></script> */}
      </head>
      <body>
        <div className="max-w-6xl mx-auto px-4 py-6">{children}</div>
      </body>
    </html>
  )
}


