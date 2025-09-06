"use client"
import { useState } from 'react'

interface UploadAreaProps {
  onUploaded?: (v: { key: string, url: string }) => void
  onFileUpload?: (file: File) => void
  acceptedFileTypes?: string[]
  maxFileSize?: number
  placeholder?: string
}

export function UploadArea({ 
  onUploaded, 
  onFileUpload, 
  acceptedFileTypes = ['video/*'], 
  maxFileSize = 100 * 1024 * 1024, // 100MB
  placeholder = "내 스윙 영상 업로드"
}: UploadAreaProps) {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [dragActive, setDragActive] = useState(false)

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const f = e.dataTransfer.files[0]
      if (f.type.startsWith('video/') && f.size <= maxFileSize) {
        setFile(f)
        setPreview(URL.createObjectURL(f))
        
        // 사용자 비교 분석 페이지용 - 즉시 콜백 호출
        if (onFileUpload) {
          console.log('onFileUpload called with:', f.name) // 디버깅용
          onFileUpload(f)
        }
      }
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] || null
    if (f && f.type.startsWith('video/') && f.size <= maxFileSize) {
      setFile(f)
      setPreview(URL.createObjectURL(f))
      
      // 사용자 비교 분석 페이지용 - 즉시 콜백 호출
      if (onFileUpload) {
        console.log('onFileUpload called with:', f.name) // 디버깅용
        onFileUpload(f)
      }
    }
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h3 className="text-2xl font-bold text-gray-900 mb-2 flex items-center justify-center">
          <svg className="w-6 h-6 text-green-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          {placeholder}
        </h3>
        <p className="text-gray-600">MP4, MOV, AVI 형식의 영상을 업로드해주세요 (최대 {Math.round(maxFileSize / (1024 * 1024))}MB)</p>
      </div>

      {/* 드래그 앤 드롭 영역 */}
      <div
        className={`relative border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-200 ${
          dragActive 
            ? 'border-green-500 bg-green-50' 
            : 'border-gray-300 hover:border-gray-400 bg-gray-50'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept={acceptedFileTypes.join(',')}
          onChange={handleFileSelect}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        
        {!preview ? (
          <div className="space-y-4">
            <div className="w-20 h-20 bg-gray-200 rounded-full flex items-center justify-center mx-auto">
              <svg className="w-10 h-10 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <div>
              <p className="text-lg font-medium text-gray-900 mb-2">
                영상을 여기에 드래그하거나 클릭하여 선택하세요
              </p>
              <p className="text-sm text-gray-500">
                또는 <span className="text-blue-600 font-medium">파일 선택</span> 버튼을 클릭하세요
              </p>
            </div>
            <div className="flex items-center justify-center space-x-4 text-xs text-gray-500">
              <span>📹 MP4</span>
              <span>📹 MOV</span>
              <span>📹 AVI</span>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="relative">
              <video 
                src={preview} 
                className="w-full max-w-md mx-auto rounded-xl shadow-lg" 
                controls
                preload="metadata"
              />
              <button
                onClick={() => {
                  setFile(null)
                  setPreview(null)
                }}
                className="absolute -top-2 -right-2 w-8 h-8 bg-red-500 text-white rounded-full flex items-center justify-center hover:bg-red-600 transition-colors"
              >
                ✕
              </button>
            </div>
            <div className="text-sm text-gray-600">
              <p><strong>파일명:</strong> {file?.name}</p>
              <p><strong>크기:</strong> {(file?.size / (1024 * 1024)).toFixed(2)} MB</p>
            </div>
          </div>
        )}
      </div>

      {/* 업로드 가이드 */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h4 className="font-medium text-blue-900 mb-2">업로드 가이드</h4>
        <ul className="text-sm text-blue-800 space-y-1">
          <li>• 사이드뷰에서 촬영된 스윙 영상을 업로드해주세요</li>
          <li>• 영상은 3-10초 길이로 스윙 동작이 명확하게 보이도록 촬영해주세요</li>
          <li>• 최대 {Math.round(maxFileSize / (1024 * 1024))}MB까지 업로드 가능합니다</li>
          <li>• 업로드된 영상은 분석 후 자동으로 삭제됩니다</li>
        </ul>
      </div>
    </div>
  )
}


