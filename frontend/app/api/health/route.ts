import { NextResponse } from 'next/server'

export const maxDuration = 60

// Lets the page wake the backend (a free host that sleeps when idle) on load.
export async function GET() {
  const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  try {
    const response = await fetch(`${backendUrl}/health`, { cache: 'no-store' })
    const data = await response.json()
    return NextResponse.json(data, { status: response.ok ? 200 : 502 })
  } catch {
    return NextResponse.json({ status: 'unreachable' }, { status: 502 })
  }
}
