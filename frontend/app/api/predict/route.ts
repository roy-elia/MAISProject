import { NextRequest, NextResponse } from "next/server"

export const maxDuration = 60

export async function POST(request: NextRequest) {
  const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
  let body: unknown
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 })
  }

  try {
    const response = await fetch(`${backendUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
      // Stay under maxDuration so the page gets a clean "waking" answer to retry on.
      signal: AbortSignal.timeout(50000),
    })
    const data = await response.json().catch(() => ({}))
    if (!response.ok) {
      // 502/503 from the host means it is still starting up.
      const waking = response.status === 502 || response.status === 503
      return NextResponse.json(
        { error: waking ? "waking" : typeof data?.detail === "string" ? data.detail : "Prediction failed" },
        { status: waking ? 503 : response.status },
      )
    }
    return NextResponse.json(data)
  } catch (error) {
    console.error("Proxy error:", error)
    return NextResponse.json({ error: "waking" }, { status: 503 })
  }
}
